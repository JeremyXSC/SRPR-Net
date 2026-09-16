"""Target-domain YOLOv7-seg pretraining for generalized BLO-Inst."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from generalized.training_utils import (
    capture_rng_state,
    create_optimizer,
    create_scheduler,
    restore_rng_state,
)
from segment.generalized_runtime import (
    build_detector,
    configure_detector_for_loss,
    dataset_metadata,
    load_config,
)
from segment.train_generalized import prepare_batch, seed_everything
from utils.general import check_img_size
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.torch_utils import de_parallel, select_device


def _make_loaders(config: Dict[str, Any], data_dict, detector):
    training = config["training"]
    dataset_cfg = config["dataset"]
    stride = max(int(detector.stride.max()), 32)
    common = {
        "imgsz": int(training["imgsz"]),
        "batch_size": int(training.get("batch_size", 1)),
        "stride": stride,
        "single_cls": False,
        "workers": int(training.get("workers", 0)),
        "mask_downsample_ratio": int(dataset_cfg.get("mask_ratio", 4)),
        "overlap_mask": bool(dataset_cfg.get("overlap_masks", True)),
    }
    train_loader, dataset = create_dataloader(
        data_dict["train"],
        hyp=config["hyperparameters"],
        augment=True,
        cache=training.get("cache"),
        rect=False,
        shuffle=True,
        prefix="detector train: ",
        **common,
    )
    val_loader = create_dataloader(
        data_dict["val"],
        rect=False,
        pad=0.0,
        shuffle=False,
        prefix="detector val: ",
        **common,
    )[0]
    return train_loader, val_loader, dataset


@torch.no_grad()
def evaluate_loss(detector, compute_loss, dataloader, device: torch.device) -> float:
    detector.eval()
    values = []
    for batch in dataloader:
        images, targets, _, _, masks = prepare_batch(batch, device)
        _, train_output = detector(images)
        loss, _ = compute_loss(train_output, targets, masks)
        values.append(float(loss.detach().cpu()))
    return float(sum(values) / len(values)) if values else math.inf


def _save_checkpoint(
    path: Path,
    epoch: int,
    best_loss: float,
    config: Dict[str, Any],
    detector,
    optimizer,
    scheduler,
) -> None:
    torch.save(
        {
            "kind": "detector_pretrain",
            "epoch": int(epoch),
            "score": -float(best_loss),
            "best_loss": float(best_loss),
            "config": config,
            "detector": de_parallel(detector).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    pretraining = config.get("pretraining", {})
    seed_everything(int(training.get("seed", 0)))
    device = select_device(
        str(training.get("device", "")),
        batch_size=int(training.get("batch_size", 1)),
    )
    data_dict, names = dataset_metadata(config)
    detector = build_detector(config, len(names), device)
    imgsz = check_img_size(
        int(training["imgsz"]),
        s=max(int(detector.stride.max()), 32),
    )
    training["imgsz"] = int(imgsz)
    config["hyperparameters"]["imgsz"] = int(imgsz)
    train_loader, val_loader, dataset = _make_loaders(config, data_dict, detector)
    configure_detector_for_loss(detector, config["hyperparameters"], names, dataset.labels)
    compute_loss = ComputeLoss(
        detector,
        overlap=bool(config["dataset"].get("overlap_masks", True)),
    )

    optimization = config.get("optimization", {})
    optimizer = create_optimizer(
        detector.parameters(),
        optimization,
        lr=float(optimization.get("detector_lr", 1e-3)),
        weight_decay=float(optimization.get("weight_decay", 5e-4)),
    )
    epochs = int(args.epochs or pretraining.get("epochs", 50))
    scheduler = create_scheduler(optimizer, optimization, epochs)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(args.output or pretraining.get("output", "runs/pretrain_detector"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_detector.pt"
    last_path = output_dir / "last_detector.pt"
    history_path = output_dir / "history.csv"
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        resume_path = last_path if args.resume == "auto" else Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        detector.load_state_dict(checkpoint["detector"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint.get("rng_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", math.inf))
    if start_epoch == 0 or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(
                handle,
                fieldnames=("epoch", "train_loss", "val_loss", "lr", "seconds"),
            ).writeheader()

    patience = int(pretraining.get("patience", 10))
    epochs_without_improvement = 0
    max_grad_norm = float(optimization.get("max_grad_norm", 10.0))
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        detector.train()
        train_losses = []
        progress = tqdm(train_loader, desc="detector epoch {}/{}".format(epoch + 1, epochs))
        for batch in progress:
            images, targets, _, _, masks = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _, train_output = detector(images)
                loss, _ = compute_loss(train_output, targets, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite detector loss at epoch {}.".format(epoch + 1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss="{:.4f}".format(train_losses[-1]))

        val_loss = evaluate_loss(detector, compute_loss, val_loader, device)
        scheduler.step()
        train_loss = sum(train_losses) / len(train_losses) if train_losses else math.inf
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path, epoch, best_loss, config, detector, optimizer, scheduler
            )
        else:
            epochs_without_improvement += 1
        _save_checkpoint(
            last_path, epoch, best_loss, config, detector, optimizer, scheduler
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=row.keys()).writerow(row)
        print(json.dumps(row, ensure_ascii=False))
        if patience > 0 and epochs_without_improvement >= patience:
            print("Early stopping after {} epochs without improvement.".format(patience))
            break

    print("Detector pretraining completed: {}".format(best_path))


if __name__ == "__main__":
    main()

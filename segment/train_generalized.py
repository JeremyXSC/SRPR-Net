"""First-order bi-level training for generalized BLO-Inst.

Inner level (D1): the detector and prompt refiner are fixed while SAM LoRA is
adapted to the current prompt distribution.
Outer level (D2): the updated SAM is fixed while the detector, Bayesian semantic
calibrator and Transformer prompt refiner are optimized against validation-mask
quality and explicit prompt constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import random_split
from torchvision.ops import box_iou
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from generalized.losses import prompt_constraint_loss
from generalized.sam_bridge import (
    instance_masks_for_image,
    sam_loss_for_image,
    targets_for_image,
)
from generalized.training_utils import (
    capture_rng_state,
    create_optimizer,
    create_scheduler,
    restore_rng_state,
)
from segment.generalized_runtime import (
    build_detector,
    build_generalized_modules,
    build_sam,
    configure_detector_for_loss,
    dataset_metadata,
    load_config,
    load_generalized_weights,
    proposal_batch,
    resolve_max_det,
    restore_trainable_parameters,
    set_requires_grad,
)
from segment.val_generalized import run_validation
from utils.dataloaders import InfiniteDataLoader, seed_worker
from utils.general import check_img_size
from utils.segment.dataloaders import LoadImagesAndLabelsAndMasks, create_dataloader
from utils.segment.loss import ComputeLoss
from utils.torch_utils import de_parallel, select_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_config_override(config: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError("--set values must use dotted.key=value syntax.")
    key, raw_value = expression.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("--set key cannot be empty.")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def next_batch(iterator: Iterator, loader) -> Tuple[Iterator, tuple]:
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator)


def make_split_loaders(config: Dict[str, Any], data_dict, detector):
    training = config["training"]
    dataset_cfg = config["dataset"]
    imgsz = int(training["imgsz"])
    batch_size = int(training.get("batch_size", 1))
    workers = int(training.get("workers", 4))
    stride = max(int(detector.stride.max()), 32)
    base_loader, dataset = create_dataloader(
        data_dict["train"],
        imgsz,
        batch_size,
        stride,
        False,
        hyp=config["hyperparameters"],
        augment=True,
        cache=training.get("cache", None),
        rect=False,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix="dataset: ",
        shuffle=True,
        mask_downsample_ratio=int(dataset_cfg.get("mask_ratio", 4)),
        overlap_mask=bool(dataset_cfg.get("overlap_masks", True)),
    )
    del base_loader
    split_ratio = float(training.get("inner_split_ratio", 0.5))
    inner_size = max(1, min(len(dataset) - 1, int(round(len(dataset) * split_ratio))))
    outer_size = len(dataset) - inner_size
    generator = torch.Generator().manual_seed(int(training.get("seed", 0)))
    inner_dataset, outer_dataset = random_split(dataset, [inner_size, outer_size], generator=generator)
    collate = LoadImagesAndLabelsAndMasks.collate_fn
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        sampler=None,
        collate_fn=collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    inner_loader = InfiniteDataLoader(inner_dataset, shuffle=True, **common)
    outer_loader = InfiniteDataLoader(outer_dataset, shuffle=True, **common)
    return inner_loader, outer_loader, dataset


def make_validation_loader(config: Dict[str, Any], data_dict, detector):
    training = config["training"]
    split = str(config.get("validation", {}).get("split", "test"))
    path = data_dict.get(split) or data_dict["val"]
    return create_dataloader(
        path,
        int(training["imgsz"]),
        int(config.get("validation", {}).get("batch_size", 1)),
        max(int(detector.stride.max()), 32),
        False,
        workers=int(training.get("workers", 4)),
        pad=0.0,
        rect=False,
        mask_downsample_ratio=int(config["dataset"].get("mask_ratio", 4)),
        overlap_mask=bool(config["dataset"].get("overlap_masks", True)),
        prefix="validation: ",
    )[0]


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def prompt_losses(
    config: Dict[str, Any],
    images: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    proposals,
    sam,
) -> Dict[str, torch.Tensor]:
    overlap = bool(config["dataset"].get("overlap_masks", True))
    losses = config.get("loss", {})
    min_iou = float(losses.get("min_match_iou", 0.05))
    mask_terms: List[torch.Tensor] = []
    constraint_terms: List[torch.Tensor] = []
    quality_terms: List[torch.Tensor] = []
    sam_quality_terms: List[torch.Tensor] = []
    for image_index, proposal in enumerate(proposals):
        target_indices, target_boxes, target_labels = targets_for_image(
            targets, image_index, tuple(images.shape[-2:])
        )
        target_masks = instance_masks_for_image(masks, target_indices, image_index, overlap)
        sam_result = sam_loss_for_image(
            sam,
            images[image_index],
            proposal["boxes"],
            proposal["labels"],
            target_boxes,
            target_labels,
            target_masks,
            int(config["training"]["imgsz"]),
            min_iou,
            boundary_weight=float(losses.get("boundary_weight", 0.0)),
        )
        mask_terms.append(sam_result["loss"])
        constraint_terms.append(
            prompt_constraint_loss(
                proposal["boxes"],
                proposal["original_boxes"],
                tuple(images.shape[-2:]),
                min_area_ratio=float(losses.get("min_area_ratio", 0.55)),
                max_area_ratio=float(losses.get("max_area_ratio", 1.80)),
                max_center_shift=float(losses.get("max_center_shift", 0.20)),
            )
        )
        pred_index = sam_result["pred_index"]
        target_index = sam_result["target_index"]
        if pred_index.numel() > 0:
            if str(losses.get("quality_target", "box_iou")).lower() == "mask_iou":
                quality_target = sam_result["mask_iou_target"]
            else:
                quality_target = box_iou(
                    proposal["boxes"][pred_index], target_boxes[target_index]
                ).diagonal().detach().clamp(0.0, 1.0)
            quality_terms.append(
                F.mse_loss(proposal["quality"][pred_index], quality_target)
            )
            sam_quality_terms.append(
                F.mse_loss(sam_result["iou_quality"], sam_result["mask_iou_target"])
            )

    reference = images
    return {
        "mask": torch.stack(mask_terms).mean() if mask_terms else _zero(reference),
        "constraint": torch.stack(constraint_terms).mean() if constraint_terms else _zero(reference),
        "quality": torch.stack(quality_terms).mean() if quality_terms else _zero(reference),
        "sam_quality": (
            torch.stack(sam_quality_terms).mean()
            if sam_quality_terms
            else _zero(reference)
        ),
    }


def prepare_batch(batch, device: torch.device):
    images, targets, paths, shapes, masks = batch
    images = images.to(device, non_blocking=True).float() / 255.0
    targets = targets.to(device)
    masks = masks.to(device).float()
    return images, targets, list(paths), shapes, masks


def save_checkpoint(
    path: Path,
    epoch: int,
    score: float,
    config: Dict[str, Any],
    detector,
    sam,
    refiner,
    calibrator,
    metrics: Dict[str, float],
    detector_optimizer=None,
    sam_optimizer=None,
    outer_scheduler=None,
    sam_scheduler=None,
    best_score: float = -math.inf,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "score": score,
            "config": config,
            "detector": de_parallel(detector).state_dict(),
            "sam": sam.state_dict(),
            "refiner": refiner.state_dict(),
            "calibrator": calibrator.state_dict(),
            "metrics": metrics,
            "best_score": float(best_score),
            "detector_optimizer": (
                detector_optimizer.state_dict() if detector_optimizer is not None else None
            ),
            "sam_optimizer": sam_optimizer.state_dict() if sam_optimizer is not None else None,
            "outer_scheduler": (
                outer_scheduler.state_dict() if outer_scheduler is not None else None
            ),
            "sam_scheduler": sam_scheduler.state_dict() if sam_scheduler is not None else None,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from a checkpoint path, or use the output last checkpoint",
    )
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--output", help="override output.directory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable dotted configuration override",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for expression in args.set:
        set_config_override(config, expression)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.output:
        config.setdefault("output", {})["directory"] = args.output
    training = config["training"]
    selection_split = str(config.get("validation", {}).get("split", "val"))
    if selection_split in {"test", "paper_test"}:
        raise ValueError(
            "Training-time model selection must use validation.split=val. "
            "Evaluate test only with segment/val_generalized.py after freezing the configuration."
        )
    seed_everything(int(training.get("seed", 0)))
    device = select_device(str(training.get("device", "")), batch_size=int(training.get("batch_size", 1)))
    data_dict, names = dataset_metadata(config)

    detector = build_detector(config, len(names), device)
    imgsz = check_img_size(int(training["imgsz"]), s=max(int(detector.stride.max()), 32))
    training["imgsz"] = int(imgsz)
    config["hyperparameters"]["imgsz"] = int(imgsz)
    inner_loader, outer_loader, dataset = make_split_loaders(config, data_dict, detector)
    configure_detector_for_loss(detector, config["hyperparameters"], names, dataset.labels)
    compute_loss = ComputeLoss(detector, overlap=bool(config["dataset"].get("overlap_masks", True)))

    sam = build_sam(config, device)
    sam_trainable_names = [name for name, parameter in sam.named_parameters() if parameter.requires_grad]
    foundation, refiner, calibrator, context_resolver, class_features = build_generalized_modules(
        config, names, device
    )
    validation_loader = make_validation_loader(config, data_dict, detector)
    resolve_max_det(config, validation_loader.dataset.labels)

    optimization = config.get("optimization", {})
    detector_optimizer = create_optimizer(
        [
            {"params": detector.parameters(), "lr": float(optimization.get("detector_lr", 1e-4))},
            {"params": refiner.parameters(), "lr": float(optimization.get("refiner_lr", 5e-4))},
            {"params": calibrator.parameters(), "lr": float(optimization.get("prior_lr", 1e-3))},
        ],
        optimization,
        lr=float(optimization.get("detector_lr", 1e-4)),
        weight_decay=float(optimization.get("weight_decay", 1e-4)),
    )
    sam_optimizer = create_optimizer(
        [parameter for parameter in sam.parameters() if parameter.requires_grad],
        optimization,
        lr=float(optimization.get("sam_lr", 2e-4)),
        weight_decay=float(optimization.get("sam_weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 40))
    outer_scheduler = create_scheduler(detector_optimizer, optimization, epochs)
    sam_scheduler = create_scheduler(sam_optimizer, optimization, epochs)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(str(config.get("output", {}).get("directory", "runs/generalized")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_generalized.pt"
    last_path = output_dir / "last_generalized.pt"
    history_fields = [
        "epoch", "inner_mask_loss", "outer_total_loss", "detector_loss", "outer_mask_loss",
        "constraint_loss", "semantic_loss", "quality_loss", "mask_map", "mask_map50",
        "mask_map75", "box_map", "bayes_alpha", "seconds"
    ]
    weights = config.get("loss", {})
    lambda_mask = float(weights.get("lambda_mask", 0.7))
    lambda_constraint = float(weights.get("lambda_constraint", 0.08))
    lambda_semantic = float(weights.get("lambda_semantic", 0.05))
    lambda_quality = float(weights.get("lambda_quality", 0.05))
    lambda_sam_quality = float(weights.get("lambda_sam_quality", 0.0))
    max_grad_norm = float(optimization.get("max_grad_norm", 10.0))
    best_score = -math.inf
    start_epoch = 0
    resume_path = args.resume
    if resume_path:
        if resume_path == "auto":
            resume_path = str(last_path)
        resume_file = Path(resume_path)
        if not resume_file.is_absolute():
            resume_file = ROOT / resume_file
        if not resume_file.exists():
            raise FileNotFoundError("Resume checkpoint was not found: {}".format(resume_file))
        checkpoint = load_generalized_weights(
            str(resume_file), detector, sam, refiner, calibrator, device
        )
        if checkpoint.get("detector_optimizer"):
            detector_optimizer.load_state_dict(checkpoint["detector_optimizer"])
        if checkpoint.get("sam_optimizer"):
            sam_optimizer.load_state_dict(checkpoint["sam_optimizer"])
        if checkpoint.get("outer_scheduler"):
            outer_scheduler.load_state_dict(checkpoint["outer_scheduler"])
        if checkpoint.get("sam_scheduler"):
            sam_scheduler.load_state_dict(checkpoint["sam_scheduler"])
        restore_rng_state(checkpoint.get("rng_state", {}))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", -math.inf)))

    if start_epoch == 0 or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writeheader()
    outer_iterator = iter(outer_loader)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        inner_values: List[float] = []
        outer_values: List[float] = []
        detector_values: List[float] = []
        mask_values: List[float] = []
        constraint_values: List[float] = []
        semantic_values: List[float] = []
        quality_values: List[float] = []
        sam_quality_values: List[float] = []

        progress = tqdm(inner_loader, total=len(inner_loader), desc="epoch {}/{}".format(epoch + 1, epochs))
        for inner_batch in progress:
            # Inner level: update SAM adapter on D1.
            set_requires_grad(detector, False)
            set_requires_grad(refiner, False)
            set_requires_grad(calibrator, False)
            restore_trainable_parameters(sam, sam_trainable_names)
            detector.eval()
            refiner.eval()
            calibrator.eval()
            sam.train()
            images, targets, paths, _, masks = prepare_batch(inner_batch, device)
            with torch.no_grad():
                prediction, _ = detector(images)
                proposals, _, _ = proposal_batch(
                    prediction, images, paths, detector, foundation, refiner, calibrator,
                    context_resolver, class_features, config
                )
            inner_terms = prompt_losses(config, images, targets, masks, proposals, sam)
            inner_loss = (
                inner_terms["mask"]
                + lambda_sam_quality * inner_terms["sam_quality"]
            )
            if inner_loss.requires_grad and torch.isfinite(inner_loss):
                sam_optimizer.zero_grad(set_to_none=True)
                inner_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in sam.parameters() if parameter.requires_grad], max_grad_norm
                )
                sam_optimizer.step()
            inner_values.append(float(inner_loss.detach().cpu()))
            sam_quality_values.append(
                float(inner_terms["sam_quality"].detach().cpu())
            )

            # Outer level: update detector/refiner/prior on a persistent D2 iterator.
            outer_iterator, outer_batch = next_batch(outer_iterator, outer_loader)
            set_requires_grad(detector, True)
            set_requires_grad(refiner, True)
            set_requires_grad(calibrator, True)
            set_requires_grad(sam, False)
            detector.train()
            refiner.train()
            calibrator.train()
            sam.eval()
            outer_images, outer_targets, outer_paths, _, outer_masks = prepare_batch(outer_batch, device)
            detector_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction, train_output = detector(outer_images)
                detector_loss, _ = compute_loss(train_output, outer_targets, outer_masks)
                proposals, _, contexts = proposal_batch(
                    prediction, outer_images, outer_paths, detector, foundation, refiner, calibrator,
                    context_resolver, class_features, config
                )
                terms = prompt_losses(config, outer_images, outer_targets, outer_masks, proposals, sam)
                semantic_loss = calibrator.alignment_loss(prediction, contexts, len(names))
                total_loss = (
                    detector_loss
                    + lambda_mask * terms["mask"]
                    + lambda_constraint * terms["constraint"]
                    + lambda_semantic * semantic_loss
                    + lambda_quality * terms["quality"]
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite outer loss at epoch {}, batch.".format(epoch + 1))
            scaler.scale(total_loss).backward()
            scaler.unscale_(detector_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(detector.parameters()) + list(refiner.parameters()) + list(calibrator.parameters()),
                max_grad_norm,
            )
            scaler.step(detector_optimizer)
            scaler.update()

            outer_values.append(float(total_loss.detach().cpu()))
            detector_values.append(float(detector_loss.detach().cpu()))
            mask_values.append(float(terms["mask"].detach().cpu()))
            constraint_values.append(float(terms["constraint"].detach().cpu()))
            semantic_values.append(float(semantic_loss.detach().cpu()))
            quality_values.append(float(terms["quality"].detach().cpu()))
            progress.set_postfix(
                inner="{:.3f}".format(inner_values[-1]), outer="{:.3f}".format(outer_values[-1])
            )

        outer_scheduler.step()
        sam_scheduler.step()
        validate_interval = int(config.get("validation", {}).get("interval", 1))
        if (epoch + 1) % validate_interval == 0 or epoch + 1 == epochs:
            metrics = run_validation(
                config, detector, sam, foundation, refiner, calibrator, context_resolver,
                class_features, validation_loader, names, device,
                output_dir / "val_epoch_{:03d}".format(epoch + 1),
                save_artifacts=False,
            )
        else:
            metrics = {"mask_map": 0.0, "mask_map50": 0.0, "mask_map75": 0.0, "box_map": 0.0}
        score = float(metrics.get("mask_map", 0.0))
        save_checkpoint(
            last_path,
            epoch,
            score,
            config,
            detector,
            sam,
            refiner,
            calibrator,
            metrics,
            detector_optimizer,
            sam_optimizer,
            outer_scheduler,
            sam_scheduler,
            max(best_score, score),
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                best_path,
                epoch,
                score,
                config,
                detector,
                sam,
                refiner,
                calibrator,
                metrics,
                detector_optimizer,
                sam_optimizer,
                outer_scheduler,
                sam_scheduler,
                best_score,
            )

        row = {
            "epoch": epoch + 1,
            "inner_mask_loss": float(np.mean(inner_values)) if inner_values else 0.0,
            "outer_total_loss": float(np.mean(outer_values)) if outer_values else 0.0,
            "detector_loss": float(np.mean(detector_values)) if detector_values else 0.0,
            "outer_mask_loss": float(np.mean(mask_values)) if mask_values else 0.0,
            "constraint_loss": float(np.mean(constraint_values)) if constraint_values else 0.0,
            "semantic_loss": float(np.mean(semantic_values)) if semantic_values else 0.0,
            "quality_loss": (
                float(np.mean(quality_values + sam_quality_values))
                if quality_values or sam_quality_values
                else 0.0
            ),
            "mask_map": float(metrics.get("mask_map", 0.0)),
            "mask_map50": float(metrics.get("mask_map50", 0.0)),
            "mask_map75": float(metrics.get("mask_map75", 0.0)),
            "box_map": float(metrics.get("box_map", 0.0)),
            "bayes_alpha": float(calibrator.alpha.detach().cpu()),
            "seconds": time.time() - epoch_start,
        }
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writerow(row)
        print(json.dumps(row, ensure_ascii=False))

    print("Training completed. Best mask mAP={:.4f}; checkpoint={}".format(best_score, best_path))


if __name__ == "__main__":
    main()

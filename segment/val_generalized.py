"""Validation for the generalized BLO-Inst model.

The evaluator reports box and mask AP at IoU 0.50:0.95, AP50 and AP75, matching
BLO-Inst's main table. It uses refined boxes as SAM prompts and preserves the
upstream metric implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from generalized.losses import geometric_score_fusion
from generalized.sam_bridge import instance_masks_for_image, targets_for_image
from segment.generalized_runtime import (
    build_detector,
    build_generalized_modules,
    build_sam,
    dataset_metadata,
    detector_predictions,
    load_config,
    load_generalized_weights,
    proposal_batch,
    resolve_max_det,
)
from segment.val import process_batch
from utils.general import check_img_size
from utils.segment.dataloaders import create_dataloader
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.torch_utils import de_parallel, select_device


METRIC_NAMES = [
    "box_precision",
    "box_recall",
    "box_map50",
    "box_map75",
    "box_map",
    "mask_precision",
    "mask_recall",
    "mask_map50",
    "mask_map75",
    "mask_map",
]


def _empty_stats(device: torch.device, niou: int):
    return (
        torch.zeros((0, niou), dtype=torch.bool, device=device),
        torch.zeros((0, niou), dtype=torch.bool, device=device),
        torch.zeros((0,), device=device),
        torch.zeros((0,), device=device),
        torch.zeros((0,), device=device),
    )


def save_qualitative_result(
    image: torch.Tensor,
    target_boxes: torch.Tensor,
    target_masks: torch.Tensor,
    prediction_boxes: torch.Tensor,
    prediction_masks: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a compact GT (green) versus prediction (red) mask/box overlay."""
    canvas = (
        image.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    )
    height, width = canvas.shape[:2]
    overlay = canvas.astype(np.float32)

    def paint_masks(mask_tensor: torch.Tensor, color):
        if mask_tensor.numel() == 0:
            return
        for mask in mask_tensor.detach().float().cpu().numpy():
            resized = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            ) > 0.5
            overlay[resized] = 0.65 * overlay[resized] + 0.35 * np.asarray(color)

    paint_masks(target_masks, (0, 255, 0))
    paint_masks(prediction_masks, (255, 0, 0))
    canvas = np.ascontiguousarray(overlay.clip(0, 255).astype(np.uint8))
    for box in target_boxes.detach().cpu().numpy().astype(int):
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    for box in prediction_boxes.detach().cpu().numpy().astype(int):
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), 2)
    cv2.putText(
        canvas,
        "GT: green | prediction: red",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def run_validation(
    config: Dict[str, Any],
    detector,
    sam,
    foundation,
    refiner,
    calibrator,
    context_resolver,
    class_features,
    dataloader,
    names,
    device: torch.device,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, float]:
    detector.eval()
    sam.eval()
    refiner.eval()
    calibrator.eval()
    foundation.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation_start = time.perf_counter()

    nc = len(names)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = len(iouv)
    stats = []
    seen = 0
    overlap = bool(config.get("dataset", {}).get("overlap_masks", True))
    imgsz = int(config["training"]["imgsz"])
    qualitative_limit = (
        int(config.get("validation", {}).get("qualitative_count", 4))
        if save_artifacts
        else 0
    )
    qualitative_saved = 0

    progress = tqdm(dataloader, desc="validation", leave=False)
    for images, targets, paths, _, masks in progress:
        images = images.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        masks = masks.to(device).float()
        prediction, _ = detector_predictions(detector, images, config)
        proposals, _, _ = proposal_batch(
            prediction,
            images,
            list(paths),
            detector,
            foundation,
            refiner,
            calibrator,
            context_resolver,
            class_features,
            config,
        )

        for image_index, proposal in enumerate(proposals):
            seen += 1
            target_indices, target_boxes, target_labels = targets_for_image(
                targets, image_index, tuple(images.shape[-2:])
            )
            target_masks = instance_masks_for_image(masks, target_indices, image_index, overlap)
            boxes = proposal["boxes"]
            scores = proposal["scores"]
            labels = proposal["labels"]

            if boxes.numel() == 0:
                if output_dir and qualitative_saved < qualitative_limit:
                    save_qualitative_result(
                        images[image_index],
                        target_boxes,
                        target_masks,
                        boxes,
                        target_masks.new_zeros((0, *target_masks.shape[-2:])),
                        output_dir
                        / "qualitative"
                        / "{:03d}_{}.jpg".format(
                            qualitative_saved, Path(paths[image_index]).stem
                        ),
                    )
                    qualitative_saved += 1
                if target_labels.numel() > 0:
                    empty = _empty_stats(device, niou)
                    stats.append((empty[0], empty[1], empty[2], empty[3], target_labels.float()))
                continue

            sam_output = sam(
                images[image_index].unsqueeze(0),
                multimask_output=False,
                image_size=imgsz,
                bbox=boxes,
            )
            pred_masks = (sam_output["low_res_logits"][:, 0] > 0).float()
            if output_dir and qualitative_saved < qualitative_limit:
                save_qualitative_result(
                    images[image_index],
                    target_boxes,
                    target_masks,
                    boxes,
                    pred_masks,
                    output_dir
                    / "qualitative"
                    / "{:03d}_{}.jpg".format(
                        qualitative_saved, Path(paths[image_index]).stem
                    ),
                )
                qualitative_saved += 1
            sam_quality = sam_output.get(
                "iou_predictions",
                scores.new_ones((scores.shape[0], 1)),
            ).reshape(scores.shape[0], -1)[:, 0].clamp(0.0, 1.0)
            quality_source = str(
                config.get("inference", {}).get("mask_quality_source", "mean")
            ).lower()
            if quality_source == "sam":
                mask_quality = sam_quality
            elif quality_source == "refiner":
                mask_quality = proposal["quality"].clamp(0.0, 1.0)
            elif quality_source == "mean":
                mask_quality = 0.5 * (
                    sam_quality + proposal["quality"].clamp(0.0, 1.0)
                )
            else:
                raise ValueError("Unknown mask_quality_source: {}".format(quality_source))
            scores = geometric_score_fusion(
                scores,
                mask_quality,
                float(config.get("inference", {}).get("mask_quality_weight", 0.0)),
            )
            detections = torch.cat(
                (boxes, scores[:, None], labels.float()[:, None]), dim=1
            )
            labels_for_metric = torch.cat((target_labels.float()[:, None], target_boxes), dim=1)
            if target_labels.numel() > 0:
                correct_boxes = process_batch(detections, labels_for_metric, iouv)
                correct_masks = process_batch(
                    detections,
                    labels_for_metric,
                    iouv,
                    pred_masks=pred_masks,
                    gt_masks=target_masks,
                    overlap=False,
                    masks=True,
                )
            else:
                correct_boxes = torch.zeros((boxes.shape[0], niou), dtype=torch.bool, device=device)
                correct_masks = torch.zeros_like(correct_boxes)
            stats.append((correct_masks, correct_boxes, scores, labels.float(), target_labels.float()))

    metrics = Metrics()
    if stats:
        concatenated = [torch.cat(values, 0).cpu().numpy() for values in zip(*stats)]
        if len(concatenated[0]) and concatenated[0].any():
            result = ap_per_class_box_and_mask(
                *concatenated,
                plot=bool(output_dir) and save_artifacts,
                save_dir=output_dir or Path("."),
                names={index: name for index, name in enumerate(names)},
            )
            metrics.update(result)
        targets_per_class = np.bincount(concatenated[4].astype(int), minlength=nc)
    else:
        targets_per_class = np.zeros(nc, dtype=int)

    values = metrics.mean_results()
    result_dict = {key: float(value) for key, value in zip(METRIC_NAMES, values)}
    result_dict["images"] = int(seen)
    result_dict["instances"] = int(targets_per_class.sum())
    result_dict["bayes_alpha"] = float(calibrator.alpha.detach().cpu())
    result_dict["max_det"] = int(
        config.get("inference", {}).get("_resolved_max_det", 300)
    )
    elapsed = time.perf_counter() - validation_start
    result_dict["inference_seconds"] = float(elapsed)
    result_dict["fps"] = float(seen / elapsed) if elapsed > 0 else 0.0
    result_dict["parameters"] = int(
        sum(parameter.numel() for module in (detector, sam, refiner, calibrator)
            for parameter in module.parameters())
    )
    result_dict["trainable_parameters"] = int(
        sum(parameter.numel() for module in (detector, sam, refiner, calibrator)
            for parameter in module.parameters() if parameter.requires_grad)
    )
    result_dict["peak_vram_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        if device.type == "cuda"
        else 0.0
    )
    per_class = {}
    metric_positions = {
        int(class_id): position
        for position, class_id in enumerate(metrics.ap_class_index)
    }
    for class_id, name in enumerate(names):
        if class_id in metric_positions:
            values_for_class = metrics.class_result(metric_positions[class_id])
            per_class[str(name)] = {
                key: float(value)
                for key, value in zip(METRIC_NAMES, values_for_class)
            }
        else:
            per_class[str(name)] = {key: 0.0 for key in METRIC_NAMES}
    result_dict["per_class"] = per_class

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            flat_result = {
                key: value for key, value in result_dict.items()
                if not isinstance(value, (dict, list))
            }
            writer = csv.DictWriter(handle, fieldnames=list(flat_result.keys()))
            writer.writeheader()
            writer.writerow(flat_result)
    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="generalized experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="best_generalized.pt")
    parser.add_argument("--output", default="runs/generalized_val")
    parser.add_argument("--split", choices=("val", "test", "paper_test"))
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--mask-quality-weight", type=float)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="skip PR curves and qualitative images (used during grid search)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.split:
        config.setdefault("validation", {})["split"] = args.split
    if args.nms_iou is not None:
        config.setdefault("inference", {})["nms_iou"] = args.nms_iou
    if args.max_det is not None:
        config.setdefault("inference", {})["max_det"] = args.max_det
    if args.mask_quality_weight is not None:
        config.setdefault("inference", {})["mask_quality_weight"] = args.mask_quality_weight
    if args.tta:
        config.setdefault("inference", {}).setdefault("tta", {})["enabled"] = True
    device = select_device(str(config["training"].get("device", "")), batch_size=1)
    data_dict, names = dataset_metadata(config)
    detector = build_detector(config, len(names), device)
    sam = build_sam(config, device)
    foundation, refiner, calibrator, context_resolver, class_features = build_generalized_modules(
        config, names, device
    )
    load_generalized_weights(args.checkpoint, detector, sam, refiner, calibrator, device)

    imgsz = check_img_size(int(config["training"]["imgsz"]), s=max(int(detector.stride.max()), 32))
    split = str(config.get("validation", {}).get("split", "test"))
    data_path = data_dict.get(split) or data_dict["val"]
    dataloader = create_dataloader(
        data_path,
        imgsz,
        int(config.get("validation", {}).get("batch_size", 1)),
        max(int(detector.stride.max()), 32),
        False,
        workers=int(config["training"].get("workers", 4)),
        pad=0.0,
        rect=False,
        mask_downsample_ratio=int(config["dataset"].get("mask_ratio", 4)),
        overlap_mask=bool(config["dataset"].get("overlap_masks", True)),
        prefix="test: ",
    )[0]
    resolve_max_det(config, dataloader.dataset.labels)
    result = run_validation(
        config,
        detector,
        sam,
        foundation,
        refiner,
        calibrator,
        context_resolver,
        class_features,
        dataloader,
        names,
        device,
        Path(args.output),
        save_artifacts=not args.no_artifacts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

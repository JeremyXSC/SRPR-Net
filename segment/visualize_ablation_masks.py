from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
from utils.general import check_img_size
from utils.segment.dataloaders import create_dataloader
from utils.torch_utils import select_device


DEFAULT_RUNS = {
    "Baseline": {
        "config": "runs/ablations/pennfudan/A0_blo_baseline/resolved_config.yaml",
        "checkpoint": "runs/ablations/pennfudan/A0_blo_baseline/best_generalized.pt",
    },
    "+ Semantic": {
        "config": "runs/ablations/pennfudan/A1_clip_refiner/resolved_config.yaml",
        "checkpoint": "runs/ablations/pennfudan/A1_clip_refiner/best_generalized.pt",
    },
    "Full SRPR-Net": {
        "config": "runs/ablations/pennfudan/A2_clip_transformer/resolved_config.yaml",
        "checkpoint": "runs/ablations/pennfudan/A2_clip_transformer/best_generalized.pt",
    },
}

GT_COLOR = (35, 205, 85)
PRED_COLOR = (240, 92, 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a five-panel qualitative ablation comparison."
    )
    parser.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0")
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Index of the image in the configured evaluation split.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Override the config split, e.g. val or test.",
    )
    parser.add_argument(
        "--candidate-conf",
        type=float,
        default=0.001,
        help="Minimum proposal score before GT-to-prediction matching.",
    )
    parser.add_argument(
        "--min-match-iou",
        type=float,
        default=0.10,
        help="Minimum box IoU required to regard a prediction as matched.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Threshold applied to SAM mask logits.",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.38)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--title-height", type=int, default=58)
    parser.add_argument("--panel-gap", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="runs/qualitative_pennfudan",
    )
    parser.add_argument(
        "--a0-config", default=DEFAULT_RUNS["Baseline"]["config"]
    )
    parser.add_argument(
        "--a0-checkpoint", default=DEFAULT_RUNS["Baseline"]["checkpoint"]
    )
    parser.add_argument(
        "--a1-config", default=DEFAULT_RUNS["+ Semantic"]["config"]
    )
    parser.add_argument(
        "--a1-checkpoint", default=DEFAULT_RUNS["+ Semantic"]["checkpoint"]
    )
    parser.add_argument(
        "--a2-config", default=DEFAULT_RUNS["Full SRPR-Net"]["config"]
    )
    parser.add_argument(
        "--a2-checkpoint", default=DEFAULT_RUNS["Full SRPR-Net"]["checkpoint"]
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_inputs(runs: Dict[str, Dict[str, str]]) -> None:
    missing: List[str] = []
    for run in runs.values():
        for key in ("config", "checkpoint"):
            path = resolve_path(run[key])
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_bundle(
    run: Dict[str, str], device: torch.device
) -> Dict[str, Any]:
    config_path = resolve_path(run["config"])
    checkpoint_path = resolve_path(run["checkpoint"])
    config = load_config(str(config_path))
    config["training"]["device"] = str(device)

    _, names = dataset_metadata(config)
    detector = build_detector(config, len(names), device)
    sam = build_sam(config, device)
    foundation, refiner, calibrator, context_resolver, class_features = (
        build_generalized_modules(config, names, device)
    )
    load_generalized_weights(
        str(checkpoint_path), detector, sam, refiner, calibrator, device
    )

    for module in (detector, sam, foundation, refiner, calibrator):
        module.eval()

    return {
        "config": config,
        "names": names,
        "detector": detector,
        "sam": sam,
        "foundation": foundation,
        "refiner": refiner,
        "calibrator": calibrator,
        "context_resolver": context_resolver,
        "class_features": class_features,
    }


def build_loader(
    config: Dict[str, Any], detector: torch.nn.Module, split_override: str
):
    data_dict, _ = dataset_metadata(config)
    stride = max(int(detector.stride.max()), 32)
    image_size = check_img_size(int(config["training"]["imgsz"]), s=stride)
    split = split_override or str(config.get("validation", {}).get("split", "test"))
    data_path = data_dict.get(split)
    if not data_path:
        raise KeyError(
            f"Split '{split}' is unavailable. Dataset keys: {sorted(data_dict)}"
        )

    loader = create_dataloader(
        data_path,
        image_size,
        1,
        stride,
        False,
        workers=0,
        pad=0.0,
        rect=False,
        mask_downsample_ratio=int(config["dataset"].get("mask_ratio", 4)),
        overlap_mask=bool(config["dataset"].get("overlap_masks", True)),
        prefix="Qualitative comparison: ",
    )[0]
    resolve_max_det(config, loader.dataset.labels)
    return loader


def get_selected_batch(loader, image_index: int):
    if image_index < 0:
        raise ValueError("--image-index must be non-negative.")
    for index, batch in enumerate(loader):
        if index == image_index:
            return batch
    raise IndexError(f"Image index {image_index} is outside the dataset.")


def tensor_to_rgb_u8(image: torch.Tensor) -> np.ndarray:
    array = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(array)


def resize_binary_masks(
    masks: torch.Tensor, height: int, width: int
) -> np.ndarray:
    if masks.numel() == 0:
        return np.zeros((0, height, width), dtype=bool)
    resized = F.interpolate(
        masks[:, None].float(),
        size=(height, width),
        mode="nearest",
    )[:, 0]
    return np.ascontiguousarray(resized.detach().cpu().numpy() > 0.5)


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def draw_overlay(
    rgb_image: np.ndarray,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    color: Tuple[int, int, int],
    alpha: float,
    line_width: int,
) -> np.ndarray:
    canvas = np.ascontiguousarray(rgb_image.copy())
    height, width = canvas.shape[:2]
    binary_masks = resize_binary_masks(masks, height, width)
    overlay = canvas.astype(np.float32)
    color_array = np.asarray(color, dtype=np.float32)

    # Paint each mask once. Overlapping masks do not become progressively darker.
    union = binary_masks.any(axis=0) if len(binary_masks) else None
    if union is not None and union.any():
        overlay[union] = (1.0 - alpha) * overlay[union] + alpha * color_array
    canvas = np.ascontiguousarray(np.clip(overlay, 0, 255).astype(np.uint8))

    for box in boxes.detach().cpu().numpy():
        x1, y1, x2, y2 = clip_box(box, width, height)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                line_width,
                cv2.LINE_AA,
            )
    return canvas


def pairwise_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_size = (first[:, 2:] - first[:, :2]).clamp_min(0)
    second_size = (second[:, 2:] - second[:, :2]).clamp_min(0)
    first_area = first_size[:, 0] * first_size[:, 1]
    second_area = second_size[:, 0] * second_size[:, 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


def greedy_instance_matching(
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    min_iou: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match every GT to at most one same-class prediction."""
    if target_boxes.numel() == 0 or candidate_boxes.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=target_boxes.device)
        return empty, empty, target_boxes.new_zeros((0,))

    ious = pairwise_box_iou(target_boxes, candidate_boxes)
    same_class = target_labels[:, None] == candidate_labels[None, :]
    working = torch.where(same_class, ious, torch.full_like(ious, -1.0))
    matches = []
    for _ in range(min(working.shape)):
        flat_index = int(torch.argmax(working).item())
        target_index = flat_index // working.shape[1]
        candidate_index = flat_index % working.shape[1]
        value = working[target_index, candidate_index]
        if float(value) < float(min_iou):
            break
        matches.append((target_index, candidate_index, value))
        working[target_index, :] = -1.0
        working[:, candidate_index] = -1.0

    matches.sort(key=lambda item: item[0])
    if not matches:
        empty = torch.empty(0, dtype=torch.long, device=target_boxes.device)
        return empty, empty, target_boxes.new_zeros((0,))
    gt_indices = torch.tensor(
        [item[0] for item in matches], dtype=torch.long, device=target_boxes.device
    )
    pred_indices = torch.tensor(
        [item[1] for item in matches], dtype=torch.long, device=target_boxes.device
    )
    return gt_indices, pred_indices, torch.stack([item[2] for item in matches])


def matched_mask_iou(
    predicted_masks: torch.Tensor, target_masks: torch.Tensor
) -> torch.Tensor:
    resized_targets = F.interpolate(
        target_masks[:, None].float(),
        size=predicted_masks.shape[-2:],
        mode="nearest",
    )[:, 0] > 0.5
    predicted = predicted_masks > 0.5
    intersection = (predicted & resized_targets).flatten(1).sum(1).float()
    union = (predicted | resized_targets).flatten(1).sum(1).float()
    return intersection / union.clamp_min(1.0)


@torch.no_grad()
def infer_matched_masks(
    bundle: Dict[str, Any],
    images: torch.Tensor,
    paths: Sequence[str],
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    candidate_conf: float,
    min_match_iou: float,
    mask_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    config = bundle["config"]
    prediction, _ = detector_predictions(bundle["detector"], images, config)
    proposals, _, _ = proposal_batch(
        prediction,
        images,
        list(paths),
        bundle["detector"],
        bundle["foundation"],
        bundle["refiner"],
        bundle["calibrator"],
        bundle["context_resolver"],
        bundle["class_features"],
        config,
    )
    proposal = proposals[0]
    keep = proposal["scores"] >= float(candidate_conf)
    candidate_boxes = proposal["boxes"][keep]
    candidate_scores = proposal["scores"][keep]
    candidate_labels = proposal["labels"][keep]
    gt_indices, pred_indices, box_ious = greedy_instance_matching(
        target_boxes,
        target_labels,
        candidate_boxes,
        candidate_labels,
        min_match_iou,
    )
    boxes = candidate_boxes[pred_indices]
    scores = candidate_scores[pred_indices]

    if boxes.numel() == 0:
        mask_height = images.shape[-2] // int(config["dataset"].get("mask_ratio", 4))
        mask_width = images.shape[-1] // int(config["dataset"].get("mask_ratio", 4))
        empty_masks = images.new_zeros((0, mask_height, mask_width))
        return boxes, empty_masks, {
            "ground_truth_count": int(target_boxes.shape[0]),
            "candidate_count": int(candidate_boxes.shape[0]),
            "matched_count": 0,
            "missed_gt_indices": list(range(int(target_boxes.shape[0]))),
            "matches": [],
        }

    sam_output = bundle["sam"](
        images[0].unsqueeze(0),
        multimask_output=False,
        image_size=int(config["training"]["imgsz"]),
        bbox=boxes,
    )
    masks = (sam_output["low_res_logits"][:, 0] > mask_threshold).float()
    mask_ious = matched_mask_iou(masks, target_masks[gt_indices])
    matched_gt = set(gt_indices.detach().cpu().tolist())
    report = {
        "ground_truth_count": int(target_boxes.shape[0]),
        "candidate_count": int(candidate_boxes.shape[0]),
        "matched_count": int(len(gt_indices)),
        "missed_gt_indices": [
            index for index in range(int(target_boxes.shape[0])) if index not in matched_gt
        ],
        "matches": [
            {
                "gt_index": int(gt_indices[index]),
                "score": round(float(scores[index]), 6),
                "box_iou": round(float(box_ious[index]), 6),
                "mask_iou": round(float(mask_ious[index]), 6),
            }
            for index in range(len(gt_indices))
        ],
    }
    return boxes, masks, report


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def add_title(panel: np.ndarray, title: str, title_height: int) -> np.ndarray:
    image = Image.fromarray(panel)
    titled = Image.new(
        "RGB", (image.width, image.height + title_height), color=(255, 255, 255)
    )
    titled.paste(image, (0, title_height))
    draw = ImageDraw.Draw(titled)
    font = get_font(max(16, int(title_height * 0.43)))
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    x = (image.width - (right - left)) // 2
    y = (title_height - (bottom - top)) // 2 - top
    draw.text((x, y), title, fill=(20, 20, 20), font=font)
    return np.asarray(titled)


def join_panels(panels: Sequence[np.ndarray], gap: int) -> np.ndarray:
    max_height = max(panel.shape[0] for panel in panels)
    total_width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.full((max_height, total_width, 3), 255, dtype=np.uint8)
    x = 0
    for panel in panels:
        canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return canvas


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Failed to save image: {path}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.candidate_conf <= 1.0:
        raise ValueError("--candidate-conf must be between 0 and 1.")
    if not 0.0 <= args.min_match_iou <= 1.0:
        raise ValueError("--min-match-iou must be between 0 and 1.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1.")

    runs = {
        "Baseline": {"config": args.a0_config, "checkpoint": args.a0_checkpoint},
        "+ Semantic": {"config": args.a1_config, "checkpoint": args.a1_checkpoint},
        "Full SRPR-Net": {"config": args.a2_config, "checkpoint": args.a2_checkpoint},
    }
    validate_inputs(runs)
    device = select_device(args.device, batch_size=1)

    # A0 supplies the shared data loader; all configurations receive this exact batch.
    bundles: Dict[str, Dict[str, Any]] = {}
    bundles["Baseline"] = load_bundle(runs["Baseline"], device)
    loader = build_loader(bundles["Baseline"]["config"], bundles["Baseline"]["detector"], args.split)
    selected_batch = get_selected_batch(loader, args.image_index)
    images, targets, paths, _, packed_masks = selected_batch
    images = images.to(device).float() / 255.0
    targets = targets.to(device)
    packed_masks = packed_masks.to(device).float()

    config = bundles["Baseline"]["config"]
    overlap = bool(config["dataset"].get("overlap_masks", True))
    target_indices, target_boxes, target_labels = targets_for_image(
        targets, 0, tuple(images.shape[-2:])
    )
    target_masks = instance_masks_for_image(
        packed_masks, target_indices, 0, overlap
    )
    if target_boxes.numel() == 0:
        raise RuntimeError("The selected image has no ground-truth instances.")

    rgb_image = tensor_to_rgb_u8(images[0])
    raw_panels: List[Tuple[str, np.ndarray]] = [
        ("Original Image", rgb_image),
        (
            "Ground Truth",
            draw_overlay(
                rgb_image,
                target_boxes,
                target_masks,
                GT_COLOR,
                args.mask_alpha,
                args.line_width,
            ),
        ),
    ]

    reports: Dict[str, Any] = {
        "input_image": str(paths[0]),
        "image_index": int(args.image_index),
        "candidate_conf": float(args.candidate_conf),
        "min_match_iou": float(args.min_match_iou),
        "models": {},
    }
    for title in ("Baseline", "+ Semantic", "Full SRPR-Net"):
        if title not in bundles:
            bundles[title] = load_bundle(runs[title], device)
        boxes, masks, report = infer_matched_masks(
            bundles[title],
            images,
            paths,
            target_boxes,
            target_labels,
            target_masks,
            args.candidate_conf,
            args.min_match_iou,
            args.mask_threshold,
        )
        panel = draw_overlay(
            rgb_image,
            boxes,
            masks,
            PRED_COLOR,
            args.mask_alpha,
            args.line_width,
        )
        raw_panels.append((title, panel))
        reports["models"][title] = report
        print(
            f"{title}: matched {report['matched_count']}/"
            f"{report['ground_truth_count']} GT instances"
        )
        for match in report["matches"]:
            print(
                "  GT {gt_index}: score={score:.3f}, box IoU={box_iou:.3f}, "
                "mask IoU={mask_iou:.3f}".format(**match)
            )
        if report["missed_gt_indices"]:
            print(f"  Missed GT indices: {report['missed_gt_indices']}")

    output_dir = resolve_path(args.output_dir)
    stem = Path(paths[0]).stem
    filenames = (
        "01_original.png",
        "02_ground_truth.png",
        "03_baseline.png",
        "04_semantic.png",
        "05_full_srpr_net.png",
    )
    titled_panels: List[np.ndarray] = []
    for filename, (title, panel) in zip(filenames, raw_panels):
        save_rgb(output_dir / stem / filename, panel)
        titled_panels.append(add_title(panel, title, args.title_height))

    comparison = join_panels(titled_panels, args.panel_gap)
    comparison_path = output_dir / f"{stem}_five_panel.png"
    save_rgb(comparison_path, comparison)
    report_path = output_dir / f"{stem}_matching_metrics.json"
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Input image: {paths[0]}")
    print(f"Individual panels: {(output_dir / stem).resolve()}")
    print(f"Five-panel figure: {comparison_path.resolve()}")
    print(f"Matching report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

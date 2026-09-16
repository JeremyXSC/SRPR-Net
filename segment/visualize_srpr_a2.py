"""Create three paper-ready visualizations for PennFudan SRPR-Net A2.

Place this file at: blo_generalized/segment/visualize_srpr_a2.py
Run it from the blo_generalized project root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from generalized.losses import geometric_score_fusion
from segment.generalized_runtime import (
    build_detector,
    build_generalized_modules,
    build_sam,
    detector_predictions,
    load_config,
    load_generalized_weights,
    proposal_batch,
    resolve_max_det,
    resolve_project_path,
)
from utils.augmentations import letterbox
from utils.general import check_img_size, imread, imwrite
from utils.segment.general import scale_masks
from utils.torch_utils import select_device


RED = (0, 0, 255)  # OpenCV uses BGR.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SRPR-Net A2 on ped.png")
    parser.add_argument(
        "--config",
        default="runs/ablations/pennfudan/A2_clip_transformer/resolved_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/ablations/pennfudan/A2_clip_transformer/best_generalized.pt",
    )
    parser.add_argument("--source", default="ped.png")
    parser.add_argument("--sam-checkpoint", default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--output", default="runs/framework_visualization/pennfudan_A2")
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu")
    parser.add_argument("--score-thres", type=float, default=0.08)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--box-thickness", type=int, default=2)
    return parser.parse_args()


def load_class_names(config: Dict[str, Any]) -> list[str]:
    data_value = config.get("dataset", {}).get("yaml")
    data_path = resolve_project_path(data_value, config["_project_root"])
    if not data_path or not Path(data_path).is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    data = yaml.safe_load(Path(data_path).read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda key: int(key))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("Dataset YAML must define class names.")


def prepare_image(
    image_bgr: np.ndarray, imgsz: int, stride: int
) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[float, float]]:
    padded, ratio, pad = letterbox(
        image_bgr,
        new_shape=(imgsz, imgsz),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=stride,
    )
    rgb_chw = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_chw)).float().div_(255.0)
    return tensor.unsqueeze(0), ratio, pad


def scale_boxes_to_original(
    boxes: torch.Tensor,
    original_shape: Sequence[int],
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
) -> np.ndarray:
    output = boxes.detach().clone().float()
    if output.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    output[:, [0, 2]] = (output[:, [0, 2]] - float(pad[0])) / float(ratio[0])
    output[:, [1, 3]] = (output[:, [1, 3]] - float(pad[1])) / float(ratio[1])
    height, width = int(original_shape[0]), int(original_shape[1])
    output[:, 0].clamp_(0, width)
    output[:, 2].clamp_(0, width)
    output[:, 1].clamp_(0, height)
    output[:, 3].clamp_(0, height)
    return output.cpu().numpy()


def final_scores(
    proposal: Dict[str, torch.Tensor],
    sam_output: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> torch.Tensor:
    detector_scores = proposal["scores"]
    sam_quality = sam_output.get(
        "iou_predictions", detector_scores.new_ones((len(detector_scores), 1))
    ).reshape(len(detector_scores), -1)[:, 0].clamp(0.0, 1.0)
    refiner_quality = proposal["quality"].clamp(0.0, 1.0)
    source = str(config.get("inference", {}).get("mask_quality_source", "mean")).lower()
    if source == "sam":
        mask_quality = sam_quality
    elif source == "refiner":
        mask_quality = refiner_quality
    elif source == "mean":
        mask_quality = 0.5 * (sam_quality + refiner_quality)
    else:
        raise ValueError(f"Unknown mask_quality_source: {source}")
    return geometric_score_fusion(
        detector_scores,
        mask_quality,
        float(config.get("inference", {}).get("mask_quality_weight", 0.0)),
    )


def draw_red_boxes(image: np.ndarray, boxes: np.ndarray, thickness: int) -> np.ndarray:
    result = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            result,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            RED,
            thickness,
            cv2.LINE_AA,
        )
    return result


def draw_red_masks_and_boxes(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    alpha: float,
    thickness: int,
) -> np.ndarray:
    result = image.copy()
    occupied = np.zeros(image.shape[:2], dtype=bool)
    for mask in masks:
        visible = np.asarray(mask, dtype=bool) & ~occupied
        if not visible.any():
            continue
        occupied |= visible
        blended = (
            (1.0 - alpha) * result[visible].astype(np.float32)
            + alpha * np.asarray(RED, dtype=np.float32)
        )
        result[visible] = blended.clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, RED, 1, cv2.LINE_AA)
    return draw_red_boxes(result, boxes, thickness)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_thres <= 1.0:
        raise ValueError("--score-thres must be between 0 and 1.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1.")

    source = (ROOT / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    checkpoint_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    sam_path = (ROOT / args.sam_checkpoint).resolve() if not Path(args.sam_checkpoint).is_absolute() else Path(args.sam_checkpoint)
    output_dir = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    for label, path in (
        ("source image", source),
        ("A2 config", config_path),
        ("A2 checkpoint", checkpoint_path),
        ("SAM checkpoint", sam_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = load_config(str(config_path))
    config.setdefault("training", {})["device"] = args.device
    config.setdefault("sam", {})["checkpoint"] = str(sam_path)
    # The A2 generalized checkpoint restores its own detector state.
    config.setdefault("model", {})["weights"] = ""

    device = select_device(args.device, batch_size=1)
    names = load_class_names(config)
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

    stride = max(int(detector.stride.max()), 32)
    imgsz = int(check_img_size(int(config["training"]["imgsz"]), s=stride))
    resolve_max_det(config)

    image_bgr = imread(str(source))
    if image_bgr is None:
        raise OSError(f"Could not read source image: {source}")
    image_tensor, ratio, pad = prepare_image(image_bgr, imgsz, stride)
    image_tensor = image_tensor.to(device, non_blocking=True)

    prediction, _ = detector_predictions(detector, image_tensor, config)
    proposals, _, _ = proposal_batch(
        prediction,
        image_tensor,
        [str(source)],
        detector,
        foundation,
        refiner,
        calibrator,
        context_resolver,
        class_features,
        config,
    )
    proposal = proposals[0]
    if "original_boxes" not in proposal:
        raise KeyError(
            "proposal_batch() did not return 'original_boxes'. "
            "Check segment/generalized_runtime.py before changing the inference path."
        )

    initial_boxes = proposal["original_boxes"]
    refined_boxes = proposal["boxes"]
    if refined_boxes.numel() == 0:
        keep = torch.empty((0,), dtype=torch.long, device=device)
        mask_logits = image_tensor.new_zeros((0, imgsz, imgsz))
    else:
        sam_output = sam(
            image_tensor,
            multimask_output=False,
            image_size=imgsz,
            bbox=refined_boxes,
        )
        scores = final_scores(proposal, sam_output, config)
        keep = torch.nonzero(scores >= args.score_thres, as_tuple=False).flatten()
        if keep.numel():
            keep = keep[torch.argsort(scores[keep], descending=True)]
        mask_logits = sam_output["masks"][:, 0]

    # Use the same final selection for both views so boxes correspond one-to-one.
    initial_selected = initial_boxes[keep]
    refined_selected = refined_boxes[keep]
    initial_original = scale_boxes_to_original(
        initial_selected, image_bgr.shape, ratio, pad
    )
    refined_original = scale_boxes_to_original(
        refined_selected, image_bgr.shape, ratio, pad
    )

    if keep.numel():
        selected_logits = mask_logits[keep].detach().cpu().float().numpy()
        mask_hwc = np.moveaxis(selected_logits, 0, -1)
        scaled_logits = scale_masks(
            image_tensor.shape[-2:],
            mask_hwc,
            image_bgr.shape,
            ratio_pad=(ratio, pad),
        )
        masks = np.moveaxis(scaled_logits > 0.0, -1, 0)
    else:
        masks = np.zeros(
            (0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool
        )

    original_view = image_bgr.copy()
    initial_view = draw_red_boxes(
        image_bgr, initial_original, args.box_thickness
    )
    final_view = draw_red_masks_and_boxes(
        image_bgr,
        masks,
        refined_original,
        args.mask_alpha,
        args.box_thickness,
    )

    save_png(output_dir / "01_original.png", original_view)
    save_png(output_dir / "02_yolo_initial_boxes.png", initial_view)
    save_png(output_dir / "03_srpr_a2_sam_final.png", final_view)

    print(f"Detected/kept instances: {len(keep)}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()

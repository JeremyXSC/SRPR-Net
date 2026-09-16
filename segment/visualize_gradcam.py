from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from generalized.sam_bridge import (
    instance_masks_for_image,
    targets_for_image,
)
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


RUNS = {
    "A1 + CLIP": {
        "config": Path(
            "runs/ablations/pennfudan/"
            "A1_clip_refiner/resolved_config.yaml"
        ),
        "checkpoint": Path(
            "runs/ablations/pennfudan/"
            "A1_clip_refiner/best_generalized.pt"
        ),
        "use_cam": True,
    },
    "A2 + Transformer": {
        "config": Path(
            "runs/ablations/pennfudan/"
            "A2_clip_transformer/resolved_config.yaml"
        ),
        "checkpoint": Path(
            "runs/ablations/pennfudan/"
            "A2_clip_transformer/best_generalized.pt"
        ),
        "use_cam": True,
    },
}


class ScalarOutputTarget:
    """Select the scalar output returned by the visualization wrapper."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output.reshape(-1)[0]


class SAMInstanceCAMWrapper(torch.nn.Module):
    """Explain all matched prompt-conditioned SAM instance masks."""

    def __init__(
        self,
        sam,
        boxes: torch.Tensor,
        target_indices: torch.Tensor,
        image_size: int,
        target_regions: torch.Tensor,
    ) -> None:
        super().__init__()

        self.sam = sam
        self.register_buffer("boxes", boxes.detach().clone())
        self.register_buffer(
            "target_indices",
            target_indices.detach().long().clone(),
        )
        self.register_buffer(
            "target_regions",
            target_regions.detach().bool().clone(),
        )

        self.image_size = int(image_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sam_output = self.sam(
            images,
            multimask_output=False,
            image_size=self.image_size,
            bbox=self.boxes,
        )

        all_logits = sam_output["low_res_logits"][:, 0]
        regions = self.target_regions
        if regions.shape[-2:] != all_logits.shape[-2:]:
            regions = F.interpolate(
                regions[:, None].float(),
                size=all_logits.shape[-2:],
                mode="nearest",
            )[:, 0].bool()

        scores = []
        for pair_index, prediction_index in enumerate(self.target_indices):
            region = regions[pair_index]
            if not region.any():
                continue
            logits = all_logits[int(prediction_index.item())]
            # Foreground-only target, following segmentation CAM practice.
            scores.append(logits[region].mean())

        if not scores:
            raise RuntimeError("All selected ground-truth masks are empty.")

        score = torch.stack(scores).mean()

        return score.reshape(1, 1)


def validate_paths() -> None:
    missing = []

    for run in RUNS.values():
        if not run["config"].exists():
            missing.append(str(run["config"]))
        if not run["checkpoint"].exists():
            missing.append(str(run["checkpoint"]))

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            f"Required files were not found:\n{joined}"
        )


def build_test_loader(config, detector):
    data_dict, names = dataset_metadata(config)

    image_size = check_img_size(
        int(config["training"]["imgsz"]),
        s=max(int(detector.stride.max()), 32),
    )

    split = str(
        config.get("validation", {}).get("split", "test")
    )
    data_path = data_dict.get(split) or data_dict["val"]

    loader = create_dataloader(
        data_path,
        image_size,
        1,
        max(int(detector.stride.max()), 32),
        False,
        workers=0,
        pad=0.0,
        rect=False,
        mask_downsample_ratio=int(
            config["dataset"].get("mask_ratio", 4)
        ),
        overlap_mask=bool(
            config["dataset"].get("overlap_masks", True)
        ),
        prefix="Grad-CAM: ",
    )[0]

    resolve_max_det(config, loader.dataset.labels)
    return loader, names


def load_model(
    run: Dict,
    device: torch.device,
):
    config = load_config(str(run["config"]))
    config["training"]["device"] = str(device)

    data_dict, names = dataset_metadata(config)
    del data_dict

    detector = build_detector(config, len(names), device)
    sam = build_sam(config, device)

    (
        foundation,
        refiner,
        calibrator,
        context_resolver,
        class_features,
    ) = build_generalized_modules(
        config,
        names,
        device,
    )

    load_generalized_weights(
        str(run["checkpoint"]),
        detector,
        sam,
        refiner,
        calibrator,
        device,
    )

    detector.eval()
    sam.eval()
    foundation.eval()
    refiner.eval()
    calibrator.eval()

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


def get_sam_target_layer(sam):
    """Find the final Conv2d layer in the SAM image-encoder neck."""

    candidates = []
    for name, module in sam.named_modules():
        normalized = name.lower()
        if "image_encoder" not in normalized or "neck" not in normalized:
            continue
        if isinstance(module, torch.nn.Conv2d):
            candidates.append((name, module))

    if not candidates:
        available = [
            name
            for name, _ in sam.named_modules()
            if "image_encoder" in name.lower()
            and "neck" in name.lower()
        ]
        preview = "\n".join(available[-20:])
        raise RuntimeError(
            "No Conv2d layer was found in the SAM image-encoder neck. "
            "Available neck modules:\n"
            f"{preview}"
        )

    layer_name, target_layer = candidates[-1]
    return layer_name, target_layer


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    rgb = (
        image.detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    # OpenCV drawing functions require writable C-contiguous arrays.
    return np.ascontiguousarray(rgb)


def resize_masks(
    masks: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    if masks.numel() == 0:
        return np.zeros((0, height, width), dtype=bool)

    resized = F.interpolate(
        masks[:, None].float(),
        size=(height, width),
        mode="nearest",
    )[:, 0]

    return resized.detach().cpu().numpy() > 0.5


def draw_ground_truth(
    rgb_image: np.ndarray,
    boxes: torch.Tensor,
    masks: torch.Tensor,
) -> np.ndarray:
    color = (40, 210, 80)
    canvas = np.ascontiguousarray(
        np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    )
    height, width = canvas.shape[:2]
    resized_masks = resize_masks(masks, height, width)

    overlay = canvas.astype(np.float32)

    for mask in resized_masks:
        overlay[mask] = (
            0.60 * overlay[mask]
            + 0.40 * np.asarray(color, dtype=np.float32)
        )

    canvas = np.ascontiguousarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    )

    boxes_np = boxes.detach().cpu().numpy().astype(np.int32)

    for box in boxes_np:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def box_iou_matrix(
    reference_boxes: torch.Tensor,
    candidate_boxes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU between ground-truth and predicted boxes."""

    top_left = torch.maximum(
        reference_boxes[:, None, :2],
        candidate_boxes[None, :, :2],
    )
    bottom_right = torch.minimum(
        reference_boxes[:, None, 2:],
        candidate_boxes[None, :, 2:],
    )
    intersection_hw = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_hw[..., 0] * intersection_hw[..., 1]

    reference_hw = (
        reference_boxes[:, 2:] - reference_boxes[:, :2]
    ).clamp_min(0)
    candidate_hw = (
        candidate_boxes[:, 2:] - candidate_boxes[:, :2]
    ).clamp_min(0)
    reference_area = reference_hw[:, 0] * reference_hw[:, 1]
    candidate_area = candidate_hw[:, 0] * candidate_hw[:, 1]
    union = (
        reference_area[:, None]
        + candidate_area[None, :]
        - intersection
    )
    return intersection / union.clamp_min(1e-6)


def match_instances(
    target_boxes: torch.Tensor,
    candidate_boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy one-to-one IoU matching for all available instances."""

    if target_boxes.shape[0] == 0:
        raise RuntimeError("The selected image has no ground-truth instances.")
    if candidate_boxes.shape[0] == 0:
        raise RuntimeError("A model produced no detections for this image.")

    ious = box_iou_matrix(target_boxes, candidate_boxes)
    working = ious.clone()
    matches = []

    for _ in range(min(target_boxes.shape[0], candidate_boxes.shape[0])):
        flat_index = int(torch.argmax(working).item())
        target_index = flat_index // working.shape[1]
        candidate_index = flat_index % working.shape[1]
        score = working[target_index, candidate_index]
        if score < 0:
            break
        matches.append((target_index, candidate_index, ious[target_index, candidate_index]))
        working[target_index, :] = -1
        working[:, candidate_index] = -1

    matches.sort(key=lambda item: item[0])
    target_indices = torch.tensor(
        [item[0] for item in matches],
        dtype=torch.long,
        device=target_boxes.device,
    )
    candidate_indices = torch.tensor(
        [item[1] for item in matches],
        dtype=torch.long,
        device=candidate_boxes.device,
    )
    matched_ious = torch.stack([item[2] for item in matches])
    return target_indices, candidate_indices, matched_ious


def infer_proposal(
    bundle: Dict,
    images: torch.Tensor,
    paths,
):
    config = bundle["config"]

    with torch.no_grad():
        prediction, _ = detector_predictions(
            bundle["detector"],
            images,
            config,
        )

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

    return proposal


def generate_cam(
    bundle: Dict,
    images: torch.Tensor,
    proposal: Dict,
    prediction_indices: torch.Tensor,
    target_regions: torch.Tensor,
) -> Tuple[np.ndarray, str]:
    wrapper = SAMInstanceCAMWrapper(
        sam=bundle["sam"],
        boxes=proposal["boxes"],
        target_indices=prediction_indices,
        image_size=int(
            bundle["config"]["training"]["imgsz"]
        ),
        target_regions=target_regions,
    )

    layer_name, target_layer = get_sam_target_layer(bundle["sam"])

    cam_input = images.detach().clone()
    cam_input.requires_grad_(True)

    with GradCAM(
        model=wrapper,
        target_layers=[target_layer],
    ) as cam:
        grayscale_cam = cam(
            input_tensor=cam_input,
            targets=[ScalarOutputTarget()],
            aug_smooth=False,
            eigen_smooth=False,
        )[0]

    return grayscale_cam, layer_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        help="Use cpu, 0, or cuda:0.",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Image index in the PennFudanPed test loader.",
    )
    parser.add_argument(
        "--output",
        default="runs/gradcam_pennfudan/comparison.png",
    )
    args = parser.parse_args()

    validate_paths()

    device = select_device(args.device, batch_size=1)

    reference_bundle = load_model(
        RUNS["A1 + CLIP"],
        device,
    )

    loader, _ = build_test_loader(
        reference_bundle["config"],
        reference_bundle["detector"],
    )

    selected_batch = None

    for batch_index, batch in enumerate(loader):
        if batch_index == args.image_index:
            selected_batch = batch
            break

    if selected_batch is None:
        raise IndexError(
            f"Image index {args.image_index} is outside "
            "the test dataset."
        )

    images, targets, paths, _, masks = selected_batch
    images = images.to(device).float() / 255.0
    targets = targets.to(device)
    masks = masks.to(device).float()

    overlap = bool(
        reference_bundle["config"]["dataset"].get(
            "overlap_masks",
            True,
        )
    )

    target_indices, target_boxes, _ = targets_for_image(
        targets,
        0,
        tuple(images.shape[-2:]),
    )

    target_masks = instance_masks_for_image(
        masks,
        target_indices,
        0,
        overlap,
    )

    if target_boxes.shape[0] == 0:
        raise RuntimeError("The selected image has no ground-truth instances.")

    bundles = {
        "A1 + CLIP": reference_bundle,
        "A2 + Transformer": load_model(
            RUNS["A2 + Transformer"],
            device,
        ),
    }

    proposals = {}

    for name, bundle in bundles.items():
        proposal = infer_proposal(
            bundle,
            images,
            paths,
        )
        proposals[name] = proposal

    rgb_image = tensor_to_rgb(images[0])
    panels = [
        (rgb_image * 255).astype(np.uint8),
        draw_ground_truth(
            rgb_image,
            target_boxes,
            target_masks,
        ),
    ]
    titles = ["Original Image", "Ground Truth"]

    for name in ("A1 + CLIP", "A2 + Transformer"):
        proposal = proposals[name]
        boxes = proposal["boxes"]

        gt_indices, prediction_indices, matched_ious = match_instances(
            target_boxes,
            boxes,
        )

        grayscale_cam, layer_name = generate_cam(
            bundles[name],
            images,
            proposal,
            prediction_indices,
            target_masks[gt_indices] > 0.5,
        )

        cam_panel = show_cam_on_image(
            rgb_image.astype(np.float32),
            grayscale_cam,
            use_rgb=True,
        )

        panels.append(cam_panel)
        titles.append(name)
        print(f"{name} target layer: {layer_name}")
        print(
            f"{name} matched {len(prediction_indices)} instances; "
            f"IoUs: {[round(float(v), 4) for v in matched_ious]}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(12, 3.4),
    )

    for axis, panel, title in zip(
        axes,
        panels,
        titles,
    ):
        axis.imshow(panel)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    figure.tight_layout(pad=0.5)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Input image: {paths[0]}")
    print(f"Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

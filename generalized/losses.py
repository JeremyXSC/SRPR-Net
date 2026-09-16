from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    numerator = 2.0 * (probabilities * targets).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    return (1.0 - (numerator + eps) / (denominator + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def sobel_boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Match mask boundaries with a differentiable Sobel-gradient loss."""
    if logits.ndim != 3 or targets.ndim != 3:
        raise ValueError("logits and targets must have shape [N, H, W].")
    dtype = logits.dtype
    device = logits.device
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    probabilities = torch.sigmoid(logits).unsqueeze(1)
    target_values = targets.to(dtype=dtype).unsqueeze(1)

    def gradients(values: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(values, kernel_x, padding=1)
        grad_y = F.conv2d(values, kernel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    return F.l1_loss(gradients(probabilities), gradients(target_values))


def bce_dice_boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    boundary_weight: float = 0.0,
) -> torch.Tensor:
    loss = bce_dice_loss(logits, targets)
    if float(boundary_weight) > 0.0:
        loss = loss + float(boundary_weight) * sobel_boundary_loss(logits, targets)
    return loss


def binary_mask_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one detached IoU target per predicted mask."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes.")
    predictions = logits.detach().gt(float(threshold))
    target_masks = targets.detach().gt(0.5)
    intersection = (predictions & target_masks).flatten(1).sum(dim=1).float()
    union = (predictions | target_masks).flatten(1).sum(dim=1).float()
    return (intersection + eps) / (union + eps)


def geometric_score_fusion(
    detector_scores: torch.Tensor,
    mask_quality: torch.Tensor,
    mask_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse detector confidence and predicted mask quality in log space."""
    weight = min(max(float(mask_weight), 0.0), 1.0)
    detector_scores = detector_scores.clamp(eps, 1.0)
    mask_quality = mask_quality.clamp(eps, 1.0)
    return torch.exp(
        (1.0 - weight) * detector_scores.log() + weight * mask_quality.log()
    ).clamp(0.0, 1.0)


def prompt_constraint_loss(
    refined_boxes: torch.Tensor,
    original_boxes: torch.Tensor,
    image_hw: Tuple[int, int],
    min_area_ratio: float = 0.55,
    max_area_ratio: float = 1.80,
    max_center_shift: float = 0.20,
) -> torch.Tensor:
    if refined_boxes.numel() == 0:
        return refined_boxes.sum() * 0.0
    height, width = image_hw
    ref_wh = (refined_boxes[:, 2:] - refined_boxes[:, :2]).clamp_min(1.0)
    org_wh = (original_boxes[:, 2:] - original_boxes[:, :2]).clamp_min(1.0)
    area_ratio = ref_wh.prod(dim=1) / org_wh.prod(dim=1).clamp_min(1.0)
    area_penalty = F.relu(min_area_ratio - area_ratio) + F.relu(area_ratio - max_area_ratio)

    ref_center = 0.5 * (refined_boxes[:, :2] + refined_boxes[:, 2:])
    org_center = 0.5 * (original_boxes[:, :2] + original_boxes[:, 2:])
    diagonal = torch.sqrt(org_wh[:, 0].pow(2) + org_wh[:, 1].pow(2)).clamp_min(1.0)
    shift = torch.norm(ref_center - org_center, dim=1) / diagonal
    shift_penalty = F.relu(shift - max_center_shift)

    boundary = (
        F.relu(-refined_boxes[:, 0])
        + F.relu(-refined_boxes[:, 1])
        + F.relu(refined_boxes[:, 2] - float(width - 1))
        + F.relu(refined_boxes[:, 3] - float(height - 1))
    ) / float(max(height, width))
    return (area_penalty + shift_penalty + boundary).mean()


def match_boxes_by_class(
    predicted_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    min_iou: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if predicted_boxes.numel() == 0 or target_boxes.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=predicted_boxes.device)
        return empty, empty
    ious = box_iou(predicted_boxes, target_boxes).detach()
    class_match = predicted_labels[:, None].long() == target_labels[None, :].long()
    scores = torch.where(class_match, ious, torch.full_like(ious, -1.0))
    selected_predictions = []
    selected_targets = []
    while scores.numel() and float(scores.max()) >= float(min_iou):
        flat_index = int(scores.argmax())
        prediction_index = flat_index // scores.shape[1]
        target_index = flat_index % scores.shape[1]
        selected_predictions.append(prediction_index)
        selected_targets.append(target_index)
        scores[prediction_index, :] = -1.0
        scores[:, target_index] = -1.0
    if not selected_predictions:
        empty = torch.empty(0, dtype=torch.long, device=predicted_boxes.device)
        return empty, empty
    return (
        torch.tensor(selected_predictions, dtype=torch.long, device=predicted_boxes.device),
        torch.tensor(selected_targets, dtype=torch.long, device=predicted_boxes.device),
    )

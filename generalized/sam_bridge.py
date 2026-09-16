from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .losses import bce_dice_boundary_loss, binary_mask_iou, match_boxes_by_class


def targets_for_image(targets: torch.Tensor, image_index: int, image_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(targets[:, 0].long() == int(image_index), as_tuple=False).squeeze(1)
    local = targets[indices]
    height, width = image_hw
    if local.numel() == 0:
        return indices, targets.new_zeros((0, 4)), targets.new_zeros((0,), dtype=torch.long)
    xywh = local[:, 2:6].clone()
    xywh[:, [0, 2]] *= float(width)
    xywh[:, [1, 3]] *= float(height)
    boxes = torch.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
    return indices, boxes, local[:, 1].long()


def instance_masks_for_image(
    masks: torch.Tensor,
    global_target_indices: torch.Tensor,
    image_index: int,
    overlap: bool,
) -> torch.Tensor:
    if global_target_indices.numel() == 0:
        if masks.ndim >= 2:
            return masks.new_zeros((0, masks.shape[-2], masks.shape[-1])).float()
        return masks.new_zeros((0, 1, 1)).float()
    if overlap:
        mask_map = masks[image_index] if masks.ndim == 3 else masks
        return torch.stack([(mask_map == (index + 1)).float() for index in range(global_target_indices.numel())], dim=0)
    return masks[global_target_indices].float()


def sam_loss_for_image(
    sam_model,
    image: torch.Tensor,
    refined_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    image_size: int,
    min_match_iou: float,
    boundary_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    pred_index, target_index = match_boxes_by_class(
        refined_boxes, predicted_labels, target_boxes, target_labels, min_iou=min_match_iou
    )
    if pred_index.numel() == 0:
        zero = refined_boxes.sum() * 0.0
        return {
            "loss": zero,
            "logits": refined_boxes.new_zeros((0, 1, 1)),
            "pred_index": pred_index,
            "target_index": target_index,
            "iou_quality": refined_boxes.new_zeros((0,)),
            "mask_iou_target": refined_boxes.new_zeros((0,)),
        }

    selected_boxes = refined_boxes[pred_index]
    output = sam_model(
        image.unsqueeze(0),
        multimask_output=False,
        image_size=image_size,
        bbox=selected_boxes,
    )
    logits = output["low_res_logits"][:, 0]
    selected_masks = target_masks[target_index].float()
    selected_masks = F.interpolate(
        selected_masks.unsqueeze(1), size=logits.shape[-2:], mode="nearest"
    ).squeeze(1)
    loss = bce_dice_boundary_loss(
        logits,
        selected_masks,
        boundary_weight=float(boundary_weight),
    )
    iou_quality = output.get("iou_predictions", logits.new_zeros((logits.shape[0], 1))).reshape(logits.shape[0], -1)[:, 0]
    mask_iou_target = binary_mask_iou(logits, selected_masks)
    return {
        "loss": loss,
        "logits": logits,
        "pred_index": pred_index,
        "target_index": target_index,
        "iou_quality": iou_quality,
        "mask_iou_target": mask_iou_target,
    }

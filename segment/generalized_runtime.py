"""Shared runtime helpers for the generalized BLO-Inst training and validation scripts.

This file is copied into the upstream BLO-Inst repository. It intentionally
uses the upstream detector, dataloader, NMS and SAM implementations rather than
forking those components.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from generalized.context import ContextResolver
from generalized.foundation_encoder import build_foundation_encoder
from generalized.prompt_refiner import MultiModalBoxRefiner
from generalized.semantic_prior import BayesianSemanticCalibrator, SemanticPriorTable
from models.experimental import attempt_load
from models.segment_anything import sam_model_registry
from models.sam_lora_mask_decoder import LoRA_Sam
from models.yolo import SegmentationModel
from utils.downloads import attempt_download
from utils.general import check_dataset, check_suffix, intersect_dicts, non_max_suppression
from utils.torch_utils import de_parallel, scale_img


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _expand(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(Path(__file__).resolve().parents[1])
    return config


def resolve_project_path(value: Optional[str], project_root: str) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return value
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((Path(project_root) / path).resolve())


def dataset_metadata(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    project_root = config["_project_root"]
    data_path = resolve_project_path(config["dataset"]["yaml"], project_root)
    data_dict = check_dataset(data_path)
    names_value = data_dict["names"]
    if isinstance(names_value, dict):
        names = [str(names_value[index]) for index in sorted(names_value)]
    else:
        names = [str(name) for name in names_value]
    return data_dict, names


def build_detector(config: Dict[str, Any], nc: int, device: torch.device):
    project_root = config["_project_root"]
    detector_cfg = resolve_project_path(config["model"].get("cfg", ""), project_root)
    weights = resolve_project_path(config["model"].get("weights", ""), project_root) or ""
    hyp = config.get("hyperparameters", {})

    pretrained = bool(weights and str(weights).endswith(".pt"))
    if pretrained:
        check_suffix(weights, ".pt")
        weights = attempt_download(weights)
        # Detector checkpoints contain a serialized YOLO module. PyTorch 2.6+
        # defaults to weights_only=True, which rejects this trusted local file.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        source_model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        state_only = isinstance(checkpoint, dict) and "detector" in checkpoint
        if source_model is None and not state_only:
            source_model = checkpoint
        if state_only and not detector_cfg:
            raise ValueError("model.cfg is required for detector state-dict checkpoints.")
        model_yaml = detector_cfg or source_model.yaml
        model = SegmentationModel(model_yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
        source_state = (
            checkpoint["detector"]
            if state_only
            else source_model.float().state_dict()
        )
        exclusions = ["anchor"] if detector_cfg or hyp.get("anchors") else []
        source_state = intersect_dicts(source_state, model.state_dict(), exclude=exclusions)
        model.load_state_dict(source_state, strict=False)
    else:
        if not detector_cfg:
            raise ValueError("model.cfg or model.weights must be configured.")
        model = SegmentationModel(detector_cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)

    # Training later sets this metadata while configuring the loss, but the
    # standalone validation path also needs it before proposal generation.
    model.nc = int(nc)
    return model


def configure_detector_for_loss(model, hyp: Dict[str, Any], names: List[str], dataset_labels) -> None:
    from utils.general import labels_to_class_weights

    model = de_parallel(model)
    nl = model.model[-1].nl
    scaled_hyp = dict(hyp)
    scaled_hyp["box"] = scaled_hyp.get("box", 0.05) * 3.0 / nl
    scaled_hyp["cls"] = scaled_hyp.get("cls", 0.5) * len(names) / 80.0 * 3.0 / nl
    scaled_hyp["obj"] = scaled_hyp.get("obj", 1.0) * (float(scaled_hyp.get("imgsz", 640)) / 640.0) ** 2 * 3.0 / nl
    scaled_hyp.setdefault("cls_pw", 1.0)
    scaled_hyp.setdefault("obj_pw", 1.0)
    scaled_hyp.setdefault("fl_gamma", 0.0)
    scaled_hyp.setdefault("anchor_t", 4.0)
    scaled_hyp.setdefault("label_smoothing", 0.0)
    model.nc = len(names)
    model.hyp = scaled_hyp
    model.names = names
    model.class_weights = labels_to_class_weights(dataset_labels, len(names)).to(next(model.parameters()).device) * len(names)


def build_sam(config: Dict[str, Any], device: torch.device):
    project_root = config["_project_root"]
    sam_cfg = config["sam"]
    checkpoint = resolve_project_path(sam_cfg.get("checkpoint", ""), project_root)
    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(
            "SAM checkpoint was not found: {}. Set sam.checkpoint in the configuration file.".format(checkpoint)
        )
    sam, _ = sam_model_registry[str(sam_cfg.get("vit_name", "vit_b"))](
        image_size=int(config["training"]["imgsz"]), checkpoint=checkpoint
    )
    rank = int(sam_cfg.get("lora_rank", 4))
    return LoRA_Sam(sam, rank).to(device)


def build_generalized_modules(
    config: Dict[str, Any], names: List[str], device: torch.device
) -> Tuple[torch.nn.Module, MultiModalBoxRefiner, BayesianSemanticCalibrator, ContextResolver, torch.Tensor]:
    project_root = config["_project_root"]
    foundation = build_foundation_encoder(config.get("foundation", {}), device)
    refiner_cfg = config.get("refiner", {})
    refiner = MultiModalBoxRefiner(
        foundation_dim=int(foundation.output_dim),
        hidden_dim=int(refiner_cfg.get("hidden_dim", 256)),
        num_heads=int(refiner_cfg.get("num_heads", 8)),
        num_layers=int(refiner_cfg.get("num_layers", 2)),
        dropout=float(refiner_cfg.get("dropout", 0.1)),
        max_center_shift=float(refiner_cfg.get("max_center_shift", 0.25)),
        max_log_scale=float(refiner_cfg.get("max_log_scale", 0.35)),
        enabled=bool(refiner_cfg.get("enabled", True)),
        use_attention=bool(refiner_cfg.get("use_attention", True)),
    ).to(device)

    prior_cfg = config.get("semantic_prior", {})
    prior_path = resolve_project_path(prior_cfg.get("path"), project_root)
    prior_table = SemanticPriorTable.from_json(prior_path, names)
    calibrator = BayesianSemanticCalibrator(
        prior_table=prior_table,
        initial_alpha=float(prior_cfg.get("initial_alpha", 0.35)),
        learnable_alpha=bool(prior_cfg.get("learnable_alpha", True)),
        max_alpha=float(prior_cfg.get("max_alpha", 2.0)),
        enabled=bool(prior_cfg.get("enabled", True)),
    ).to(device)

    context_path = resolve_project_path(prior_cfg.get("context_map"), project_root)
    context_resolver = ContextResolver(context_path, str(prior_cfg.get("default_context", "default")))
    class_features = foundation.encode_text(names).to(device)
    return foundation, refiner, calibrator, context_resolver, class_features


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def restore_trainable_parameters(module: torch.nn.Module, trainable_names: List[str]) -> None:
    trainable = set(trainable_names)
    for name, parameter in module.named_parameters():
        parameter.requires_grad = name in trainable


def resolve_max_det(
    config: Dict[str, Any],
    labels: Optional[List[Any]] = None,
) -> int:
    """Resolve max_det, deriving it from annotation density when configured as auto."""
    inference = config.setdefault("inference", {})
    raw_value = inference.get("max_det", 300)
    if str(raw_value).lower() != "auto":
        value = int(raw_value)
        inference["_resolved_max_det"] = value
        return value
    if labels is not None:
        maximum = max((len(label) for label in labels), default=0)
        value = max(300, min(1000, int(math.ceil(maximum * 1.25))))
        inference["_resolved_max_det"] = value
        return value
    return int(inference.get("_resolved_max_det", 300))


def detector_predictions(
    detector,
    images: torch.Tensor,
    config: Dict[str, Any],
    tta: Optional[bool] = None,
):
    """Run the detector once or with configured multi-scale/flip TTA."""
    tta_cfg = config.get("inference", {}).get("tta", {})
    enabled = bool(tta_cfg.get("enabled", False)) if tta is None else bool(tta)
    if not enabled:
        return detector(images)

    scales = [float(value) for value in tta_cfg.get("scales", [0.83, 1.0, 1.17])]
    flip_names = [str(value).lower() for value in tta_cfg.get("flips", ["none", "horizontal"])]
    flip_dimensions = {"none": None, "horizontal": 3, "vertical": 2}
    unknown = sorted(set(flip_names) - set(flip_dimensions))
    if unknown:
        raise ValueError("Unknown TTA flips: {}".format(unknown))
    image_size = images.shape[-2:]
    base_model = de_parallel(detector)
    predictions = []
    for scale in scales:
        for flip_name in flip_names:
            flip_dimension = flip_dimensions[flip_name]
            augmented = images.flip(flip_dimension) if flip_dimension is not None else images
            augmented = scale_img(augmented, scale, gs=int(base_model.stride.max()))
            prediction, _ = detector(augmented)
            prediction = descale_tta_prediction(
                prediction,
                flip_dimension,
                scale,
                image_size,
            )
            predictions.append(prediction)
    return torch.cat(predictions, dim=1), None


def descale_tta_prediction(
    prediction: torch.Tensor,
    flip_dimension: Optional[int],
    scale: float,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Map augmented YOLO xywh predictions back to the original image."""
    output = prediction.clone()
    output[..., :4] /= float(scale)
    if flip_dimension == 2:
        output[..., 1] = float(image_size[0]) - output[..., 1]
    elif flip_dimension == 3:
        output[..., 0] = float(image_size[1]) - output[..., 0]
    elif flip_dimension is not None:
        raise ValueError("flip_dimension must be None, 2 or 3.")
    return output


def proposal_batch(
    prediction: torch.Tensor,
    images: torch.Tensor,
    paths: List[str],
    model,
    foundation: torch.nn.Module,
    refiner: MultiModalBoxRefiner,
    calibrator: BayesianSemanticCalibrator,
    context_resolver: ContextResolver,
    class_features: torch.Tensor,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, List[str]]:
    nc = int(de_parallel(model).nc)
    contexts = context_resolver.resolve(paths)
    calibrated = calibrator(prediction, contexts, nc=nc)
    inference = config.get("inference", {})
    nm = int(de_parallel(model).model[-1].nm)
    detections = non_max_suppression(
        calibrated,
        conf_thres=float(inference.get("conf_thres", 0.001)),
        iou_thres=float(inference.get("nms_iou", 0.25)),
        labels=(),
        multi_label=True,
        agnostic=bool(inference.get("agnostic_nms", False)),
        max_det=resolve_max_det(config),
        nm=nm,
    )
    image_features = foundation.encode_image(images)
    height, width = images.shape[-2:]
    quality_gain = float(inference.get("quality_gain", 0.25))
    proposals: List[Dict[str, torch.Tensor]] = []
    for image_index, detection in enumerate(detections):
        if detection.numel() == 0:
            zero_boxes = images.new_zeros((0, 4))
            proposals.append(
                {
                    "boxes": zero_boxes,
                    "original_boxes": zero_boxes,
                    "scores": images.new_zeros((0,)),
                    "labels": torch.empty(0, dtype=torch.long, device=images.device),
                    "quality": images.new_zeros((0,)),
                }
            )
            continue
        original_boxes = detection[:, :4]
        scores = detection[:, 4]
        labels = detection[:, 5].long()
        refined_boxes, quality = refiner(
            original_boxes,
            scores,
            labels,
            image_features[image_index],
            class_features,
            (height, width),
        )
        adjusted_scores = (scores * (1.0 + quality_gain * (quality - 0.5))).clamp(0.0, 1.0)
        proposals.append(
            {
                "boxes": refined_boxes,
                "original_boxes": original_boxes,
                "scores": adjusted_scores,
                "raw_scores": scores,
                "labels": labels,
                "quality": quality,
            }
        )
    return proposals, calibrated, contexts


def load_generalized_weights(
    checkpoint_path: str,
    detector,
    sam,
    refiner,
    calibrator,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    detector.load_state_dict(checkpoint["detector"], strict=True)
    sam.load_state_dict(checkpoint["sam"], strict=True)
    refiner.load_state_dict(checkpoint["refiner"], strict=True)
    calibrator.load_state_dict(checkpoint["calibrator"], strict=True)
    return checkpoint

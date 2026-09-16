"""Controlled ablation experiments for the final BLO-Inst extension.

A0: BLO baseline
A1: + CLIP-guided prompt refiner (without Transformer attention)
A2: + Transformer attention
A3: + explicit geometry constraint (Full model)

Semantic prior, weak supervision, SAM-quality supervision,
Sobel boundary loss and mask-quality score fusion are disabled.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


VARIANTS = {
    # Original BLO-style baseline under the unified training/evaluation pipeline.
    "A0_blo_baseline": {
        "foundation.enabled": False,
        "refiner.enabled": False,
        "refiner.use_attention": False,

        "semantic_prior.enabled": False,

        "loss.lambda_constraint": 0.0,
        "loss.lambda_semantic": 0.0,
        "loss.lambda_quality": 0.0,
        "loss.lambda_sam_quality": 0.0,
        "loss.quality_target": "box_iou",
        "loss.boundary_weight": 0.0,

        "inference.quality_gain": 0.0,
        "inference.mask_quality_weight": 0.0,
        "inference.tta.enabled": False,
    },

    # CLIP semantic features + independent MLP box refinement.
    "A1_clip_refiner": {
        "foundation.enabled": True,
        "refiner.enabled": True,
        "refiner.use_attention": False,

        "semantic_prior.enabled": False,

        "loss.lambda_constraint": 0.0,
        "loss.lambda_semantic": 0.0,
        "loss.lambda_quality": 0.05,
        "loss.lambda_sam_quality": 0.0,
        "loss.quality_target": "box_iou",
        "loss.boundary_weight": 0.0,

        "inference.quality_gain": 0.25,
        "inference.mask_quality_weight": 0.0,
        "inference.tta.enabled": False,
    },

    # CLIP semantic features + Transformer relation modeling.
    "A2_clip_transformer": {
        "foundation.enabled": True,
        "refiner.enabled": True,
        "refiner.use_attention": True,

        "semantic_prior.enabled": False,

        "loss.lambda_constraint": 0.0,
        "loss.lambda_semantic": 0.0,
        "loss.lambda_quality": 0.05,
        "loss.lambda_sam_quality": 0.0,
        "loss.quality_target": "box_iou",
        "loss.boundary_weight": 0.0,

        "inference.quality_gain": 0.25,
        "inference.mask_quality_weight": 0.0,
        "inference.tta.enabled": False,
    },

    # Full model: CLIP + Transformer + explicit geometry constraint.
    "A3_full": {
        "foundation.enabled": True,
        "refiner.enabled": True,
        "refiner.use_attention": True,

        "semantic_prior.enabled": False,

        "loss.lambda_constraint": 0.08,
        "loss.lambda_semantic": 0.0,
        "loss.lambda_quality": 0.05,
        "loss.lambda_sam_quality": 0.0,
        "loss.quality_target": "box_iou",
        "loss.boundary_weight": 0.0,

        "inference.quality_gain": 0.25,
        "inference.mask_quality_weight": 0.0,
        "inference.tta.enabled": False,
    },
}


def set_nested(config, key, value):
    parts = key.split(".")
    cursor = config

    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})

    cursor[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        required=True,
        help="Base generalized experiment YAML.",
    )

    parser.add_argument(
        "--output",
        default="configs/ablations",
        help="Directory for generated ablation YAMLs.",
    )

    parser.add_argument(
        "--detector-weight",
        default=None,
        help="Optional target-domain detector checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute training after generating configs.",
    )

    args = parser.parse_args()

    base_path = Path(args.base)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    dataset_yaml = Path(base["dataset"]["yaml"])
    dataset_tag = dataset_yaml.stem.replace("_generalized", "")

    config_root = Path(args.output) / dataset_tag
    run_root = Path("runs") / "ablations" / dataset_tag

    config_root.mkdir(parents=True, exist_ok=True)

    for name, changes in VARIANTS.items():
        config = copy.deepcopy(base)

        config["experiment"]["name"] = f"{dataset_tag}_{name}"

        config["validation"]["split"] = "val"

        config["output"]["directory"] = str(
            run_root / name
        )

        if args.detector_weight:
            config["model"]["weights"] = args.detector_weight

        if args.epochs is not None:
            config["training"]["epochs"] = int(args.epochs)

        for key, value in changes.items():
            set_nested(config, key, value)

        target = config_root / f"{name}.yaml"

        target.write_text(
            yaml.safe_dump(
                config,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        print(target)

        if args.execute:
            subprocess.run(
                [
                    sys.executable,
                    "segment/train_generalized.py",
                    "--config",
                    str(target),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
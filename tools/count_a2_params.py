import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment.generalized_runtime import (
    load_config,
    dataset_metadata,
    build_detector,
    build_sam,
    build_generalized_modules,
    load_generalized_weights,
)
from utils.torch_utils import select_device


def count_parameters(module):
    """统计模块的总参数量和可训练参数量。"""
    if module is None:
        return 0, 0

    total = 0
    trainable = 0

    for parameter in module.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()

    return total, trainable


def main():
    device = select_device("cpu")

    configs = {
        "PennFudanPed": (
            "runs/ablations/pennfudan/A2_clip_transformer/resolved_config.yaml",
            "runs/ablations/pennfudan/A2_clip_transformer/best_generalized.pt",
        ),
        "RWCellIns": (
            "runs/ablations/rwcell/A2_clip_transformer/resolved_config.yaml",
            "runs/ablations/rwcell/A2_clip_transformer/best_generalized.pt",
        ),
        "WheatIns": (
            "runs/ablations/wheat/A2_clip_transformer/resolved_config.yaml",
            "runs/ablations/wheat/A2_clip_transformer/best_generalized.pt",
        ),
    }

    for dataset_name, (config_path, checkpoint_path) in configs.items():
        print("\n" + "=" * 65)
        print(f"{dataset_name} / A2_clip_transformer")
        print("=" * 65)

        config_path = ROOT / config_path
        checkpoint_path = ROOT / checkpoint_path

        config = load_config(str(config_path))
        data_dict, names = dataset_metadata(config)

        detector = build_detector(config, len(names), device)
        sam = build_sam(config, device)

        foundation, refiner, calibrator, context_resolver, class_features = (
            build_generalized_modules(config, names, device)
        )

        load_generalized_weights(
            str(checkpoint_path),
            detector,
            sam,
            refiner,
            calibrator,
            device,
        )

        modules = {
            "Detector": detector,
            "SAM": sam,
            "CLIP foundation": foundation,
            "Refiner": refiner,
            "Calibrator": calibrator,
        }

        total_all = 0
        trainable_all = 0

        for module_name, module in modules.items():
            total, trainable = count_parameters(module)

            print(
                f"{module_name:<18} "
                f"Total: {total / 1e6:>10.4f}M | "
                f"Trainable: {trainable / 1e6:>10.4f}M"
            )

            total_all += total
            trainable_all += trainable

        print("-" * 65)
        print(f"Total Params:     {total_all / 1e6:.4f}M")
        print(f"Trainable Params: {trainable_all / 1e6:.4f}M")

        sam_trainable = [
            name
            for name, parameter in sam.named_parameters()
            if parameter.requires_grad
        ]

        print("\nTrainable SAM parameters:")
        for name in sam_trainable:
            print(f"  {name}")


if __name__ == "__main__":
    main()
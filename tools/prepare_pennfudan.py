"""Convert the official PennFudanPed masks to YOLO instance-polygon labels.

Expected source layout::

    PennFudanPed/
    ├─ PNGImages/*.png
    └─ PedMasks/*_mask.png

The default 74/96 split is deterministic but is not claimed to be identical to
BLO-Inst's unpublished file list. Pass --train-list when an exact reproduction
list is available.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def load_train_stems(path: str) -> Set[str]:
    if not path:
        return set()
    values = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if item:
            values.add(Path(item).stem.replace("_mask", ""))
    return values


def find_mask(source: Path, image: Path) -> Path:
    direct = source / "PedMasks" / (image.stem + "_mask.png")
    if direct.exists():
        return direct
    candidates = sorted((source / "PedMasks").glob(image.stem + "*"))
    if not candidates:
        raise FileNotFoundError("Mask not found for {}".format(image))
    return candidates[0]


def contour_to_line(contour: np.ndarray, width: int, height: int, class_id: int = 0) -> str:
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 3:
        return ""
    points[:, 0] = np.clip(points[:, 0] / float(width), 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / float(height), 0.0, 1.0)
    coordinates = " ".join("{:.6f} {:.6f}".format(x, y) for x, y in points)
    return "{} {}".format(class_id, coordinates)


def mask_to_yolo_lines(mask_path: Path, epsilon_ratio: float, min_area: float) -> List[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Unable to read mask: {}".format(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape[:2]
    lines = []
    for instance_id in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        binary = (mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.0, epsilon_ratio) * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0.0 else contour
        line = contour_to_line(simplified, width, height)
        if line:
            lines.append(line)
    return lines


def select_train_images(
    images: Sequence[Path], train_count: int, seed: int, train_stems: Set[str]
) -> Set[str]:
    stems = {image.stem for image in images}
    if train_stems:
        unknown = sorted(train_stems - stems)
        if unknown:
            raise ValueError("Train list contains unknown images: {}".format(unknown[:10]))
        return set(train_stems)
    if not 0 < train_count < len(images):
        raise ValueError("train-count must be between 1 and {}".format(len(images) - 1))
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    return {image.stem for image in shuffled[:train_count]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Official PennFudanPed root")
    parser.add_argument("--output", required=True, help="Converted dataset root")
    parser.add_argument("--train-count", type=int, default=74)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-list",
        default="",
        help="Optional exact training filenames/stems; remaining images form val/test",
    )
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--epsilon-ratio", type=float, default=0.001)
    parser.add_argument("--min-area", type=float, default=4.0)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    image_dir = source / "PNGImages"
    mask_dir = source / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Expected PNGImages and PedMasks under {}".format(source))

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("At least two images are required.")
    train_stems = select_train_images(
        images, int(args.train_count), int(args.seed), load_train_stems(args.train_list)
    )

    split_images: Dict[str, List[str]] = {"train": [], "val": []}
    statistics = {"images": len(images), "instances": 0, "train": 0, "val": 0, "empty_labels": 0}
    for image in images:
        split = "train" if image.stem in train_stems else "val"
        target_image = output / "images" / split / image.name
        target_label = output / "labels" / split / (image.stem + ".txt")
        copy_or_link(image, target_image, args.mode)
        lines = mask_to_yolo_lines(find_mask(source, image), args.epsilon_ratio, args.min_area)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_images[split].append(target_image.as_posix())
        statistics[split] += 1
        statistics["instances"] += len(lines)
        statistics["empty_labels"] += int(not lines)

    for split, records in split_images.items():
        (output / (split + ".txt")).write_text("\n".join(records) + "\n", encoding="utf-8")
    # The original BLO-Inst setup uses the held-out 96 images as both validation and test.
    (output / "test.txt").write_text("\n".join(split_images["val"]) + "\n", encoding="utf-8")
    data_yaml = {
        "path": output.as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "nc": 1,
        "names": ["person"],
    }
    (output / "pennfudan.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    split_manifest = {
        "seed": int(args.seed),
        "train_count_requested": int(args.train_count),
        "exact_train_list": bool(args.train_list),
        "train_stems": sorted(train_stems),
        "statistics": statistics,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print("Dataset YAML: {}".format(output / "pennfudan.yaml"))


if __name__ == "__main__":
    main()

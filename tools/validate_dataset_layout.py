"""Validate the image-list and YOLO polygon-label layout before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lower = [part.lower() for part in parts]
    if "images" in lower:
        index = len(lower) - 1 - lower[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    yaml_path = Path(args.data).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    declared_root = Path(data.get("path", "."))
    if declared_root.is_absolute():
        root = declared_root.resolve()
    else:
        cwd_candidate = (Path.cwd() / declared_root).resolve()
        yaml_candidate = (yaml_path.parent / declared_root).resolve()
        root = cwd_candidate if cwd_candidate.exists() or not yaml_candidate.exists() else yaml_candidate
    errors = []
    counts = {}
    split_images = {}
    class_count = int(data.get("nc", len(data.get("names", []))))
    names = data.get("names", [])
    if len(names) != class_count:
        errors.append("nc={} does not match {} class names.".format(class_count, len(names)))
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        list_path = resolve(root, data[split]).resolve()
        if not list_path.exists():
            errors.append("{} list missing: {}".format(split, list_path))
            continue
        images = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        resolved_images = {
            str((image if image.is_absolute() else root / image).resolve()).casefold()
            for image in images
        }
        if len(resolved_images) != len(images):
            errors.append("{} contains duplicate image paths.".format(split))
        split_images[split] = resolved_images
        missing_images = 0
        missing_labels = 0
        invalid_labels = 0
        for image in images:
            image_path = image if image.is_absolute() else root / image
            if not image_path.exists():
                missing_images += 1
                continue
            label_path = image_to_label(image_path)
            if not label_path.exists():
                missing_labels += 1
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 7 or (len(values) - 1) % 2 != 0:
                    invalid_labels += 1
                    continue
                try:
                    class_id = int(float(values[0]))
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    invalid_labels += 1
                    continue
                if class_id < 0 or class_id >= class_count:
                    invalid_labels += 1
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    invalid_labels += 1
        counts[split] = {
            "images": len(images),
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "invalid_label_lines": invalid_labels,
        }
    split_names = sorted(split_images)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            overlap = split_images[first] & split_images[second]
            if overlap:
                errors.append(
                    "{} and {} overlap by {} images.".format(first, second, len(overlap))
                )
    manifest_path = root / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_names = [str(value) for value in manifest.get("class_names", [])]
        if manifest_names and manifest_names != [str(value) for value in names]:
            errors.append(
                "Dataset class names do not match split_manifest.json: {} != {}.".format(
                    names, manifest_names
                )
            )
        for split in ("train", "val", "test"):
            expected = manifest.get("splits", {}).get(split, {}).get("images")
            if expected is not None and split in counts and int(expected) != counts[split]["images"]:
                errors.append(
                    "{} count {} does not match manifest {}.".format(
                        split, counts[split]["images"], expected
                    )
                )
    print(yaml.safe_dump(counts, allow_unicode=True, sort_keys=False))
    if errors:
        raise SystemExit("\n".join(errors))
    if any(item[key] for item in counts.values() for key in ("missing_images", "missing_labels", "invalid_label_lines")):
        raise SystemExit("Dataset validation failed; see counts above.")
    print("Dataset layout validation passed.")


if __name__ == "__main__":
    main()

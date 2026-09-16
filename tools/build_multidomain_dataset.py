"""Build a unified YOLO-seg dataset from multiple already-labeled domains.

The source annotations must use YOLO polygon format. A JSON specification maps
each local class into a shared global vocabulary. Files are symlinked by default,
so the tool does not duplicate large image collections.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def image_to_label(image: Path) -> Path:
    text = str(image)
    normalized = text.replace("\\", "/")
    if "/images/" in normalized:
        normalized = normalized.replace("/images/", "/labels/")
        return Path(normalized).with_suffix(".txt")
    return image.with_suffix(".txt")


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def parse_mapping(dataset, global_index):
    local_names = [str(name) for name in dataset["class_names"]]
    declared = dataset.get("class_map", {})
    mapping = {}
    for local_index, local_name in enumerate(local_names):
        target = declared.get(str(local_index), declared.get(local_name, local_name))
        if isinstance(target, int):
            mapping[local_index] = target
        else:
            if str(target) not in global_index:
                raise ValueError("Global class not found: {}".format(target))
            mapping[local_index] = global_index[str(target)]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    specification = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    global_names = [str(name) for name in specification["global_class_names"]]
    global_index = {name: index for index, name in enumerate(global_names)}
    output = Path(args.output).resolve()
    split_records = {"train": [], "val": [], "test": []}
    statistics = {}

    for dataset in specification["datasets"]:
        dataset_name = str(dataset["name"])
        root = Path(os.path.expandvars(os.path.expanduser(dataset["root"]))).resolve()
        mapping = parse_mapping(dataset, global_index)
        statistics[dataset_name] = {}
        for split in ("train", "val", "test"):
            if split not in dataset:
                continue
            list_path = Path(dataset[split])
            if not list_path.is_absolute():
                list_path = root / list_path
            image_entries = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            converted = 0
            for sequence, entry in enumerate(image_entries):
                image = Path(entry)
                if not image.is_absolute():
                    image = root / image
                image = image.resolve()
                label = image_to_label(image)
                if not image.exists() or not label.exists():
                    raise FileNotFoundError("Missing image/label pair: {} / {}".format(image, label))
                stem = "{}_{:06d}_{}".format(dataset_name, sequence, image.stem)
                target_image = output / "images" / split / (stem + image.suffix.lower())
                target_label = output / "labels" / split / (stem + ".txt")
                link_or_copy(image, target_image, args.mode)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                converted_lines = []
                for line in label.read_text(encoding="utf-8").splitlines():
                    values = line.split()
                    if not values:
                        continue
                    local_class = int(float(values[0]))
                    if local_class not in mapping:
                        continue
                    values[0] = str(mapping[local_class])
                    converted_lines.append(" ".join(values))
                target_label.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
                split_records[split].append(target_image.as_posix())
                converted += 1
            statistics[dataset_name][split] = converted

    for split, records in split_records.items():
        if records:
            (output / "{}.txt".format(split)).write_text("\n".join(records) + "\n", encoding="utf-8")
    yaml_data = {
        "path": output.as_posix(),
        "train": "train.txt",
        "val": "val.txt" if split_records["val"] else "train.txt",
        "test": "test.txt" if split_records["test"] else ("val.txt" if split_records["val"] else "train.txt"),
        "nc": len(global_names),
        "names": global_names,
    }
    (output / "multidomain.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print("Dataset YAML: {}".format(output / "multidomain.yaml"))


if __name__ == "__main__":
    main()

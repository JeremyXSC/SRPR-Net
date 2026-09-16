"""Import the paper's Roboflow datasets into a reproducible YOLO-seg layout.

The importer accepts either a pre-downloaded Roboflow ZIP or a
ROBOFLOW_API_KEY. Credentials are read from the environment and are never
written to manifests or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    directory: str
    workspace: str
    project: str
    version: int
    expected_train: int
    expected_heldout: int
    expected_classes: Tuple[str, ...]


DATASETS: Dict[str, DatasetSpec] = {
    "wheat": DatasetSpec(
        key="wheat",
        directory="WheatIns",
        workspace="albara-shehadeh-o8han",
        project="test-oaige",
        version=1,
        expected_train=392,
        expected_heldout=170,
        expected_classes=("wheat",),
    ),
    "rwcell": DatasetSpec(
        key="rwcell",
        directory="RWCellIns",
        workspace="atri-gly8b",
        project="cell-p3pcx",
        version=1,
        expected_train=126,
        expected_heldout=212,
        expected_classes=("RBC", "WBC"),
    ),
}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError("Archive contains an unsafe path: {!r}".format(member.filename))
        handle.extractall(destination)


def _download_archive(spec: DatasetSpec, api_key: str, destination: Path) -> None:
    endpoint = (
        "https://api.roboflow.com/{}/{}/{}/yolov7"
        "?api_key={}".format(spec.workspace, spec.project, spec.version, api_key)
    )
    try:
        with urllib.request.urlopen(endpoint, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
        if "json" in content_type.lower() or payload.lstrip().startswith(b"{"):
            metadata = json.loads(payload.decode("utf-8"))
            download_url = metadata.get("export", {}).get("link") or metadata.get("link")
            if not download_url:
                raise RuntimeError("Roboflow did not return an export link.")
            with urllib.request.urlopen(download_url, timeout=120) as response:
                payload = response.read()
        destination.write_bytes(payload)
    except Exception:
        # Suppress the original URL-bearing exception because Roboflow embeds
        # the API key in the request query string.
        raise RuntimeError(
            "Roboflow download failed. Supply a downloaded ZIP with --archive "
            "or verify ROBOFLOW_API_KEY."
        ) from None


def _find_dataset_root(extracted: Path) -> Path:
    candidates = [path.parent for path in extracted.rglob("data.yaml")]
    candidates.extend(path.parent for path in extracted.rglob("dataset.yaml"))
    for candidate in sorted(set(candidates), key=lambda path: len(path.parts)):
        if (candidate / "train" / "images").is_dir():
            return candidate
    for candidate in [extracted, *[path for path in extracted.iterdir() if path.is_dir()]]:
        if (candidate / "train" / "images").is_dir():
            return candidate
    raise FileNotFoundError("Could not find a Roboflow train/images directory in the archive.")


def _read_names(root: Path) -> List[str]:
    metadata_path = root / "data.yaml"
    if not metadata_path.exists():
        metadata_path = root / "dataset.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError("Roboflow data.yaml is missing.")
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("Roboflow data.yaml does not contain a valid names list.")


def _images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_polygon_label(path: Path, class_count: int) -> int:
    """Validate one YOLO polygon file and return its instance count."""
    if not path.exists():
        raise FileNotFoundError("Missing label for image: {}".format(path))
    count = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(
                "{}:{} is not a YOLO polygon (class plus at least three x/y points).".format(
                    path, line_number
                )
            )
        try:
            class_id = int(float(fields[0]))
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError("{}:{} contains a non-numeric value.".format(path, line_number)) from exc
        if class_id < 0 or class_id >= class_count:
            raise ValueError("{}:{} has class id {} outside [0, {}).".format(
                path, line_number, class_id, class_count
            ))
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError("{}:{} has coordinates outside [0, 1].".format(path, line_number))
        count += 1
    return count


def _copy_split(
    source_root: Path,
    destination_root: Path,
    source_name: str,
    destination_name: str,
    class_count: int,
    mode: str,
) -> Tuple[List[Path], int]:
    source_images = _images(source_root / source_name / "images")
    source_labels = source_root / source_name / "labels"
    output_images = destination_root / "images" / destination_name
    output_labels = destination_root / "labels" / destination_name
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    instances = 0
    seen_names = set()
    for image in source_images:
        if image.name.lower() in seen_names:
            raise ValueError("Duplicate image name in {}: {}".format(source_name, image.name))
        seen_names.add(image.name.lower())
        label = source_labels / (image.stem + ".txt")
        instances += validate_polygon_label(label, class_count)
        image_target = output_images / image.name
        label_target = output_labels / label.name
        if mode == "symlink":
            os.symlink(image.resolve(), image_target)
            os.symlink(label.resolve(), label_target)
        else:
            shutil.copy2(image, image_target)
            shutil.copy2(label, label_target)
        output_paths.append(image_target.absolute())
    return output_paths, instances


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_list(path: Path, images: Iterable[Path]) -> None:
    path.write_text(
        "".join(image.as_posix() + "\n" for image in images),
        encoding="utf-8",
    )


def _check_expected_counts(
    spec: DatasetSpec,
    split_paths: Dict[str, Sequence[Path]],
    strict: bool,
) -> None:
    train_count = len(split_paths["train"])
    heldout_count = len(split_paths["val"]) + len(split_paths["test"])
    if strict and train_count != spec.expected_train:
        raise ValueError(
            "{} train count is {}, expected {}.".format(
                spec.key, train_count, spec.expected_train
            )
        )
    if strict and heldout_count != spec.expected_heldout:
        raise ValueError(
            "{} val+test count is {}, expected {}.".format(
                spec.key, heldout_count, spec.expected_heldout
            )
        )
    if not split_paths["val"] or not split_paths["test"]:
        raise ValueError("Independent Roboflow valid and test splits are required.")


def prepare_dataset(
    spec: DatasetSpec,
    archive: Path,
    output: Path,
    mode: str = "copy",
    strict: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    if output.exists():
        if not overwrite:
            raise FileExistsError("{} already exists; use --overwrite explicitly.".format(output))
        shutil.rmtree(output)
    with tempfile.TemporaryDirectory(prefix="blo_roboflow_") as temporary:
        extracted = Path(temporary)
        _safe_extract(archive, extracted)
        source_root = _find_dataset_root(extracted)
        names = _read_names(source_root)
        if len(names) != len(spec.expected_classes):
            raise ValueError(
                "{} has {} classes, expected {}: {}".format(
                    spec.key, len(names), len(spec.expected_classes), spec.expected_classes
                )
            )
        if len(names) > 1 and [name.casefold() for name in names] != [
            name.casefold() for name in spec.expected_classes
        ]:
            raise ValueError(
                "{} class order is {}, expected {}.".format(
                    spec.key, names, spec.expected_classes
                )
            )

        split_sources = {"train": "train", "val": "valid", "test": "test"}
        split_paths: Dict[str, List[Path]] = {}
        split_instances: Dict[str, int] = {}
        for destination_name, source_name in split_sources.items():
            paths, instances = _copy_split(
                source_root,
                output,
                source_name,
                destination_name,
                len(names),
                mode,
            )
            split_paths[destination_name] = paths
            split_instances[destination_name] = instances

    _check_expected_counts(spec, split_paths, strict)
    for split, paths in split_paths.items():
        _write_list(output / "{}.txt".format(split), paths)
    _write_list(output / "paper_test.txt", split_paths["val"] + split_paths["test"])

    dataset_yaml = {
        "path": output.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "paper_test": "paper_test.txt",
        "nc": len(names),
        "names": names,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "dataset": spec.key,
        "source": {
            "workspace": spec.workspace,
            "project": spec.project,
            "version": spec.version,
        },
        "class_names": names,
        "splits": {
            split: {
                "images": len(paths),
                "instances": split_instances[split],
                "sha256": {
                    image.relative_to(output.resolve()).as_posix(): _sha256(image)
                    for image in paths
                },
            }
            for split, paths in split_paths.items()
        },
        "paper_counts": {
            "train": spec.expected_train,
            "heldout": spec.expected_heldout,
        },
        "strict": bool(strict),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    output = args.output or Path("datasets") / spec.directory
    archive = args.archive
    temporary_archive = None
    if archive is None:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                "Provide --archive or set {}. No credential was written or logged.".format(
                    args.api_key_env
                )
            )
        handle = tempfile.NamedTemporaryFile(prefix="blo_dataset_", suffix=".zip", delete=False)
        handle.close()
        temporary_archive = Path(handle.name)
        _download_archive(spec, api_key, temporary_archive)
        archive = temporary_archive
    try:
        manifest = prepare_dataset(
            spec,
            archive.resolve(),
            output.resolve(),
            mode=args.mode,
            strict=not args.allow_count_mismatch,
            overwrite=args.overwrite,
        )
    finally:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

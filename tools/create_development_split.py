"""Create a deterministic train/validation split without touching test data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def create_split(
    source: Path,
    train_output: Path,
    val_output: Path,
    validation_count: int,
    seed: int,
) -> None:
    images = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if validation_count <= 0 or validation_count >= len(images):
        raise ValueError("validation_count must be between 1 and len(source)-1.")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    if set(training) & set(validation):
        raise AssertionError("Generated train and validation splits overlap.")
    train_output.write_text("".join(value + "\n" for value in training), encoding="utf-8")
    val_output.write_text("".join(value + "\n" for value in validation), encoding="utf-8")
    manifest = {
        "source": str(source.resolve()),
        "seed": int(seed),
        "training_images": len(training),
        "validation_images": len(validation),
        "train_output": str(train_output.resolve()),
        "val_output": str(val_output.resolve()),
    }
    train_output.with_name("development_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--validation-count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    create_split(
        args.source,
        args.train_output,
        args.val_output,
        args.validation_count,
        args.seed,
    )
    print("Saved {} and {}".format(args.train_output, args.val_output))


if __name__ == "__main__":
    main()

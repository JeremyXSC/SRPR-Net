"""Filter candidate masks into weak-supervision pseudo labels.

The tool consumes a COCO-like prediction JSON whose annotations contain:
image_id, category_id, segmentation, detection_score, sam_quality,
prior_compatibility and stability. It outputs a filtered COCO JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-detection", type=float, default=0.45)
    parser.add_argument("--min-sam-quality", type=float, default=0.70)
    parser.add_argument("--min-prior", type=float, default=0.10)
    parser.add_argument("--min-stability", type=float, default=0.80)
    parser.add_argument(
        "--allowed-tags",
        default="",
        help="Optional JSON mapping image_id to allowed category IDs",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    allowed = {}
    if args.allowed_tags:
        raw = json.loads(Path(args.allowed_tags).read_text(encoding="utf-8"))
        allowed = {str(key): {int(value) for value in values} for key, values in raw.items()}

    kept = []
    rejected = {"detection": 0, "sam": 0, "prior": 0, "stability": 0, "tag": 0}
    for annotation in data.get("annotations", []):
        if float(annotation.get("detection_score", annotation.get("score", 0.0))) < args.min_detection:
            rejected["detection"] += 1
            continue
        if float(annotation.get("sam_quality", 0.0)) < args.min_sam_quality:
            rejected["sam"] += 1
            continue
        if float(annotation.get("prior_compatibility", 1.0)) < args.min_prior:
            rejected["prior"] += 1
            continue
        if float(annotation.get("stability", 0.0)) < args.min_stability:
            rejected["stability"] += 1
            continue
        image_allowed = allowed.get(str(annotation["image_id"]))
        if image_allowed is not None and int(annotation["category_id"]) not in image_allowed:
            rejected["tag"] += 1
            continue
        copied = dict(annotation)
        copied["id"] = len(kept) + 1
        copied["iscrowd"] = int(copied.get("iscrowd", 0))
        kept.append(copied)

    output = {
        "images": data.get("images", []),
        "categories": data.get("categories", []),
        "annotations": kept,
        "filter_summary": {
            "input": len(data.get("annotations", [])),
            "kept": len(kept),
            "rejected": rejected,
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["filter_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

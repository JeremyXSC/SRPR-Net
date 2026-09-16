"""Build a Laplace-smoothed context-class prior JSON from a CSV table.

Input CSV columns: context,class_name,count
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--classes", required=True, help="Comma-separated ordered class names")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()

    class_names = [value.strip() for value in args.classes.split(",") if value.strip()]
    if not class_names:
        raise ValueError("At least one class name is required.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    counts = defaultdict(lambda: [float(args.smoothing)] * len(class_names))
    total = [float(args.smoothing)] * len(class_names)

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            context = row["context"].strip()
            class_name = row["class_name"].strip()
            if class_name not in class_to_index:
                raise ValueError("Unknown class_name: {}".format(class_name))
            value = float(row.get("count", 1.0))
            index = class_to_index[class_name]
            counts[context][index] += value
            total[index] += value

    def normalize(values):
        denominator = sum(values)
        return [value / denominator for value in values]

    output = {
        "class_names": class_names,
        "default": normalize(total),
        "contexts": {context: normalize(values) for context, values in sorted(counts.items())},
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved semantic prior to {}".format(destination))


if __name__ == "__main__":
    main()

"""Aggregate generalized BLO-Inst experiment metrics into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


COLUMNS = [
    "experiment",
    "box_precision",
    "box_recall",
    "box_map",
    "box_map50",
    "box_map75",
    "mask_precision",
    "mask_recall",
    "mask_map",
    "mask_map50",
    "mask_map75",
    "images",
    "instances",
    "bayes_alpha",
    "max_det",
    "inference_seconds",
    "fps",
    "parameters",
    "trainable_parameters",
    "peak_vram_mb",
    "training_seconds",
]
PER_CLASS_COLUMNS = [
    "experiment",
    "class_name",
    "box_precision",
    "box_recall",
    "box_map",
    "box_map50",
    "box_map75",
    "mask_precision",
    "mask_recall",
    "mask_map",
    "mask_map50",
    "mask_map75",
]
STAT_METRICS = [
    "box_map",
    "box_map50",
    "box_map75",
    "mask_map",
    "mask_map50",
    "mask_map75",
]


def find_training_seconds(metrics_path: Path, root: Path):
    """Sum epoch durations from the nearest enclosing history.csv."""
    directory = metrics_path.parent
    while True:
        history_path = directory / "history.csv"
        if history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
                seconds = [
                    float(row["seconds"])
                    for row in csv.DictReader(handle)
                    if row.get("seconds")
                ]
            return sum(seconds) if seconds else ""
        if directory == root or root not in directory.parents:
            return ""
        directory = directory.parent


def format_value(row, key):
    value = row[key]
    return "{:.4f}".format(float(value)) if isinstance(value, (float, int)) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--csv", default="experiment_summary.csv")
    parser.add_argument("--markdown", default="experiment_summary.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = []
    per_class_rows = []
    for path in sorted(root.rglob("metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {column: data.get(column, "") for column in COLUMNS}
        row["experiment"] = str(path.parent.relative_to(root))
        row["training_seconds"] = find_training_seconds(path, root)
        rows.append(row)

        for class_name, class_metrics in sorted(data.get("per_class", {}).items()):
            class_row = {
                column: class_metrics.get(column, "")
                for column in PER_CLASS_COLUMNS
            }
            class_row["experiment"] = row["experiment"]
            class_row["class_name"] = class_name
            per_class_rows.append(class_row)

    if not rows:
        raise FileNotFoundError("No metrics.json files were found under {}".format(root))

    csv_path = root / args.csv
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if per_class_rows:
        with (root / "per_class_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_CLASS_COLUMNS)
            writer.writeheader()
            writer.writerows(per_class_rows)

    md_path = root / args.markdown
    lines = [
        "| 实验 | Box mAP | Box AP50 | Box AP75 | Mask mAP | Mask AP50 | Mask AP75 | 图像数 | 实例数 | α | 训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = {key: format_value(row, key) for key in COLUMNS}
        lines.append(
            "| {experiment} | {box_map} | {box_map50} | {box_map75} | "
            "{mask_map} | {mask_map50} | {mask_map75} | {images} | "
            "{instances} | {bayes_alpha} | {training_seconds} |".format(**values)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    groups = {}
    for row in rows:
        relative = row["experiment"].replace("\\", "/")
        match = re.match(r"(.+)_seed\d+/(final_test(?:_tta)?)$", relative)
        if match:
            key = "{}/{}".format(match.group(1), match.group(2))
            groups.setdefault(key, []).append(row)

    summary_rows = []
    for experiment, group_rows in sorted(groups.items()):
        summary = {"experiment": experiment, "seeds": len(group_rows)}
        for metric in STAT_METRICS:
            values = [float(row[metric]) for row in group_rows if row[metric] != ""]
            summary[metric + "_mean"] = statistics.mean(values) if values else ""
            summary[metric + "_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)

    if summary_rows:
        with (root / "three_seed_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        summary_lines = [
            "| 实验 | 种子数 | Mask mAP | Mask AP50 | Mask AP75 | Box mAP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary_rows:
            summary_lines.append(
                "| {experiment} | {seeds} | {mask_map_mean:.4f} ± {mask_map_std:.4f} | "
                "{mask_map50_mean:.4f} ± {mask_map50_std:.4f} | "
                "{mask_map75_mean:.4f} ± {mask_map75_std:.4f} | "
                "{box_map_mean:.4f} ± {box_map_std:.4f} |".format(**row)
            )
        (root / "three_seed_summary.md").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

    print("Saved {} and {}".format(csv_path, md_path))


if __name__ == "__main__":
    main()

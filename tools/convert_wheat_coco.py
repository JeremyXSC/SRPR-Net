import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "downloads" / "wheat_v2_coco"
DST = ROOT / "datasets" / "WheatIns"

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

if DST.exists():
    shutil.rmtree(DST)

for src_split, dst_split in SPLITS.items():
    src_dir = SRC / src_split
    json_path = src_dir / "_annotations.coco.json"

    data = json.loads(json_path.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data["images"]}
    categories = sorted(data["categories"], key=lambda x: x["id"])
    class_map = {c["id"]: i for i, c in enumerate(categories)}

    anns = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            anns[ann["image_id"]].append(ann)

    image_dir = DST / "images" / dst_split
    label_dir = DST / "labels" / dst_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    list_lines = []

    for image_id, img in images.items():
        filename = img["file_name"]
        width = float(img["width"])
        height = float(img["height"])

        src_image = src_dir / filename
        dst_image = image_dir / Path(filename).name
        shutil.copy2(src_image, dst_image)

        label_path = label_dir / (dst_image.stem + ".txt")
        rows = []

        for ann in anns.get(image_id, []):
            seg = ann.get("segmentation", [])

            if isinstance(seg, dict):
                raise RuntimeError(
                    f"发现 RLE Mask，当前脚本不做静默转换：{filename}"
                )

            polygons = [
                p for p in seg
                if isinstance(p, list) and len(p) >= 6
            ]

            if len(polygons) > 1:
                raise RuntimeError(
                    f"发现一个实例包含多个 Polygon：{filename}，请停止并检查。"
                )

            if not polygons:
                continue

            polygon = polygons[0]
            cls = class_map[ann["category_id"]]

            coords = []
            for i in range(0, len(polygon), 2):
                x = min(max(polygon[i] / width, 0.0), 1.0)
                y = min(max(polygon[i + 1] / height, 0.0), 1.0)
                coords.extend([x, y])

            rows.append(
                str(cls) + " " + " ".join(f"{v:.6f}" for v in coords)
            )

        label_path.write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

        list_lines.append(f"./images/{dst_split}/{dst_image.name}")

    (DST / f"{dst_split}.txt").write_text(
        "\n".join(list_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"{dst_split}: images={len(images)}, "
        f"instances={sum(len(v) for v in anns.values())}"
    )

manifest = {
    "source": "Roboflow Wheat v2 COCO Segmentation",
    "splits": {
        s: sum(1 for _ in (DST / f"{s}.txt").open())
        for s in ["train", "val", "test"]
    },
}
(DST / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("转换完成：", DST)
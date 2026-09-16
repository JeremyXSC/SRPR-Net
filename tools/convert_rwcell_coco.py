import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "downloads" / "rwcell_v2_coco"
DST = ROOT / "datasets" / "RWCellIns"

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

# 不依赖 Roboflow 原始 category_id，按类别名称重新编号
NAME_TO_CLASS = {
    "rbc": 0,
    "wbc": 1,
}

if DST.exists():
    shutil.rmtree(DST)

for src_split, dst_split in SPLITS.items():
    src_dir = SRC / src_split
    json_path = src_dir / "_annotations.coco.json"

    data = json.loads(json_path.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data["images"]}

    class_map = {}
    for c in data["categories"]:
        name = c["name"].strip().lower()
        if name in NAME_TO_CLASS:
            class_map[c["id"]] = NAME_TO_CLASS[name]

    anns = defaultdict(list)

    for ann in data["annotations"]:
        cid = ann["category_id"]

        if cid not in class_map:
            raise RuntimeError(
                f"发现未知类别ID {cid}，请检查 categories。"
            )

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
                    f"发现 RLE mask：{filename}"
                )

            polygons = [
                p for p in seg
                if isinstance(p, list) and len(p) >= 6
            ]

            if len(polygons) > 1:
                raise RuntimeError(
                    f"一个实例存在多个 Polygon：{filename}"
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
                str(cls) + " " +
                " ".join(f"{v:.6f}" for v in coords)
            )

        label_path.write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8"
        )

        list_lines.append(
            f"./images/{dst_split}/{dst_image.name}"
        )

    (DST / f"{dst_split}.txt").write_text(
        "\n".join(list_lines) + "\n",
        encoding="utf-8"
    )

    print(
        f"{dst_split}: images={len(images)}, "
        f"instances={sum(len(v) for v in anns.values())}"
    )

manifest = {
    "source": "Roboflow RWCell v2 COCO Segmentation",
    "splits": {}
}

for s in ["train", "val", "test"]:
    images = sum(
        1 for x in open(DST / f"{s}.txt", encoding="utf-8")
        if x.strip()
    )

    instances = sum(
        sum(1 for x in p.open(encoding="utf-8") if x.strip())
        for p in (DST / "labels" / s).glob("*.txt")
    )

    manifest["splits"][s] = {
        "images": images,
        "instances": instances
    }

(DST / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("转换完成：", DST)
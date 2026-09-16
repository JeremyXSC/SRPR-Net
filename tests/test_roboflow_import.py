from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import yaml

from tools.prepare_roboflow_dataset import (
    DATASETS,
    _download_archive,
    prepare_dataset,
    validate_polygon_label,
)


class RoboflowImportTests(unittest.TestCase):
    def test_download_error_does_not_expose_api_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "tools.prepare_roboflow_dataset.urllib.request.urlopen",
                side_effect=RuntimeError(
                    "https://api.roboflow.com/export?api_key=super-secret"
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    _download_archive(
                        DATASETS["wheat"],
                        "super-secret",
                        Path(temporary) / "dataset.zip",
                    )
        self.assertNotIn("super-secret", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_polygon_validation_rejects_box_only_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "bad.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_polygon_label(label, class_count=1)

    def test_import_creates_disjoint_splits_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "data.yaml").write_text(
                yaml.safe_dump({"nc": 1, "names": ["wheat"]}),
                encoding="utf-8",
            )
            for split in ("train", "valid", "test"):
                images = source / split / "images"
                labels = source / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / "{}.jpg".format(split)).write_bytes(b"image")
                (labels / "{}.txt".format(split)).write_text(
                    "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
                    encoding="utf-8",
                )
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for path in source.rglob("*"):
                    if path.is_file():
                        handle.write(path, path.relative_to(root))
            output = root / "prepared"
            manifest = prepare_dataset(
                DATASETS["wheat"],
                archive,
                output,
                strict=False,
            )
            split_sets = {
                split: set((output / "{}.txt".format(split)).read_text(encoding="utf-8").splitlines())
                for split in ("train", "val", "test")
            }
            self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
            self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
            self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))
            self.assertEqual(manifest["splits"]["train"]["images"], 1)
            stored = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["class_names"], ["wheat"])


if __name__ == "__main__":
    unittest.main()

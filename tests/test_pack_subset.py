"""Tests for tools/pack_subset.py."""

import csv
import zipfile
from pathlib import Path

from PIL import Image

from tools.pack_subset import main


def make_split(tmp_path: Path) -> tuple[Path, Path]:
    """Create two classes with images and a small split file."""
    image_root = tmp_path / "images"
    rows = []
    for label in ("A", "B"):
        folder = image_root / label
        folder.mkdir(parents=True)
        for number in range(2):
            image = Image.new("RGB", (8, 8), color=(number * 100, 0, 0))
            image.save(folder / f"{label}{number}.png")
            split = "train" if number == 0 else "test"
            rows.append({"path": f"{label}/{label}{number}.png", "split": split})
    split_file = tmp_path / "split.csv"
    with split_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return split_file, image_root


def test_zip_contains_only_selected_split(tmp_path: Path) -> None:
    split_file, image_root = make_split(tmp_path)
    output = tmp_path / "subset.zip"
    code = main(
        [
            "--split-file",
            str(split_file),
            "--image-root",
            str(image_root),
            "--output",
            str(output),
            "--splits",
            "train",
        ]
    )
    assert code == 0
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["A/A0.png", "B/B0.png"]


def test_missing_image_returns_error(tmp_path: Path) -> None:
    split_file, image_root = make_split(tmp_path)
    (image_root / "A" / "A0.png").unlink()
    code = main(
        [
            "--split-file",
            str(split_file),
            "--image-root",
            str(image_root),
            "--output",
            str(tmp_path / "subset.zip"),
        ]
    )
    assert code == 1

"""Tests for tools/check_split.py on synthetic split files."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from tests.cnn_helpers import make_image
from tools.check_split import main, read_rows, validate_rows

COLUMNS = ["path", "label", "group_id", "split"]
LABELS = {"A": 0, "B": 1}


def make_valid_split(root: Path) -> tuple[Path, Path]:
    """Create 20 images per class split 14/3/3, plus the matching CSV."""
    image_root = root / "images"
    rows = []
    for label in LABELS:
        for split, count in (("train", 14), ("val", 3), ("test", 3)):
            for number in range(count):
                relative = f"{label}/{label}_{split}_{number}.png"
                make_image(image_root / relative, seed=len(rows))
                rows.append(
                    {
                        "path": relative,
                        "label": label,
                        "group_id": f"{label}_{split}_{number}",
                        "split": split,
                    }
                )
    split_file = root / "split.csv"
    with split_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return split_file, image_root


def write_config(root: Path, split_file: Path, image_root: Path) -> Path:
    """Write a minimal cnn.yaml with only the data keys check_split reads."""
    config = {
        "data": {
            "split_file": str(split_file),
            "image_root": str(image_root),
            "label_map": str(root / "label_map.json"),
        }
    }
    path = root / "cnn.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_valid_split_passes(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    report = validate_rows(read_rows(split_file), LABELS, image_root, 0.0, 0.0)
    assert report.ok
    assert report.counts == {"train": 28, "val": 6, "test": 6}
    assert report.ratios == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert report.per_class["A"] == {"train": 14, "val": 3, "test": 3}


def test_missing_image_is_reported(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    (image_root / "A" / "A_train_0.png").unlink()
    report = validate_rows(read_rows(split_file), LABELS, image_root, 0.0, 0.0)
    assert not report.ok
    assert any("image not found" in problem for problem in report.problems)


def test_duplicate_path_is_reported(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    rows = read_rows(split_file)
    rows[1]["path"] = rows[0]["path"]
    report = validate_rows(rows, LABELS, image_root, 0.0, 0.0)
    assert any("duplicate path" in problem for problem in report.problems)


def test_group_in_two_splits_is_reported(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    rows = read_rows(split_file)
    rows[0]["group_id"] = "shared"
    rows[-1]["group_id"] = "shared"
    report = validate_rows(rows, LABELS, image_root, 0.0, 0.0)
    assert any("more than one split" in problem for problem in report.problems)


def test_bad_label_split_and_ratio_are_reported(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    rows = read_rows(split_file)
    rows[0]["label"] = "Q"
    rows[1]["split"] = "holdout"
    report = validate_rows(rows, LABELS, image_root, 0.0, 0.0)
    assert any("not in label_map" in problem for problem in report.problems)
    assert any("split must be one of" in problem for problem in report.problems)
    assert any("overall: train share" in problem for problem in report.problems)


def test_main_exit_codes(tmp_path: Path) -> None:
    split_file, image_root = make_valid_split(tmp_path)
    label_map = tmp_path / "label_map.json"
    label_map.write_text('{"A": 0, "B": 1}', encoding="utf-8")
    config = write_config(tmp_path, split_file, image_root)
    assert main(["--config", str(config), "--tolerance", "0", "--class-tolerance", "0"]) == 0
    rows = read_rows(split_file)
    rows[0]["label"] = "Q"
    with split_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in COLUMNS} for row in rows)
    assert main(["--config", str(config), "--tolerance", "0", "--class-tolerance", "0"]) == 1

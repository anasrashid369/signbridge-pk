"""Tests for tools/split_leakage_report.py on synthetic images."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_scripts.clean.duplicates import hamming_distances
from signbridge.config import REPO_ROOT
from tests.image_helpers import make_pattern
from tools.split_leakage_report import (
    build_report,
    hashes_for,
    load_hash_cache,
    main,
    read_split,
    save_hash_cache,
)

COLUMNS = ["path", "label", "group_id", "split"]


def write_split(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a split CSV with the contract columns."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_gradient(path: Path) -> Path:
    """Save a left-to-right gradient, whose dHash is all ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = np.arange(90, dtype=np.uint8)
    Image.fromarray(np.tile(row, (80, 1))).save(path)
    return path


def save_black(path: Path) -> Path:
    """Save a uniform image, whose dHash is zero."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((80, 90), dtype=np.uint8)).save(path)
    return path


def save_pattern(path: Path, seed: int) -> Path:
    """Save a block pattern after creating its parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return make_pattern(path, seed=seed)


def test_identical_images_have_zero_distance(tmp_path: Path) -> None:
    root = tmp_path / "images"
    original = save_pattern(root / "A" / "train.png", seed=1)
    copy = root / "A" / "val.png"
    copy.write_bytes(original.read_bytes())
    rows = [
        {"path": "A/train.png", "label": "A", "group_id": "A_g0", "split": "train"},
        {"path": "A/val.png", "label": "A", "group_id": "A_g1", "split": "val"},
    ]
    hashes = hashes_for(root, ["A/train.png", "A/val.png"], {})
    report = build_report(rows, hashes)
    assert [row["scope"] for row in report] == ["overall", "A"]
    assert report[0]["scope"] == "overall"
    assert report[0]["split"] == "val"
    assert report[0]["min"] == 0
    assert report[0]["frac_le_4"] == 1.0


def test_only_the_same_class_is_compared(tmp_path: Path) -> None:
    root = tmp_path / "images"
    save_black(root / "A" / "train.png")
    save_gradient(root / "B" / "train.png")
    save_gradient(root / "A" / "val.png")
    rows = [
        {"path": "A/train.png", "label": "A", "group_id": "A_g0", "split": "train"},
        {"path": "B/train.png", "label": "B", "group_id": "B_g0", "split": "train"},
        {"path": "A/val.png", "label": "A", "group_id": "A_g1", "split": "val"},
    ]
    hashes = hashes_for(root, [row["path"] for row in rows], {})
    report = build_report(rows, hashes)
    assert [row["scope"] for row in report] == ["overall", "A"]
    assert report[1]["min"] == 64
    assert report[1]["frac_le_4"] == 0.0


def test_statistics_match_the_distances() -> None:
    rows = [
        {"path": "A/h0.jpg", "label": "A", "group_id": "A_g0", "split": "train"},
        {"path": "A/h15.jpg", "label": "A", "group_id": "A_g1", "split": "train"},
        {"path": "A/h3.jpg", "label": "A", "group_id": "A_g2", "split": "val"},
        {"path": "A/h7.jpg", "label": "A", "group_id": "A_g3", "split": "val"},
        {"path": "A/h255.jpg", "label": "A", "group_id": "A_g4", "split": "test"},
    ]
    hashes = {
        "A/h0.jpg": 0,
        "A/h15.jpg": 15,
        "A/h3.jpg": 3,
        "A/h7.jpg": 7,
        "A/h255.jpg": 255,
    }
    train = np.asarray([0, 15], dtype=np.uint64)
    val_distances = [int(hamming_distances(train, value).min()) for value in (3, 7)]
    test_distances = [int(hamming_distances(train, 255).min())]
    report = build_report(rows, hashes)
    val = next(row for row in report if row["scope"] == "overall" and row["split"] == "val")
    test = next(row for row in report if row["scope"] == "overall" and row["split"] == "test")
    assert val["images"] == 2
    assert val["min"] == min(val_distances)
    assert val["median"] == float(np.median(val_distances))
    assert val["frac_le_8"] == float(np.mean(np.asarray(val_distances) <= 8))
    assert test["images"] == 1
    assert test["min"] == test_distances[0]


def test_writes_text_and_csv(tmp_path: Path) -> None:
    root = tmp_path / "images"
    original = save_pattern(root / "A" / "train.png", seed=2)
    (root / "A" / "test.png").write_bytes(original.read_bytes())
    split = write_split(
        tmp_path / "split.csv",
        [
            {"path": "A/train.png", "label": "A", "group_id": "A_g0", "split": "train"},
            {"path": "A/test.png", "label": "A", "group_id": "A_g1", "split": "test"},
        ],
    )
    output = tmp_path / "out"
    code = main(
        ["--split-file", str(split), "--image-root", str(root), "--output-dir", str(output)]
    )
    assert code == 0
    csv_path = output / "split_leakage.csv"
    text_path = output / "split_leakage.txt"
    assert csv_path.is_file() and text_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert records[0]["scope"] == "overall"
    assert records[0]["split"] == "test"
    assert records[0]["min"] == "0"
    assert "overall" in text_path.read_text(encoding="utf-8")


def test_refuses_to_write_inside_the_repository(tmp_path: Path) -> None:
    root = tmp_path / "images"
    save_pattern(root / "A" / "train.png", seed=3)
    split = write_split(
        tmp_path / "split.csv",
        [{"path": "A/train.png", "label": "A", "group_id": "A_g0", "split": "train"}],
    )
    inside = REPO_ROOT / "leakage_must_not_be_written"
    code = main(
        ["--split-file", str(split), "--image-root", str(root), "--output-dir", str(inside)]
    )
    assert code == 1
    assert not inside.exists()


def test_hash_cache_round_trip(tmp_path: Path) -> None:
    cache = {"A/a.jpg": 5, "A/b.jpg": 1 << 63}
    path = tmp_path / "cache.npz"
    save_hash_cache(path, cache)
    assert load_hash_cache(path) == cache


def test_class_without_train_images_is_skipped() -> None:
    rows = [{"path": "A/h1.jpg", "label": "A", "group_id": "A_g0", "split": "val"}]
    assert build_report(rows, {"A/h1.jpg": 1}) == []


def test_missing_image_is_skipped(tmp_path: Path) -> None:
    hashes = hashes_for(tmp_path, ["A/missing.png"], {})
    assert hashes == {}


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("path,label\nA/A1.jpg,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_split(path)


def test_help_exits_cleanly() -> None:
    from tools.split_leakage_report import parse_args

    with pytest.raises(SystemExit) as info:
        parse_args(["--help"])
    assert info.value.code == 0


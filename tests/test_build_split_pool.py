"""Tests for data_scripts/build_split_pool.py."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from data_scripts import build_split_pool
from data_scripts.build_split_pool import group_by_distance, group_by_leader, sample_rows
from tests.image_helpers import make_pattern


def write_config(tmp_path: Path, images_per_class: int = 0) -> Path:
    """Write a config whose output paths point inside tmp_path."""
    config = {
        "seed": 42,
        "paths": {
            "label_map": str(tmp_path / "label_map.json"),
            "split_pool": str(tmp_path / "interim" / "pool.csv"),
            "split_summary": str(tmp_path / "reports" / "summary.csv"),
        },
        "quality": {
            "blur_threshold": 100.0,
            "brightness_min": 40,
            "brightness_max": 220,
            "min_side": 64,
            "near_duplicate_distance": 4,
        },
        "split": {"images_per_class": images_per_class, "group_distance": 10},
    }
    (tmp_path / "label_map.json").write_text('{"A": 0, "B": 1}', encoding="utf-8")
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def make_images(tmp_path: Path) -> Path:
    """Class A: 3 distinct images, 1 exact copy, 1 near-duplicate, 1 unreadable. Class B: 2."""
    root = tmp_path / "train"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir()
    good = make_pattern(root / "A" / "a1.png", seed=1)
    (root / "A" / "a2.png").write_bytes(good.read_bytes())
    make_pattern(root / "A" / "a3.png", seed=1, noise=2.0)
    make_pattern(root / "A" / "a4.png", seed=2)
    make_pattern(root / "A" / "a5.png", seed=3)
    (root / "A" / "a6.png").write_bytes(b"")
    make_pattern(root / "B" / "b1.png", seed=7)
    make_pattern(root / "B" / "b2.png", seed=8)
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_group_by_distance_links_chains_and_keeps_far_images_apart() -> None:
    # 0 and 0b111 differ by 3 bits, 0b111 and 0b111111 by 3 bits, but 0 and 0b111111 by 6.
    items = [("a", 0), ("b", 0b111), ("c", 0b111111), ("d", (1 << 64) - 1)]
    groups = group_by_distance(items, 3, "A")
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
    assert groups["a"].startswith("A_g")


def test_sample_rows_is_reproducible_and_limits() -> None:
    rows = [{"path": f"A/{i}.jpg", "label": "A", "group_id": f"A_g{i}"} for i in range(20)]
    first = sample_rows(rows, 5, 42, "A")
    assert first == sample_rows(rows, 5, 42, "A")
    assert len(first) == 5
    assert sample_rows(rows, 0, 42, "A") == rows


def test_run_removes_problems_and_writes_pool_and_summary(tmp_path: Path) -> None:
    root = make_images(tmp_path)
    config = write_config(tmp_path)
    code = build_split_pool.main(["--config", str(config), "--image-root", str(root)])
    assert code == 0
    pool = read_csv(tmp_path / "interim" / "pool.csv")
    paths = {row["path"] for row in pool}
    assert "A/a1.png" in paths and "A/a4.png" in paths and "B/b1.png" in paths
    assert not paths & {"A/a2.png", "A/a3.png", "A/a6.png"}
    summary = {row["label"]: row for row in read_csv(tmp_path / "reports" / "summary.csv")}
    assert summary["A"]["found"] == "6"
    assert summary["A"]["unreadable"] == "1"
    assert summary["A"]["exact_duplicates"] == "1"
    assert summary["A"]["near_duplicates"] == "1"
    assert summary["A"]["after_cleaning"] == "3"


def test_rerunning_one_class_replaces_only_its_rows(tmp_path: Path) -> None:
    root = make_images(tmp_path)
    config = write_config(tmp_path)
    assert build_split_pool.main(["--config", str(config), "--image-root", str(root)]) == 0
    args = ["--config", str(config), "--image-root", str(root), "--classes", "B"]
    assert build_split_pool.main(args) == 0
    pool = read_csv(tmp_path / "interim" / "pool.csv")
    assert sorted(row["label"] for row in pool).count("A") == 3
    assert sorted(row["label"] for row in pool).count("B") == 2


def test_images_per_class_limits_each_class(tmp_path: Path) -> None:
    root = make_images(tmp_path)
    config = write_config(tmp_path)
    args = ["--config", str(config), "--image-root", str(root), "--images-per-class", "2"]
    assert build_split_pool.main(args) == 0
    pool = read_csv(tmp_path / "interim" / "pool.csv")
    assert [row["label"] for row in pool].count("A") == 2


def test_missing_class_folder_returns_error(tmp_path: Path) -> None:
    root = make_images(tmp_path)
    config = write_config(tmp_path)
    args = ["--config", str(config), "--image-root", str(root), "--classes", "Z"]
    assert build_split_pool.main(args) == 1


def test_group_by_leader_does_not_chain_a_long_sequence() -> None:
    """Hashes 0, 3, 6, 9, 12 chain into one group, but leader grouping splits them."""
    items = [(f"A/{i}.png", (1 << (3 * i)) - 1) for i in range(5)]
    chained = group_by_distance(items, 3, "A")
    leader = group_by_leader(items, 3, "A")
    assert len(set(chained.values())) == 1
    assert len(set(leader.values())) == 3
    assert leader["A/0.png"] == leader["A/1.png"]
    assert leader["A/1.png"] != leader["A/2.png"]


def test_group_by_leader_empty_input() -> None:
    """No images means no groups."""
    assert group_by_leader([], 10, "A") == {}

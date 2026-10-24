"""Tests for data_scripts/build_manifest.py."""

import csv
from pathlib import Path

import pytest

from data_scripts import build_manifest
from data_scripts.build_manifest import (
    assign_splits,
    build_rows,
    check_no_leakage,
    parse_filename,
    scan_images,
)
from signbridge.config import DEFAULT_CONFIG_PATH, load_config

LIGHTING = {"normal", "dim", "bright"}
BACKGROUND = {"plain", "cluttered"}


def make_config(seed: int = 42) -> dict:
    """Return a minimal config with the manifest settings."""
    return {
        "seed": seed,
        "manifest": {
            "lighting_values": sorted(LIGHTING),
            "background_values": sorted(BACKGROUND),
            "val_fraction": 0.2,
            "test_fraction": 0.2,
        },
    }


def make_psl_folder(root: Path, people: int = 5, per_person: int = 4) -> Path:
    """Create empty files named like real PSL photos for several people."""
    psl_dir = root / "psl"
    psl_dir.mkdir()
    for person in range(1, people + 1):
        for index in range(per_person):
            name = f"p{person:02d}_alif_dim_plain_{index:03d}.jpg"
            (psl_dir / name).write_bytes(b"")
    return psl_dir


def test_parse_filename_reads_fields() -> None:
    row = parse_filename("p03_alif_dim_plain_004.jpg", LIGHTING, BACKGROUND)
    assert row == {
        "label": "alif",
        "person_id": "p03",
        "lighting": "dim",
        "background": "plain",
    }


@pytest.mark.parametrize(
    "name",
    [
        "p03_alif_dim_plain.jpg",
        "x03_alif_dim_plain_004.jpg",
        "p03_alif_dark_plain_004.jpg",
        "p03_alif_dim_busy_004.jpg",
        "p03_alif_dim_plain_abc.jpg",
        "p03_al_if_dim_plain_004.jpg",
        "p03__dim_plain_004.jpg",
    ],
)
def test_parse_filename_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_filename(name, LIGHTING, BACKGROUND)


def test_assign_splits_keeps_each_person_in_one_split() -> None:
    people = [f"p{number:02d}" for number in range(1, 9)]
    splits = assign_splits(people * 3, 0.2, 0.2, seed=1)
    assert set(splits) == set(people)
    assert set(splits.values()) == {"train", "val", "test"}


def test_assign_splits_is_reproducible() -> None:
    people = [f"p{number:02d}" for number in range(1, 8)]
    assert assign_splits(people, 0.2, 0.2, seed=7) == assign_splits(people, 0.2, 0.2, seed=7)


def test_assign_splits_needs_enough_people() -> None:
    with pytest.raises(ValueError):
        assign_splits(["p01", "p02"], 0.2, 0.2, seed=1)


def test_assign_splits_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        assign_splits(["p01", "p02", "p03"], 0.6, 0.6, seed=1)


def test_no_person_appears_in_more_than_one_split(tmp_path: Path) -> None:
    psl_dir = make_psl_folder(tmp_path, people=6)
    rows = build_rows(make_config(), psl_dir, root=tmp_path)
    splits_by_person: dict[str, set[str]] = {}
    for row in rows:
        splits_by_person.setdefault(row["person_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_person.values())


def test_check_no_leakage_fails_when_person_is_in_two_splits() -> None:
    rows = [
        {"person_id": "p01", "split": "train"},
        {"person_id": "p01", "split": "test"},
    ]
    with pytest.raises(ValueError, match="p01"):
        check_no_leakage(rows)


def test_scan_images_skips_badly_named_files(tmp_path: Path) -> None:
    psl_dir = make_psl_folder(tmp_path, people=3, per_person=1)
    (psl_dir / "IMG_1234.jpg").write_bytes(b"")
    (psl_dir / "notes.txt").write_bytes(b"")
    rows = scan_images(psl_dir, LIGHTING, BACKGROUND, tmp_path)
    assert len(rows) == 3


def test_scan_images_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_images(tmp_path / "missing", LIGHTING, BACKGROUND, tmp_path)


def test_scan_images_with_no_valid_images_raises(tmp_path: Path) -> None:
    psl_dir = tmp_path / "psl"
    psl_dir.mkdir()
    (psl_dir / "IMG_1.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        scan_images(psl_dir, LIGHTING, BACKGROUND, tmp_path)


def test_main_writes_manifest_with_protocol_columns(tmp_path: Path) -> None:
    psl_dir = make_psl_folder(tmp_path, people=5)
    output = tmp_path / "out" / "manifest.csv"
    code = build_manifest.main(
        ["--config", str(DEFAULT_CONFIG_PATH), "--psl-dir", str(psl_dir), "--output", str(output)]
    )
    assert code == 0
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["path", "label", "person_id", "split", "lighting", "background"]
    assert len(rows) == 20


def test_main_returns_error_for_missing_folder(tmp_path: Path) -> None:
    code = build_manifest.main(
        ["--psl-dir", str(tmp_path / "nope"), "--output", str(tmp_path / "m.csv")]
    )
    assert code == 1


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as info:
        build_manifest.parse_args(["--help"])
    assert info.value.code == 0


def test_repo_config_has_manifest_settings() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert {"lighting_values", "background_values", "val_fraction", "test_fraction"} <= set(
        config["manifest"]
    )

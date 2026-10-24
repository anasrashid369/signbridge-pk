"""Tests for data_scripts/clean/clean_images.py."""

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from data_scripts.clean import clean_images
from data_scripts.clean.clean_images import (
    find_records,
    new_record,
    sample_per_class,
    summarize,
)
from tests.image_helpers import make_pattern


def write_config(tmp_path: Path) -> Path:
    """Write a config whose paths all point inside tmp_path."""
    config = {
        "seed": 42,
        "paths": {
            "asl_raw_dir": "raw/asl",
            "psl_raw_dir": "raw/psl",
            "cleaning_report": "interim/cleaning_report.csv",
            "cleaning_summary": "reports/cleaning_summary.csv",
            "cleaning_table": "reports/cleaning_summary_table.tex",
        },
        "dataset": {"asl_train_folder": "asl_alphabet_train"},
        "manifest": {
            "lighting_values": ["normal", "dim", "bright"],
            "background_values": ["plain", "cluttered"],
        },
        "quality": {
            "blur_threshold": 100.0,
            "brightness_min": 40,
            "brightness_max": 220,
            "min_side": 64,
            "near_duplicate_distance": 4,
        },
    }
    for key, value in config["paths"].items():
        config["paths"][key] = str(tmp_path / value)
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def make_asl_folder(tmp_path: Path) -> Path:
    """Create a small ASL-like folder with one example of every problem."""
    train = tmp_path / "raw" / "asl" / "asl_alphabet_train"
    (train / "A").mkdir(parents=True)
    (train / "B").mkdir()
    (tmp_path / "raw" / "asl" / "asl_alphabet_test").mkdir()
    good = make_pattern(train / "A" / "a1.png", seed=1)
    (train / "A" / "a2.png").write_bytes(good.read_bytes())  # exact duplicate
    make_pattern(train / "A" / "a3.png", seed=1, noise=2.0)  # near-duplicate
    Image.fromarray(np.full((80, 90), 128, dtype=np.uint8)).save(train / "A" / "a4.png")  # blurry
    (train / "A" / "a5.png").write_bytes(b"")  # unreadable
    make_pattern(train / "B" / "b1.png", seed=5)
    make_pattern(train / "B" / "b2.png", seed=9, low=0, high=40)  # too dark
    make_pattern(tmp_path / "raw" / "asl" / "asl_alphabet_test" / "x_test.png", seed=11)
    return train


def make_psl_folder(tmp_path: Path) -> Path:
    """Create a small PSL folder with two people and one badly named file."""
    psl = tmp_path / "raw" / "psl"
    psl.mkdir(parents=True)
    make_pattern(psl / "p01_alif_dim_plain_001.png", seed=21)
    make_pattern(psl / "p02_alif_normal_plain_001.png", seed=22)
    make_pattern(psl / "IMG_0001.png", seed=23)
    return psl


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_find_records_labels_asl_images_by_folder_and_skips_test_folder(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    records = find_records("asl", tmp_path / "raw" / "asl", config, root=tmp_path)
    assert sorted({record["label"] for record in records}) == ["A", "B"]
    assert len(records) == 7
    assert all(record["path"].startswith("raw/asl/asl_alphabet_train") for record in records)


def test_find_records_reads_psl_labels_from_file_names(tmp_path: Path) -> None:
    make_psl_folder(tmp_path)
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    records = find_records("psl", tmp_path / "raw" / "psl", config, root=tmp_path)
    by_name = {Path(record["path"]).name: record for record in records}
    assert by_name["p01_alif_dim_plain_001.png"]["person_id"] == "p01"
    assert by_name["p01_alif_dim_plain_001.png"]["label"] == "alif"
    assert by_name["IMG_0001.png"]["no_label"] == 1


def test_find_records_rejects_unknown_source_and_missing_folder(tmp_path: Path) -> None:
    config = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        find_records("xyz", tmp_path, config)
    with pytest.raises(FileNotFoundError):
        find_records("asl", tmp_path / "missing", config)


def test_sample_per_class_limits_and_is_reproducible() -> None:
    records = []
    for label in ("A", "B"):
        for index in range(10):
            record = new_record(f"{label}/{index:02d}.png", "asl")
            record["label"] = label
            records.append(record)
    first = sample_per_class(records, 3, seed=42)
    second = sample_per_class(records, 3, seed=42)
    assert [r["path"] for r in first] == [r["path"] for r in second]
    assert len(first) == 6
    assert len(sample_per_class(records, 0, seed=42)) == 20


def test_asl_run_flags_each_problem_and_counts_steps(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    config = write_config(tmp_path)
    assert clean_images.main(["--source", "asl", "--config", str(config)]) == 0
    report = {Path(r["path"]).name: r for r in read_csv(tmp_path / "interim/cleaning_report.csv")}
    assert report["a1.png"]["keep"] == "1"
    assert report["a2.png"]["exact_duplicate"] == "1"
    assert report["a2.png"]["duplicate_of"].endswith("a1.png")
    assert report["a3.png"]["near_duplicate"] == "1"
    assert report["a4.png"]["blurry"] == "1"
    assert report["a5.png"]["corrupt"] == "1"
    assert report["b2.png"]["too_dark"] == "1"
    assert report["b1.png"]["keep"] == "1"
    assert "x_test.png" not in report
    summary = {r["label"]: r for r in read_csv(tmp_path / "reports/cleaning_summary.csv")}
    assert [int(summary["A"][c]) for c in ("found", *clean_images.STEP_COLUMNS)] == [5, 4, 3, 2, 1]
    assert [int(summary["B"][c]) for c in ("found", *clean_images.STEP_COLUMNS)] == [2, 2, 2, 2, 1]


def test_psl_run_keeps_asl_rows_and_writes_table(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    make_psl_folder(tmp_path)
    config = write_config(tmp_path)
    assert clean_images.main(["--source", "asl", "--config", str(config)]) == 0
    assert clean_images.main(["--source", "psl", "--config", str(config)]) == 0
    sources = {row["source"] for row in read_csv(tmp_path / "reports/cleaning_summary.csv")}
    assert sources == {"asl", "psl"}
    psl_rows = [
        r for r in read_csv(tmp_path / "interim/cleaning_report.csv") if r["source"] == "psl"
    ]
    unlabelled = [r for r in psl_rows if r["no_label"] == "1"]
    assert len(psl_rows) == 3
    assert len(unlabelled) == 1
    assert unlabelled[0]["keep"] == "0"
    table = (tmp_path / "reports/cleaning_summary_table.tex").read_text(encoding="utf-8")
    assert "ASL baseline" in table
    assert "PSL" in table
    assert r"\label{tab:cleaning-counts}" in table


def test_rerunning_a_source_replaces_its_old_rows(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    config = write_config(tmp_path)
    for _ in range(2):
        assert clean_images.main(["--source", "asl", "--config", str(config)]) == 0
    assert len(read_csv(tmp_path / "interim/cleaning_report.csv")) == 7
    assert len(read_csv(tmp_path / "reports/cleaning_summary.csv")) == 2


def test_limit_per_class_reduces_the_report(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    config = write_config(tmp_path)
    args = ["--source", "asl", "--config", str(config), "--limit-per-class", "1"]
    assert clean_images.main(args) == 0
    assert len(read_csv(tmp_path / "interim/cleaning_report.csv")) == 2


def test_main_returns_error_for_missing_folder(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    assert clean_images.main(["--source", "psl", "--config", str(config)]) == 1


def test_summarize_labels_unlabelled_images() -> None:
    record = new_record("x.png", "psl")
    record["no_label"] = 1
    rows = summarize([record], "psl")
    assert rows[0]["label"] == "(no label)"
    assert rows[0]["after_unreadable_or_unlabelled"] == 0


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as info:
        clean_images.parse_args(["--help"])
    assert info.value.code == 0

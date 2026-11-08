"""Tests for src/signbridge/evaluation/bias.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import f1_score

from signbridge.evaluation import bias
from signbridge.evaluation.bias import (
    analyse,
    bootstrap_interval,
    detected_from_predictions,
    detected_from_table,
    macro_f1,
    quartile_groups,
    subgroup_row,
)
from signbridge.training.predictions import save_predictions

NAMES = ["A", "B", "C"]


def test_macro_f1_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, size=200)
    predicted = np.where(rng.random(200) < 0.7, truth, rng.integers(0, 3, size=200))
    expected = f1_score(truth, predicted, average="macro")
    assert macro_f1(truth, predicted, 3) == pytest.approx(expected)


def test_quartile_groups_split_eight_values_into_pairs() -> None:
    labels = quartile_groups(np.arange(8, dtype=float))
    assert [label[:2] for label in labels] == ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4"]


def test_bootstrap_is_reproducible_and_collapses_for_a_perfect_group() -> None:
    truth = np.array([0, 1, 2] * 20)
    first = bootstrap_interval(truth, truth, 3, 200, np.random.default_rng(1))
    again = bootstrap_interval(truth, truth, 3, 200, np.random.default_rng(1))
    assert first == again
    assert first["accuracy"] == (1.0, 1.0)
    assert first["macro_f1"] == (1.0, 1.0)


def test_bootstrap_interval_contains_the_estimate() -> None:
    truth = np.array([0, 1, 2] * 20)
    predicted = truth.copy()
    predicted[:12] = (predicted[:12] + 1) % 3
    low, high = bootstrap_interval(truth, predicted, 3, 500, np.random.default_rng(2))["accuracy"]
    assert low <= float((truth == predicted).mean()) <= high


def test_small_groups_are_flagged_and_large_ones_are_not() -> None:
    rng = np.random.default_rng(0)
    small = np.array([0, 1, 2] * 3)
    large = np.array([0, 1, 2] * 20)
    assert subgroup_row("m", "g", small, small, 3, 50, rng)["low_n"] is True
    assert subgroup_row("m", "g", large, large, 3, 50, rng)["low_n"] is False


def test_analyse_counts_and_accuracy_per_group() -> None:
    truth = np.array([0, 1, 2, 0, 1, 2])
    proba = np.full((6, 3), 0.1)
    proba[np.arange(6), [0, 1, 2, 1, 1, 2]] = 0.8  # the fourth image is wrong
    data = {"y_true": truth, "y_proba": proba, "class_names": np.array(NAMES)}
    rows = analyse(data, {"hand": ["yes", "yes", "yes", "no", "no", "no"]}, repeats=50)
    by_group = {row["group"]: row for row in rows}
    assert by_group["yes"]["n"] == 3 and by_group["yes"]["accuracy"] == 1.0
    assert by_group["no"]["n"] == 3 and by_group["no"]["accuracy"] == pytest.approx(2 / 3)


def test_detected_from_table_and_from_predictions(tmp_path: Path) -> None:
    table = tmp_path / "detections.csv"
    table.write_text("path,label,split,detected\nA/1.jpg,A,test,True\nA/2.jpg,A,test,False\n")
    paths = ["A/1.jpg", "A/2.jpg"]
    assert detected_from_table(paths, table) == ["hand detected", "no hand"]
    with pytest.raises(ValueError, match="not in"):
        detected_from_table(["A/9.jpg"], table)
    landmark = {"paths": np.array(["A/2.jpg"])}
    assert detected_from_predictions(paths, landmark) == ["no hand", "hand detected"]


def write_images_and_predictions(root: Path, count: int = 40) -> Path:
    """Create noise images of rising contrast and a prediction file for them."""
    rng = np.random.default_rng(0)
    paths = []
    for index in range(count):
        relative = f"{NAMES[index % 3]}/{index}.png"
        (root / "images" / NAMES[index % 3]).mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 40 + 5 * index, size=(24, 24)).astype(np.uint8) + 20
        Image.fromarray(pixels).save(root / "images" / relative)
        paths.append(relative)
    truth = np.arange(count) % 3
    proba = np.full((count, 3), 0.1)
    proba[np.arange(count), np.where(np.arange(count) % 5 == 0, (truth + 1) % 3, truth)] = 0.8
    return save_predictions(root / "test.npz", truth, proba, paths, NAMES)


def test_run_writes_a_table_and_a_figure(tmp_path: Path) -> None:
    predictions = write_images_and_predictions(tmp_path)
    landmark = tmp_path / "landmark.npz"
    save_predictions(landmark, np.array([0]), np.array([[0.8, 0.1, 0.1]]), ["A/0.png"], NAMES)
    rows = bias.run(
        predictions, tmp_path / "images", tmp_path / "out", "cnn_test", None, landmark, 50, 42
    )
    assert {row["measure"] for row in rows} == {"brightness", "blur", "hand"}
    assert sum(row["n"] for row in rows if row["measure"] == "hand") == 40
    with (tmp_path / "out" / "cnn_test_subgroups.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
        assert reader.fieldnames == bias.COLUMNS
    assert len(written) == len(rows)
    assert (tmp_path / "out" / "cnn_test_subgroups.png").stat().st_size > 0


def test_run_without_a_detection_source_skips_the_hand_check(tmp_path: Path) -> None:
    predictions = write_images_and_predictions(tmp_path)
    rows = bias.run(predictions, tmp_path / "images", tmp_path / "out", "x", None, None, 20, 1)
    assert {row["measure"] for row in rows} == {"brightness", "blur"}


def test_main_returns_one_for_a_missing_file(tmp_path: Path) -> None:
    code = bias.main(
        ["--predictions", str(tmp_path / "no.npz"), "--name", "x", "--out", str(tmp_path)]
    )
    assert code == 1


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exit_info:
        bias.main(["--help"])
    assert exit_info.value.code == 0

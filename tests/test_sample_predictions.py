"""Tests for src/signbridge/evaluation/sample_predictions.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from signbridge.evaluation import sample_predictions
from signbridge.evaluation.sample_predictions import example_rows, sample_indices, tile_title
from signbridge.training.predictions import save_predictions

NAMES = ["A", "B", "C"]
TRUTH = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
PREDICTED = [0, 1, 1, 1, 1, 0, 2, 2, 2, 2]  # rows 1, 2 and 5 are wrong
PATHS = [f"{NAMES[label]}/{NAMES[label]}{i}.png" for i, label in enumerate(TRUTH)]


def probabilities(predicted: list[int], confidence: float = 0.8) -> np.ndarray:
    """Probability rows that put the stated confidence on each predicted class."""
    proba = np.full((len(predicted), 3), (1 - confidence) / 2)
    for row, guess in enumerate(predicted):
        proba[row, guess] = confidence
    return proba


def write_files(folder: Path) -> tuple[Path, Path]:
    """Save a CNN file with all 10 images and an SVM file with only rows 1 to 7 (6 images)."""
    cnn_path = folder / "cnn.npz"
    svm_path = folder / "svm.npz"
    save_predictions(cnn_path, TRUTH, probabilities(PREDICTED), PATHS, NAMES)
    keep = [1, 2, 3, 4, 5, 6]
    save_predictions(
        svm_path,
        TRUTH[keep],
        probabilities([PREDICTED[i] for i in keep], 0.7),
        [PATHS[i] for i in keep],
        NAMES,
    )
    return cnn_path, svm_path


def write_images(root: Path) -> None:
    for relative in PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "white").save(target)


def test_sample_indices_keeps_only_right_or_only_wrong_rows() -> None:
    proba = probabilities(PREDICTED)
    assert sample_indices(TRUTH, proba, correct=False, count=10, seed=1).tolist() == [1, 2, 5]
    right = sample_indices(TRUTH, proba, correct=True, count=10, seed=1).tolist()
    assert right == [0, 3, 4, 6, 7, 8, 9]


def test_sample_indices_respects_the_count_and_is_repeatable() -> None:
    proba = probabilities(PREDICTED)
    first = sample_indices(TRUTH, proba, correct=True, count=3, seed=7)
    again = sample_indices(TRUTH, proba, correct=True, count=3, seed=7)
    assert len(first) == 3
    assert first.tolist() == again.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert set(first.tolist()) <= {0, 3, 4, 6, 7, 8, 9}


def test_sample_indices_is_empty_when_nothing_qualifies() -> None:
    perfect = probabilities([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    assert sample_indices(TRUTH, perfect, correct=False, count=4, seed=1).size == 0


def test_example_rows_report_true_predicted_and_confidence() -> None:
    data = {
        "y_true": TRUTH,
        "y_proba": probabilities(PREDICTED, 0.8),
        "paths": np.asarray(PATHS),
        "class_names": np.asarray(NAMES),
    }
    rows = example_rows("CNN", "wrong", data, np.array([1]))
    assert rows == [
        {
            "model": "CNN",
            "group": "wrong",
            "path": "A/A1.png",
            "true": "A",
            "predicted": "B",
            "confidence": 0.8,
        }
    ]


def test_tile_title_shows_both_letters_and_the_probability() -> None:
    row = {"true": "A", "predicted": "B", "confidence": 0.8}
    assert tile_title(row) == "true A, predicted B\np = 0.80"


def test_run_writes_figure_and_a_csv_of_the_images_shown(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    write_images(tmp_path / "images")
    out = tmp_path / "figs" / "samples.png"
    summary = sample_predictions.run(cnn_path, svm_path, out, tmp_path / "images", per_group=2)
    assert out.stat().st_size > 0
    assert summary["common_images"] == 6
    assert summary["shown"] == {
        "CNN: correct": 2,
        "CNN: wrong": 2,
        "SVM: correct": 2,
        "SVM: wrong": 2,
    }
    with out.with_suffix(".csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert reader_columns(out.with_suffix(".csv")) == sample_predictions.CSV_COLUMNS
    assert len(rows) == 8


def reader_columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def test_run_only_shows_images_both_models_predicted(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    write_images(tmp_path / "images")
    out = tmp_path / "samples.png"
    sample_predictions.run(cnn_path, svm_path, out, tmp_path / "images", per_group=6)
    with out.with_suffix(".csv").open(encoding="utf-8") as handle:
        shown = {row["path"] for row in csv.DictReader(handle)}
    assert shown <= {PATHS[i] for i in [1, 2, 3, 4, 5, 6]}


def test_run_labels_the_groups_correctly(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    write_images(tmp_path / "images")
    out = tmp_path / "samples.png"
    sample_predictions.run(cnn_path, svm_path, out, tmp_path / "images", per_group=6)
    with out.with_suffix(".csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert (row["true"] == row["predicted"]) == (row["group"] == "correct")


def test_run_is_repeatable_for_the_same_seed(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    write_images(tmp_path / "images")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    sample_predictions.run(cnn_path, svm_path, first, tmp_path / "images", per_group=2, seed=5)
    sample_predictions.run(cnn_path, svm_path, second, tmp_path / "images", per_group=2, seed=5)
    assert first.with_suffix(".csv").read_text() == second.with_suffix(".csv").read_text()


def test_run_with_missing_images_still_writes_the_figure(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    out = tmp_path / "samples.png"
    sample_predictions.run(cnn_path, svm_path, out, tmp_path / "no_images", per_group=2)
    assert out.stat().st_size > 0


def test_run_rejects_a_group_size_below_one(tmp_path: Path) -> None:
    cnn_path, svm_path = write_files(tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        sample_predictions.run(cnn_path, svm_path, tmp_path / "x.png", tmp_path, per_group=0)


def test_run_rejects_files_with_different_class_names(tmp_path: Path) -> None:
    cnn_path, _ = write_files(tmp_path)
    other = tmp_path / "other.npz"
    save_predictions(other, TRUTH, probabilities(PREDICTED), PATHS, ["A", "B", "Z"])
    with pytest.raises(ValueError, match="class names"):
        sample_predictions.run(cnn_path, other, tmp_path / "x.png", tmp_path, per_group=2)


def test_main_returns_one_for_a_missing_prediction_file(tmp_path: Path) -> None:
    code = sample_predictions.main(
        ["--cnn", str(tmp_path / "missing.npz"), "--svm", str(tmp_path / "missing2.npz")]
    )
    assert code == 1


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exit_info:
        sample_predictions.main(["--help"])
    assert exit_info.value.code == 0

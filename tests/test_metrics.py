"""Tests for src/signbridge/evaluation/metrics.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from signbridge.evaluation.metrics import evaluate, save_report

NAMES = ["A", "B", "C"]


def probabilities(predicted: list[int], confidence: float = 0.8) -> np.ndarray:
    """Build a probability row per image that puts `confidence` on the predicted class."""
    rest = (1.0 - confidence) / (len(NAMES) - 1)
    proba = np.full((len(predicted), len(NAMES)), rest)
    proba[np.arange(len(predicted)), predicted] = confidence
    return proba


def test_perfect_predictions_score_one_everywhere() -> None:
    truth = np.array([0, 1, 2, 0, 1, 2])
    result = evaluate(truth, probabilities(truth.tolist()), NAMES)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["macro_auc_roc"] == 1.0
    assert np.trace(np.array(result["confusion_matrix"])) == 6


def test_small_case_matches_hand_calculation() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    result = evaluate(truth, probabilities([0, 1, 1, 1, 2, 0]), NAMES)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["per_class"]["A"]["precision"] == pytest.approx(0.5)
    assert result["per_class"]["B"]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"]["B"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["C"]["f1"] == pytest.approx(2 / 3)
    assert result["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert 0.0 <= result["macro_auc_roc"] <= 1.0


def test_float32_probabilities_are_accepted() -> None:
    truth = np.array([0, 1, 2])
    proba = probabilities([0, 1, 2]).astype(np.float32)
    assert evaluate(truth, proba, NAMES)["accuracy"] == 1.0


def test_class_missing_from_y_true_is_rejected() -> None:
    with pytest.raises(ValueError, match="C"):
        evaluate(np.array([0, 1, 0, 1]), probabilities([0, 1, 0, 1]), NAMES)


@pytest.mark.parametrize(
    "truth, proba",
    [
        (np.array([0, 1, 2]), np.ones((2, 3)) / 3),
        (np.array([0, 1, 5]), np.ones((3, 3)) / 3),
        (np.array([0, 1, 2]), np.ones((3, 2)) / 2),
        (np.array([0, 1, 2]), np.full((3, 3), 0.5)),
        (np.array([0, 1, 2]), np.full((3, 3), np.nan)),
        (np.array([], dtype=int), np.zeros((0, 3))),
    ],
)
def test_bad_inputs_are_rejected(truth: np.ndarray, proba: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate(truth, proba, NAMES)


def test_save_report_writes_json_csv_and_figure(tmp_path: Path) -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    result = evaluate(truth, probabilities([0, 1, 1, 1, 2, 0]), NAMES)
    paths = save_report(result, NAMES, tmp_path / "out", "cnn_val", "CNN, validation")
    assert [path.name for path in paths] == [
        "cnn_val_metrics.json",
        "cnn_val_per_class.csv",
        "cnn_val_confusion.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8"))["n_samples"] == 6
    with paths[1].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == NAMES

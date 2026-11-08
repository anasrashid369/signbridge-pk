"""Tests for src/signbridge/evaluation/error_analysis.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from signbridge.evaluation import error_analysis
from signbridge.evaluation.error_analysis import (
    confused_pairs,
    per_class_table,
    wrong_predictions,
)
from signbridge.evaluation.metrics import evaluate
from signbridge.training.predictions import save_predictions

NAMES = ["A", "B", "C"]
TRUTH = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
PREDICTED = [0, 1, 1, 1, 1, 0, 2, 2, 2, 2]  # A->B twice, B->A once, C always right
CONFIDENCE = [0.9, 0.9, 0.6, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.9]


def probabilities() -> np.ndarray:
    """Probability rows that put the stated confidence on the predicted class."""
    proba = np.zeros((len(PREDICTED), 3))
    for row, (guess, confidence) in enumerate(zip(PREDICTED, CONFIDENCE, strict=True)):
        proba[row] = (1 - confidence) / 2
        proba[row, guess] = confidence
    return proba


def write_prediction_file(path: Path) -> list[str]:
    """Save the synthetic prediction file and return its image paths."""
    paths = [f"{NAMES[label]}/{NAMES[label]}{i}.png" for i, label in enumerate(TRUTH)]
    save_predictions(path, TRUTH, probabilities(), paths, NAMES)
    return paths


def test_confused_pairs_lists_known_mistakes_largest_first() -> None:
    result = evaluate(TRUTH, probabilities(), NAMES)
    pairs = confused_pairs(result["confusion_matrix"], NAMES)
    assert pairs == [
        {"true": "A", "predicted": "B", "count": 2},
        {"true": "B", "predicted": "A", "count": 1},
    ]


def test_confused_pairs_respects_top_n() -> None:
    result = evaluate(TRUTH, probabilities(), NAMES)
    assert len(confused_pairs(result["confusion_matrix"], NAMES, top_n=1)) == 1


def test_per_class_table_puts_the_least_reliable_letter_first() -> None:
    rows = per_class_table(evaluate(TRUTH, probabilities(), NAMES)["per_class"])
    assert [row["label"] for row in rows] == ["A", "B", "C"]
    assert rows[0]["f1"] <= rows[1]["f1"] <= rows[2]["f1"]


def test_wrong_predictions_most_confident_first_and_limited() -> None:
    order = wrong_predictions(TRUTH, probabilities())
    assert order.tolist() == [1, 5, 2]  # wrong-class confidence 0.9, 0.7, 0.6
    assert wrong_predictions(TRUTH, probabilities(), max_images=2).tolist() == [1, 5]


def test_run_writes_tables_and_grid(tmp_path: Path) -> None:
    paths = write_prediction_file(tmp_path / "pred.npz")
    for relative in paths:
        target = tmp_path / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "white").save(target)
    summary = error_analysis.run(tmp_path / "pred.npz", tmp_path / "out", tmp_path / "images")
    assert summary["mistakes"] == 3
    assert summary["files"] == ["confused_pairs.csv", "per_class.csv", "misclassified.png"]
    assert all((tmp_path / "out" / name).stat().st_size > 0 for name in summary["files"])
    with (tmp_path / "out" / "confused_pairs.csv").open(encoding="utf-8") as handle:
        assert next(csv.DictReader(handle)) == {"true": "A", "predicted": "B", "count": "2"}


def test_run_without_image_root_skips_the_grid(tmp_path: Path) -> None:
    write_prediction_file(tmp_path / "pred.npz")
    summary = error_analysis.run(tmp_path / "pred.npz", tmp_path / "out", None)
    assert summary["files"] == ["confused_pairs.csv", "per_class.csv"]


def test_run_with_a_missing_image_still_writes_the_grid(tmp_path: Path) -> None:
    write_prediction_file(tmp_path / "pred.npz")
    summary = error_analysis.run(tmp_path / "pred.npz", tmp_path / "out", tmp_path / "none")
    assert "misclassified.png" in summary["files"]


def test_main_returns_one_for_a_missing_prediction_file(tmp_path: Path) -> None:
    code = error_analysis.main(
        ["--predictions", str(tmp_path / "missing.npz"), "--out", str(tmp_path / "out")]
    )
    assert code == 1


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exit_info:
        error_analysis.main(["--help"])
    assert exit_info.value.code == 0

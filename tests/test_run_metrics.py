"""Tests for src/signbridge/evaluation/run_metrics.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signbridge.evaluation import run_metrics
from signbridge.training.predictions import save_predictions

NAMES = ["A", "B", "C"]


def write_file(path: Path, image_names: list[str], truth: list[int], guess: list[int]) -> None:
    """Save a prediction file that puts 0.8 on the guessed class."""
    proba = np.full((len(truth), 3), 0.1)
    proba[np.arange(len(truth)), guess] = 0.8
    save_predictions(path, np.array(truth), proba, image_names, NAMES)


def test_run_writes_the_three_files_and_scores_all_images(tmp_path: Path) -> None:
    names = [f"{NAMES[t]}/{i}.jpg" for i, t in enumerate([0, 1, 2, 0, 1, 2])]
    write_file(tmp_path / "cnn.npz", names, [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 2])
    metrics = run_metrics.run(tmp_path / "cnn.npz", tmp_path / "out", "cnn_test", None)
    assert metrics["n_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "cnn_test_confusion.png",
        "cnn_test_metrics.json",
        "cnn_test_per_class.csv",
    ]


def test_restrict_to_scores_only_the_common_images(tmp_path: Path) -> None:
    names = [f"{NAMES[t]}/{i}.jpg" for i, t in enumerate([0, 1, 2, 0, 1, 2])]
    write_file(tmp_path / "cnn.npz", names, [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 2])
    write_file(tmp_path / "landmark.npz", names[:5], [0, 1, 2, 0, 1], [0, 1, 2, 0, 1])
    metrics = run_metrics.run(
        tmp_path / "cnn.npz", tmp_path / "out", "cnn_common", tmp_path / "landmark.npz"
    )
    assert metrics["n_samples"] == 5
    assert metrics["accuracy"] == pytest.approx(4 / 5)


def test_main_returns_one_for_a_missing_file(tmp_path: Path) -> None:
    code = run_metrics.main(
        ["--predictions", str(tmp_path / "x.npz"), "--out", str(tmp_path), "--name", "x"]
    )
    assert code == 1


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exit_info:
        run_metrics.main(["--help"])
    assert exit_info.value.code == 0

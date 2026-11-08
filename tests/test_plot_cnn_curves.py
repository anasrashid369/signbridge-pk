"""Tests for tools/plot_cnn_curves.py on a synthetic history file."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.plot_cnn_curves import (
    ACCURACY_FILE,
    HISTORY_COLUMNS,
    LOSS_FILE,
    best_epoch,
    load_history,
    main,
    plot_curves,
)


def write_history(path: Path) -> Path:
    """Write three synthetic epochs with a clear best validation score."""
    rows = [
        {
            "epoch": 1,
            "train_loss": 1.2,
            "val_loss": 1.1,
            "train_acc": 0.4,
            "val_acc": 0.45,
            "val_macro_f1": 0.40,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "train_acc": 0.65,
            "val_acc": 0.70,
            "val_macro_f1": 0.68,
        },
        {
            "epoch": 3,
            "train_loss": 0.6,
            "val_loss": 0.75,
            "train_acc": 0.75,
            "val_acc": 0.69,
            "val_macro_f1": 0.66,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_history_and_best_epoch(tmp_path: Path) -> None:
    history = load_history(write_history(tmp_path / "history.csv"))
    assert history["epoch"] == [1.0, 2.0, 3.0]
    assert best_epoch(history) == 2


def test_plot_curves_writes_two_figures(tmp_path: Path) -> None:
    history = load_history(write_history(tmp_path / "history.csv"))
    loss_path, accuracy_path = plot_curves(history, tmp_path / "figures")
    assert loss_path.name == LOSS_FILE
    assert accuracy_path.name == ACCURACY_FILE
    assert loss_path.stat().st_size > 1000
    assert accuracy_path.stat().st_size > 1000


def test_main_returns_zero_on_synthetic_history(tmp_path: Path) -> None:
    history_path = write_history(tmp_path / "history.csv")
    assert (
        main(
            [
                "--history",
                str(history_path),
                "--figures-dir",
                str(tmp_path / "figures"),
            ]
        )
        == 0
    )


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    path.write_text("epoch,train_loss\n1,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_history(path)

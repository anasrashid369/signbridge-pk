"""Tests for the landmark MLP curve plot."""

from pathlib import Path

import pytest

from signbridge.plot_landmark_curves import best_epoch, load_history, plot_curves

HEADER = "epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1"
ROWS = [
    "1,2.0,0.20,2.1,0.25,0.20",
    "2,1.0,0.60,1.1,0.55,0.50",
    "3,0.8,0.70,1.2,0.52,0.45",
]


def write_history(path: Path, rows: list[str] = ROWS, header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_load_history_reads_every_column(tmp_path: Path) -> None:
    history = load_history(write_history(tmp_path / "h.csv"))
    assert history["epoch"] == [1.0, 2.0, 3.0]
    assert history["val_loss"][2] == pytest.approx(1.2)


def test_best_epoch_uses_validation_macro_f1(tmp_path: Path) -> None:
    history = load_history(write_history(tmp_path / "h.csv"))
    assert best_epoch(history) == 2


def test_load_history_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path / "nope.csv")


def test_load_history_missing_column(tmp_path: Path) -> None:
    path = write_history(tmp_path / "h.csv", ["1,2.0,0.2"], header="epoch,train_loss,train_acc")
    with pytest.raises(ValueError, match="missing columns"):
        load_history(path)


def test_load_history_without_epochs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no epochs"):
        load_history(write_history(tmp_path / "h.csv", rows=[]))


def test_plot_curves_writes_a_png(tmp_path: Path) -> None:
    history = load_history(write_history(tmp_path / "h.csv"))
    saved = plot_curves(history, tmp_path / "out" / "curves.png")
    assert saved.read_bytes().startswith(b"\x89PNG")

"""Plot the CNN loss and accuracy curves from the training history CSV.

Run from the repository root after ``train_cnn`` wrote the history file:
    python -m tools.plot_cnn_curves

The dashed line marks the epoch with the best validation macro-F1, which is the
checkpoint that training keeps. The figures are written with the Agg backend so
the command also works on a headless machine.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signbridge.config import REPO_ROOT

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1")
LOSS_FILE = "cnn_loss_curve.png"
ACCURACY_FILE = "cnn_accuracy_curve.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the CNN loss and accuracy curves.")
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "reports/training/cnn_history.csv",
        help="Per-epoch history CSV written by train_cnn.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "reports/figures",
        help="Folder for the two PNG files.",
    )
    return parser.parse_args(argv)


def load_history(path: Path) -> dict[str, list[float]]:
    """Read the history CSV into one list of numbers per column.

    Raises:
        FileNotFoundError: If the history file does not exist.
        ValueError: If a column is missing or the file has no epochs.
    """
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in HISTORY_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"History file is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"History file has no epochs: {path}")
    return {column: [float(row[column]) for row in rows] for column in HISTORY_COLUMNS}


def best_epoch(history: dict[str, list[float]]) -> int:
    """Return the epoch with the highest validation macro-F1."""
    scores = history["val_macro_f1"]
    index = max(range(len(scores)), key=scores.__getitem__)
    return int(history["epoch"][index])


def _plot(
    epochs: list[float],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    destination: Path,
    best: int,
) -> None:
    """Draw one figure with one line per series and mark the best epoch."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, values in series.items():
        axis.plot(epochs, values, marker="o", label=label)
    axis.axvline(best, color="grey", linestyle="--", label=f"best epoch {best}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_curves(history: dict[str, list[float]], figures_dir: Path) -> tuple[Path, Path]:
    """Write the loss and accuracy figures and return their paths."""
    epochs = history["epoch"]
    best = best_epoch(history)
    loss_path = figures_dir / LOSS_FILE
    accuracy_path = figures_dir / ACCURACY_FILE
    _plot(
        epochs,
        {"Training loss": history["train_loss"], "Validation loss": history["val_loss"]},
        "Loss",
        "CNN loss per epoch",
        loss_path,
        best,
    )
    _plot(
        epochs,
        {"Training accuracy": history["train_acc"], "Validation accuracy": history["val_acc"]},
        "Accuracy",
        "CNN accuracy per epoch",
        accuracy_path,
        best,
    )
    return loss_path, accuracy_path


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a file or CSV error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        history = load_history(args.history)
        loss_path, accuracy_path = plot_curves(history, args.figures_dir)
    except (FileNotFoundError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Wrote %s and %s", loss_path, accuracy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

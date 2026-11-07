"""Plot the landmark MLP loss and accuracy curves from the history CSV.

Run from the repository root, after train_landmark.py wrote the history file:
    python -m signbridge.plot_landmark_curves

The dashed line marks the epoch with the best validation macro-F1, which is the epoch
whose weights the training command keeps.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

LANDMARK_CONFIG_PATH = REPO_ROOT / "configs" / "landmark.yaml"
FIGURE_FILE = "landmark_mlp_curves.png"  # saved inside paths.figures_dir
COLUMNS = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1")


def load_history(path: Path) -> dict[str, list[float]]:
    """Read the per-epoch history CSV into one list of numbers per column.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a column is missing or the file has no epochs.
    """
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"History file is missing columns {missing}: {path}")
        history: dict[str, list[float]] = {name: [] for name in COLUMNS}
        for row in reader:
            for name in COLUMNS:
                history[name].append(float(row[name]))
    if not history["epoch"]:
        raise ValueError(f"History file has no epochs: {path}")
    return history


def best_epoch(history: dict[str, list[float]]) -> int:
    """Return the epoch with the highest validation macro-F1 (the first one if tied)."""
    scores = history["val_macro_f1"]
    return int(history["epoch"][scores.index(max(scores))])


def plot_curves(
    history: dict[str, list[float]], destination: Path, title: str | None = None
) -> Path:
    """Draw loss and accuracy (train and validation) side by side and save a PNG."""
    import matplotlib  # imported here so the helpers above work without a display

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    best = best_epoch(history)
    if title is None:
        title = f"Landmark MLP, ASL baseline: {len(epochs)} epochs, best epoch {best}"
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))
    panels = (
        (loss_axis, "train_loss", "val_loss", "Loss (cross-entropy)"),
        (accuracy_axis, "train_acc", "val_acc", "Accuracy"),
    )
    for axis, train_key, val_key, label in panels:
        axis.plot(epochs, history[train_key], label="train")
        axis.plot(epochs, history[val_key], label="validation")
        axis.axvline(best, linestyle="--", color="grey", label=f"best epoch ({best})")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the landmark MLP curves.")
    parser.add_argument("--config", type=Path, default=LANDMARK_CONFIG_PATH)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--history", type=Path, help="History CSV. Default: output.history_csv in landmark.yaml."
    )
    parser.add_argument(
        "--out", type=Path, help="Figure file. Default: paths.figures_dir/landmark_mlp_curves.png."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    """Load the history and save the figure."""
    history_path = args.history
    if history_path is None:
        history_path = REPO_ROOT / load_config(args.config)["output"]["history_csv"]
    destination = args.out
    if destination is None:
        destination = resolve_path(load_config(args.data_config), "figures_dir") / FIGURE_FILE
    history = load_history(history_path)
    saved = plot_curves(history, destination)
    logger.info(
        "Plotted %d epochs (best epoch %d) to %s", len(history["epoch"]), best_epoch(history), saved
    )
    return saved


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or file error."""
    args = parse_args(argv)
    setup_logging()
    try:
        run(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

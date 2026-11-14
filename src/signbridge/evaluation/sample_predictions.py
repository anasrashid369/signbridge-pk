"""Figure of sample predictions: CNN and SVM, correct and wrong, from real prediction files.

The figure shows test images with the true letter, the predicted letter and the probability the
model gave to its prediction. Both models are shown on the images present in both files (the
images where a hand was detected), so they are judged on the same pool. The images are drawn at
random with a fixed seed, not chosen by hand, so the figure is a fair sample and not a best-case
or worst-case selection. A CSV next to the figure lists every image shown.

Run from the repository root:
    python -m signbridge.evaluation.sample_predictions \
        --cnn reports/predictions/cnn_test.npz \
        --svm reports/predictions/landmark_svm_test.npz \
        --out reports/figures/sample_predictions.png
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from signbridge.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from signbridge.evaluation.common_set import restrict_to_common  # noqa: E402
from signbridge.evaluation.error_analysis import default_image_root  # noqa: E402
from signbridge.logging_setup import setup_logging  # noqa: E402
from signbridge.training.predictions import load_predictions  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PER_GROUP = 6
DEFAULT_SEED = 42
DEFAULT_OUT = Path("reports/figures/sample_predictions.png")
CSV_COLUMNS = ["model", "group", "path", "true", "predicted", "confidence"]
RIGHT_COLOUR = "#1b7837"
WRONG_COLOUR = "#b2182b"


def sample_indices(
    y_true: np.ndarray, y_proba: np.ndarray, correct: bool, count: int, seed: int
) -> np.ndarray:
    """Pick up to ``count`` rows the model got right (or wrong), at random, in row order.

    The draw depends only on the seed, so the same files give the same figure on every run. If
    fewer than ``count`` rows qualify, all of them are returned.
    """
    predicted = y_proba.argmax(axis=1)
    pool = np.flatnonzero((predicted == y_true) == correct)
    if len(pool) <= count:
        return pool
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=count, replace=False))


def example_rows(
    model: str, group: str, data: dict[str, np.ndarray], indices: np.ndarray
) -> list[dict[str, Any]]:
    """Describe the chosen images: path, true letter, predicted letter and confidence."""
    names = [str(name) for name in data["class_names"]]
    predicted = data["y_proba"].argmax(axis=1)
    return [
        {
            "model": model,
            "group": group,
            "path": str(data["paths"][index]),
            "true": names[int(data["y_true"][index])],
            "predicted": names[int(predicted[index])],
            "confidence": round(float(data["y_proba"][index, predicted[index]]), 4),
        }
        for index in indices
    ]


def tile_title(row: dict[str, Any]) -> str:
    """Return the two-line title of one image tile."""
    return f"true {row['true']}, predicted {row['predicted']}\np = {row['confidence']:.2f}"


def plot_samples(
    panels: list[tuple[str, list[dict[str, Any]]]], image_root: Path, path: Path, columns: int
) -> None:
    """Save a grid with one row per panel and one tile per example.

    Args:
        panels: Pairs of (row label, example rows), for example ("CNN: correct", rows).
        image_root: Folder that the image paths in the rows are relative to.
        path: Destination image file.
        columns: Number of tiles per row.
    """
    figure, axes = plt.subplots(
        len(panels), columns, figsize=(2.4 * columns + 1.4, 2.9 * len(panels)), squeeze=False
    )
    for axis in axes.ravel():
        axis.axis("off")
    for row_number, (label, rows) in enumerate(panels):
        axes[row_number, 0].text(
            -0.12, 0.5, label, transform=axes[row_number, 0].transAxes,
            rotation=90, ha="right", va="center", fontsize=11, fontweight="bold",
        )  # fmt: skip
        for column, example in enumerate(rows[:columns]):
            axis = axes[row_number, column]
            image_path = image_root / example["path"]
            try:
                axis.imshow(Image.open(image_path).convert("RGB"))
            except OSError:
                logger.warning("Could not read %s", image_path)
                axis.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=8)
            colour = RIGHT_COLOUR if example["true"] == example["predicted"] else WRONG_COLOUR
            axis.set_title(tile_title(example), fontsize=8, color=colour)
    figure.suptitle("Sample test predictions (ASL baseline, random draw, fixed seed)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the list of images shown in the figure to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    cnn_path: Path,
    svm_path: Path,
    out_path: Path,
    image_root: Path,
    per_group: int = DEFAULT_PER_GROUP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Write the sample-predictions figure and the CSV that lists its images.

    Returns:
        A summary with the number of common images, the figure and CSV paths, and how many
        correct and wrong examples each model has in the figure.

    Raises:
        FileNotFoundError: If a prediction file is missing.
        ValueError: If ``per_group`` is below 1, or the files cannot be compared (see
            ``common_set.restrict_to_common``).
    """
    if per_group < 1:
        raise ValueError("per_group must be at least 1")
    cnn, svm = restrict_to_common(load_predictions(cnn_path), load_predictions(svm_path))
    panels: list[tuple[str, list[dict[str, Any]]]] = []
    for model, data in (("CNN", cnn), ("SVM", svm)):
        for group, correct in (("correct", True), ("wrong", False)):
            chosen = sample_indices(data["y_true"], data["y_proba"], correct, per_group, seed)
            panels.append((f"{model}: {group}", example_rows(model, group, data, chosen)))
    plot_samples(panels, image_root, out_path, per_group)
    rows = [row for _, panel_rows in panels for row in panel_rows]
    csv_path = out_path.with_suffix(".csv")
    write_rows(csv_path, rows)
    return {
        "common_images": len(cnn["y_true"]),
        "figure": str(out_path),
        "csv": str(csv_path),
        "shown": {label: len(panel_rows) for label, panel_rows in panels},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw sample CNN and SVM predictions.")
    parser.add_argument("--cnn", type=Path, required=True, help="CNN test .npz file.")
    parser.add_argument("--svm", type=Path, required=True, help="Landmark SVM test .npz file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Figure file to write.")
    parser.add_argument("--image-root", type=Path, help="ASL train folder (default: from config).")
    parser.add_argument("--per-group", type=int, default=DEFAULT_PER_GROUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    setup_logging()
    try:
        config = load_config(args.config)
        image_root = args.image_root or default_image_root(config)
        summary = run(args.cnn, args.svm, args.out, image_root, args.per_group, args.seed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "%d common images; wrote %s and %s (shown: %s)",
        summary["common_images"],
        summary["figure"],
        summary["csv"],
        summary["shown"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

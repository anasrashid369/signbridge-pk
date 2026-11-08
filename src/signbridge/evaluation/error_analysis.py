"""Error analysis of one prediction file: confused letter pairs, weakest letters, bad images.

Run from the repository root:
    python -m signbridge.evaluation.error_analysis \
        --predictions reports/predictions/cnn_test.npz --out reports/errors/cnn_test
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

from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from signbridge.evaluation.metrics import evaluate  # noqa: E402
from signbridge.logging_setup import setup_logging  # noqa: E402
from signbridge.training.predictions import load_predictions  # noqa: E402

logger = logging.getLogger(__name__)

TOP_PAIRS = 10
MAX_GRID_IMAGES = 12
GRID_COLUMNS = 4
PAIR_COLUMNS = ["true", "predicted", "count"]
CLASS_COLUMNS = ["label", "precision", "recall", "f1", "support"]


def confused_pairs(
    matrix: list[list[int]], class_names: list[str], top_n: int = TOP_PAIRS
) -> list[dict[str, Any]]:
    """Return the most frequent (true letter, predicted letter) mistakes, largest first.

    Ties are broken by letter order so the result is the same on every run.
    """
    counts = np.array(matrix)
    np.fill_diagonal(counts, 0)
    pairs = [(int(counts[i, j]), i, j) for i, j in zip(*np.nonzero(counts), strict=True)]
    pairs.sort(key=lambda pair: (-pair[0], class_names[pair[1]], class_names[pair[2]]))
    return [
        {"true": class_names[i], "predicted": class_names[j], "count": count}
        for count, i, j in pairs[:top_n]
    ]


def per_class_table(per_class: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Return one row per letter, the least reliable letter (lowest F1) first."""
    rows = [{"label": label, **values} for label, values in per_class.items()]
    return sorted(rows, key=lambda row: (row["f1"], row["label"]))


def wrong_predictions(
    y_true: np.ndarray, y_proba: np.ndarray, max_images: int = MAX_GRID_IMAGES
) -> np.ndarray:
    """Return indices of wrong predictions, the most confidently wrong first."""
    predicted = y_proba.argmax(axis=1)
    wrong = np.flatnonzero(predicted != y_true)
    confidence = y_proba[wrong, predicted[wrong]]
    return wrong[np.argsort(-confidence, kind="stable")][:max_images]


def plot_wrong_grid(
    indices: np.ndarray, data: dict[str, np.ndarray], image_root: Path, path: Path
) -> None:
    """Save a grid of misclassified images titled with the true and predicted letter."""
    names = data["class_names"]
    predicted = data["y_proba"].argmax(axis=1)
    rows = max(1, -(-len(indices) // GRID_COLUMNS))
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(3 * GRID_COLUMNS, 3.2 * rows))
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, index in zip(np.atleast_1d(axes).ravel(), indices, strict=False):
        image_path = image_root / str(data["paths"][index])
        try:
            axis.imshow(Image.open(image_path).convert("RGB"))
        except OSError:
            logger.warning("Could not read %s", image_path)
        confidence = data["y_proba"][index, predicted[index]]
        axis.set_title(
            f"true {names[data['y_true'][index]]}, predicted {names[predicted[index]]}\n"
            f"p = {confidence:.2f}",
            fontsize=9,
        )
    figure.suptitle("Most confidently wrong predictions")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV file with the given columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(predictions: Path, out_dir: Path, image_root: Path | None) -> dict[str, Any]:
    """Write confused_pairs.csv, per_class.csv and misclassified.png into out_dir.

    Returns:
        A summary with the number of images, the number of mistakes and the files written.

    Raises:
        FileNotFoundError: If the prediction file is missing.
        ValueError: If the predictions are invalid (see ``metrics.evaluate``).
    """
    data = load_predictions(predictions)
    names = [str(name) for name in data["class_names"]]
    metrics = evaluate(data["y_true"], data["y_proba"], names)
    write_csv(
        out_dir / "confused_pairs.csv",
        PAIR_COLUMNS,
        confused_pairs(metrics["confusion_matrix"], names),
    )
    write_csv(out_dir / "per_class.csv", CLASS_COLUMNS, per_class_table(metrics["per_class"]))
    written = ["confused_pairs.csv", "per_class.csv"]
    indices = wrong_predictions(data["y_true"], data["y_proba"])
    if image_root is not None and len(indices):
        plot_wrong_grid(indices, data, image_root, out_dir / "misclassified.png")
        written.append("misclassified.png")
    else:
        logger.warning("No image grid written (no image root, or no mistakes)")
    mistakes = int((data["y_proba"].argmax(axis=1) != data["y_true"]).sum())
    return {"images": metrics["n_samples"], "mistakes": mistakes, "files": written}


def default_image_root(config: dict[str, Any]) -> Path:
    """Return the ASL training folder named in the data config."""
    folder = config["dataset"]["asl_train_folder"]
    return resolve_path(config, "asl_raw_dir") / folder / folder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyse the mistakes in a prediction file.")
    parser.add_argument("--predictions", type=Path, required=True, help="A .npz prediction file.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for the output files.")
    parser.add_argument("--image-root", type=Path, help="ASL train folder (default: from config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    setup_logging()
    try:
        config = load_config(args.config)
        image_root = args.image_root or default_image_root(config)
        summary = run(args.predictions, args.out, image_root)
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "%d mistakes in %d images; wrote %s to %s",
        summary["mistakes"],
        summary["images"],
        ", ".join(summary["files"]),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

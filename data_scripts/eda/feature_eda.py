"""Feature EDA charts: detection rate, correlation heatmap, before/after normalization.

Run from the repository root, after data_scripts.features.extract has created the cache:
    python -m data_scripts.eda.feature_eda --help
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data_scripts.features.extract import DETECTIONS_FILE, FEATURES_FILE
from data_scripts.features.labels import load_label_map
from data_scripts.features.landmarks import NUM_COORDS, NUM_LANDMARKS
from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from signbridge.logging_setup import setup_logging

matplotlib.use("Agg")  # draw to files only, no window

logger = logging.getLogger(__name__)

MIN_STD = 1e-9  # features with a smaller spread are constant and have no correlation
AXIS_NAMES = "xyz"
TABLE_FILE = "detection_rate_per_class.csv"  # numbers behind the detection rate chart


def load_detections(path: Path) -> list[tuple[str, bool]]:
    """Read landmark_detections.csv and return (label, detected) for every image."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["label"], row["detected"] == "True") for row in csv.DictReader(handle)]


def detection_rates(records: list[tuple[str, bool]]) -> dict[str, tuple[int, int]]:
    """Return {label: (images with a hand found, images tested)}."""
    rates: dict[str, tuple[int, int]] = {}
    for label, detected in records:
        found, total = rates.get(label, (0, 0))
        rates[label] = (found + int(detected), total + 1)
    return rates


def correlation_matrix(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correlate the feature columns, leaving out constant columns.

    Returns:
        The correlation matrix and the indices of the columns that were kept.

    Raises:
        ValueError: If there are fewer than 2 samples or fewer than 2 varying columns.
    """
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute correlations.")
    kept = np.where(features.std(axis=0) > MIN_STD)[0]
    if len(kept) < 2:
        raise ValueError("Need at least 2 non-constant features.")
    return np.corrcoef(features[:, kept], rowvar=False), kept


def feature_name(index: int) -> str:
    """Return a short name such as '8y' (landmark 8, y coordinate) for a feature column."""
    return f"{index // NUM_COORDS}{AXIS_NAMES[index % NUM_COORDS]}"


def save_detection_table(
    rates: dict[str, tuple[int, int]], label_order: list[str], destination: Path
) -> Path:
    """Save the numbers behind the detection rate chart, one row per class."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "images", "hand_found", "detection_rate_percent"])
        for label in label_order:
            if label in rates:
                found, total = rates[label]
                writer.writerow([label, total, found, round(100 * found / total, 1)])
    return destination


def plot_detection_rate(
    rates: dict[str, tuple[int, int]],
    label_order: list[str],
    destination: Path,
    dataset_label: str,
) -> None:
    """Bar chart of the percentage of images where MediaPipe found a hand, per class."""
    labels = [label for label in label_order if label in rates]
    percent = [100 * rates[label][0] / rates[label][1] for label in labels]
    found = sum(rates[label][0] for label in labels)
    total = sum(rates[label][1] for label in labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, percent)
    for position, label in enumerate(labels):
        ax.text(position, percent[position] + 1, str(rates[label][1]), ha="center", fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Class")
    ax.set_ylabel("Hand detected (% of images)")
    ax.set_title(
        f"{dataset_label}: hand detection rate per class "
        f"({found} of {total} sample images had a hand detected)\n"
        "number above each bar = images tested in that class"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_correlation(
    features: np.ndarray, destination: Path, dataset_label: str, images_tested: int
) -> None:
    """Heatmap of correlations between the normalized landmark features."""
    matrix, kept = correlation_matrix(features)
    names = [feature_name(int(index)) for index in kept]
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(names)), names, fontsize=5)
    ax.set_title(
        f"{dataset_label}: correlation of normalized features\n"
        f"{features.shape[0]} hands found in {images_tested} images tested; "
        f"{len(kept)} of {features.shape[1]} features shown\n"
        "name = landmark number + axis; constant features (wrist) are left out"
    )
    fig.colorbar(image, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_before_after(
    raw: np.ndarray,
    normalized: np.ndarray,
    destination: Path,
    dataset_label: str,
    images_tested: int,
    max_hands: int = 20,
) -> None:
    """Scatter plot of the same hands before and after normalization."""
    raw_hands = raw[:max_hands].reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    norm_hands = normalized[:max_hands].reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    colours = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = (
        (axes[0], raw_hands, "Before: raw MediaPipe coordinates"),
        (axes[1], norm_hands, "After: wrist at origin, hand size = 1"),
    )
    for ax, hands, title in panels:
        for number, hand in enumerate(hands):
            ax.scatter(hand[:, 0], hand[:, 1], s=8, color=colours(number % 20))
        ax.invert_yaxis()  # image coordinates: y grows downwards
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        f"{dataset_label}: normalization effect on {len(raw_hands)} hands "
        f"(from {images_tested} images tested, one colour per hand)"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create the feature EDA charts.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Folder with the feature cache. Default: paths.processed_dir.",
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Where to save the charts. Default: paths.figures_dir."
    )
    parser.add_argument(
        "--dataset-label",
        default="ASL baseline",
        help='Name shown in every chart title, "ASL baseline" or "PSL". Default: ASL baseline.',
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Load the cache and save the three charts and the detection table."""
    config = load_config(args.config)
    cache_dir = args.cache_dir or resolve_path(config, "processed_dir")
    out_dir = args.out_dir or resolve_path(config, "figures_dir")
    features_path = cache_dir / FEATURES_FILE
    detections_path = cache_dir / DETECTIONS_FILE
    for path in (features_path, detections_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run data_scripts.features.extract first.")
    label_map = load_label_map(resolve_path(config, "label_map"))
    label_order = sorted(label_map, key=label_map.get)

    records = load_detections(detections_path)
    with np.load(features_path) as cache:
        features, raw = cache["X"], cache["raw"]
    images_tested = len(records)
    logger.info(
        "%s: %d images tested, hand found in %d",
        args.dataset_label,
        images_tested,
        len(features),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rates = detection_rates(records)
    plot_detection_rate(
        rates, label_order, out_dir / "detection_rate_per_class.png", args.dataset_label
    )
    save_detection_table(rates, label_order, out_dir / TABLE_FILE)
    logger.info("Saved detection rate chart and table")
    if len(features) >= 2:
        plot_correlation(
            features,
            out_dir / "feature_correlation_heatmap.png",
            args.dataset_label,
            images_tested,
        )
        logger.info("Saved correlation heatmap")
    else:
        logger.warning("Fewer than 2 hands found: correlation heatmap not created")
    if len(features) >= 1:
        plot_before_after(
            raw,
            features,
            out_dir / "normalization_before_after.png",
            args.dataset_label,
            images_tested,
        )
        logger.info("Saved before/after chart")
    else:
        logger.warning("No hands found: before/after chart not created")
    logger.info("Charts are in %s", out_dir)


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

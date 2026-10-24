"""Draw the raw-data charts for the report from the cleaning report.

Run the cleaning script first (it writes the per-image report this script reads):
    python -m data_scripts.clean.clean_images --source asl
    python -m data_scripts.eda.raw_data_eda --help

Figures are saved in paths.figures_dir. Every title says which dataset it shows: the ASL
baseline or the team's PSL photos.
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image

from data_scripts.clean.clean_images import SOURCE_NAMES, SOURCES, resolve_image_path
from data_scripts.clean.quality import READ_ERRORS
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging

plt.switch_backend("Agg")  # draw to files only, so no display is needed

logger = logging.getLogger(__name__)

COLORS = {"asl": "#2a78d6", "psl": "#eb6834"}  # blue and orange: a colour-blind-safe pair
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID_COLUMNS = 6  # tiles per row in the sample grids
BLUR_BINS = 60
BRIGHTNESS_BINS = 51


def load_report(path: Path) -> pd.DataFrame:
    """Read the per-image cleaning report.

    Raises:
        FileNotFoundError: If the report does not exist yet.
        ValueError: If the report holds no rows.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaning report not found: {path}. Run the cleaning script first."
        )
    report = pd.read_csv(path, keep_default_na=False, dtype={"label": str, "person_id": str})
    if report.empty:
        raise ValueError(f"Cleaning report is empty: {path}")
    for column in ("width", "height", "blur_score", "brightness"):
        report[column] = pd.to_numeric(report[column], errors="coerce")
    return report


def present_sources(report: pd.DataFrame) -> list[str]:
    """Return the datasets found in the report, in a fixed order."""
    return [source for source in SOURCES if (report["source"] == source).any()]


def style_axes(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared chart style: title in ink, quiet grid, no top or right border."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are whole numbers
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(figure: Figure, path: Path) -> None:
    """Save a figure as a PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Saved %s", path)


def plot_class_distribution(report: pd.DataFrame, output: Path) -> None:
    """Bar chart of images per class, one panel per dataset."""
    sources = present_sources(report)
    figure, axes = plt.subplots(len(sources), 1, figsize=(10, 3.6 * len(sources)), squeeze=False)
    for ax, source in zip(axes[:, 0], sources, strict=True):
        counts = report[report["source"] == source]["label"].replace("", "(no label)")
        counts = counts.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=COLORS[source], width=0.6)
        style_axes(ax, f"Images per class: {SOURCE_NAMES[source]}", "Class", "Number of images")
        ax.tick_params(axis="x", rotation=90)
    save_figure(figure, output)


def plot_image_sizes(report: pd.DataFrame, output: Path) -> None:
    """Histograms of image width and height before resizing, one row per dataset."""
    sources = present_sources(report)
    figure, axes = plt.subplots(len(sources), 2, figsize=(10, 3.4 * len(sources)), squeeze=False)
    for row, source in zip(axes, sources, strict=True):
        readable = report[(report["source"] == source) & report["width"].notna()]
        for ax, column in zip(row, ("width", "height"), strict=True):
            ax.hist(readable[column], bins=30, color=COLORS[source])
            style_axes(
                ax,
                f"{column.capitalize()}: {SOURCE_NAMES[source]}",
                f"{column.capitalize()} in pixels",
                "Number of images",
            )
    save_figure(figure, output)


def plot_images_per_person(report: pd.DataFrame, output: Path) -> bool:
    """Bar chart of PSL images per volunteer. Returns False if there are no PSL images."""
    psl = report[(report["source"] == "psl") & (report["person_id"] != "")]
    if psl.empty:
        logger.info("No PSL images with a person_id, so images_per_person.png is skipped.")
        return False
    counts = psl["person_id"].value_counts().sort_index()
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=COLORS["psl"], width=0.6)
    style_axes(ax, "Images per volunteer: PSL", "Volunteer (anonymous ID)", "Number of images")
    save_figure(figure, output)
    return True


def plot_distribution(
    report: pd.DataFrame,
    column: str,
    bins: np.ndarray,
    thresholds: dict[str, float],
    labels: tuple[str, str],
    output: Path,
    log_x: bool = False,
) -> None:
    """Histogram of one score per dataset with the cleaning thresholds as dashed lines."""
    sources = present_sources(report)
    figure, axes = plt.subplots(len(sources), 1, figsize=(9, 3.4 * len(sources)), squeeze=False)
    for ax, source in zip(axes[:, 0], sources, strict=True):
        values = report[(report["source"] == source) & report[column].notna()][column]
        ax.hist(values, bins=bins, color=COLORS[source])
        if log_x:
            ax.set_xscale("log")
        for name, value in thresholds.items():
            ax.axvline(value, color=INK, linestyle="--", linewidth=1.2)
            ax.annotate(
                f"{name} = {value:g}",
                (value, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -12),
                textcoords="offset points",
                color=INK,
                fontsize=9,
            )
        style_axes(ax, f"{labels[0]}: {SOURCE_NAMES[source]}", labels[1], "Number of images")
    save_figure(figure, output)


def plot_blur_scores(report: pd.DataFrame, blur_threshold: float, output: Path) -> None:
    """Histogram of blur scores (variance of the Laplacian) with the blur threshold."""
    positive = report["blur_score"][report["blur_score"] > 0]
    if positive.empty:
        raise ValueError("No image has a blur score above 0, so there is nothing to plot.")
    low = min(float(positive.min()), blur_threshold) * 0.9
    high = max(float(positive.max()), blur_threshold) * 1.1
    bins = np.logspace(math.log10(low), math.log10(high), BLUR_BINS)
    plot_distribution(
        report,
        "blur_score",
        bins,
        {"blur threshold": blur_threshold},
        ("Blur scores", "Variance of the Laplacian (log scale; low means blurry)"),
        output,
        log_x=True,
    )


def plot_brightness(
    report: pd.DataFrame, brightness_min: float, brightness_max: float, output: Path
) -> None:
    """Histogram of mean brightness with the lower and upper thresholds."""
    bins = np.linspace(0, 255, BRIGHTNESS_BINS)
    plot_distribution(
        report,
        "brightness",
        bins,
        {"min": brightness_min, "max": brightness_max},
        ("Image brightness", "Mean grayscale value (0 is black, 255 is white)"),
        output,
    )


def first_images_per_class(report: pd.DataFrame, source: str) -> list[tuple[str, str]]:
    """Return (label, path) for the first readable image of each class, in class order."""
    rows = report[(report["source"] == source) & (report["corrupt"] == 0) & (report["label"] != "")]
    first = rows.sort_values("path").groupby("label", sort=True).first()
    return [(label, path) for label, path in first["path"].items()]


def plot_sample_grid(
    report: pd.DataFrame, source: str, output: Path, root: Path = REPO_ROOT
) -> bool:
    """Save a grid with one raw example image per class. Returns False if there are none."""
    examples = first_images_per_class(report, source)
    if not examples:
        logger.info("No readable labelled %s images, so %s is skipped.", source, output.name)
        return False
    rows = math.ceil(len(examples) / GRID_COLUMNS)
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(2 * GRID_COLUMNS, 2.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, path) in zip(axes.flat, examples, strict=False):
        try:
            with Image.open(resolve_image_path(path, root)) as image:
                ax.imshow(image.convert("RGB"))
        except READ_ERRORS as error:
            logger.warning("Could not draw %s: %s", path, error)
        ax.set_title(label, fontsize=10, color=INK)
    figure.suptitle(f"One raw example per class: {SOURCE_NAMES[source]}", color=INK)
    save_figure(figure, output)
    return True


def log_statistics(report: pd.DataFrame) -> None:
    """Log the numbers behind the charts so the report text can quote them exactly."""
    for source in present_sources(report):
        rows = report[report["source"] == source]
        counts = rows["label"].replace("", "(no label)").value_counts()
        readable = rows[rows["width"].notna()]
        logger.info("%s: %d image files, %d classes", SOURCE_NAMES[source], len(rows), len(counts))
        logger.info(
            "%s: images per class min %d (%s), max %d (%s)",
            SOURCE_NAMES[source],
            counts.min(),
            counts.idxmin(),
            counts.max(),
            counts.idxmax(),
        )
        if not readable.empty:
            logger.info(
                "%s: width %d-%d px, height %d-%d px",
                SOURCE_NAMES[source],
                readable["width"].min(),
                readable["width"].max(),
                readable["height"].min(),
                readable["height"].max(),
            )
            logger.info(
                "%s: blur score median %.1f, brightness median %.1f",
                SOURCE_NAMES[source],
                readable["blur_score"].median(),
                readable["brightness"].median(),
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw the raw-data charts for the report.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--psl-sample-grid",
        action="store_true",
        help="Also save psl_sample_grid.png. Use it only if the volunteers agreed to be shown.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Read the cleaning report and save every raw-data figure."""
    config = load_config(args.config)
    report = load_report(resolve_path(config, "cleaning_report"))
    figures = resolve_path(config, "figures_dir")
    quality = config["quality"]
    log_statistics(report)
    plot_class_distribution(report, figures / "class_distribution.png")
    plot_image_sizes(report, figures / "image_size_distribution.png")
    plot_images_per_person(report, figures / "images_per_person.png")
    plot_blur_scores(report, quality["blur_threshold"], figures / "blur_scores.png")
    plot_brightness(
        report,
        quality["brightness_min"],
        quality["brightness_max"],
        figures / "brightness_distribution.png",
    )
    plot_sample_grid(report, "asl", figures / "asl_sample_grid.png")
    if args.psl_sample_grid:
        plot_sample_grid(report, "psl", figures / "psl_sample_grid.png")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
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

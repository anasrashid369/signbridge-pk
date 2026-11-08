"""Bias check: how the scores change across subgroups of the test images.

The ASL data has no demographic labels, so the subgroups are technical: brightness quartile, blur
quartile, and whether MediaPipe found a hand. Every subgroup gets its size, accuracy, macro F1 and
a seeded bootstrap 95 percent confidence interval. Groups with fewer than 30 images are flagged
and must not be interpreted.

Run from the repository root:
    python -m signbridge.evaluation.bias --predictions reports/predictions/cnn_test.npz \
        --name cnn_test --out reports/bias \
        --landmark-predictions reports/predictions/landmark_test.npz
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

from data_scripts.clean.quality import READ_ERRORS, measure_image  # noqa: E402
from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from signbridge.logging_setup import setup_logging  # noqa: E402
from signbridge.training.predictions import load_predictions  # noqa: E402

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 30
DEFAULT_BOOTSTRAPS = 1000
QUARTILES = (0.25, 0.5, 0.75)
COLUMNS = [
    "measure",
    "group",
    "n",
    "accuracy",
    "accuracy_low",
    "accuracy_high",
    "macro_f1",
    "macro_f1_low",
    "macro_f1_high",
    "low_n",
]


def default_image_root(config: dict[str, Any]) -> Path:
    """Return the ASL training folder named in the data config."""
    folder = config["dataset"]["asl_train_folder"]
    return resolve_path(config, "asl_raw_dir") / folder / folder


def macro_f1(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    """Macro F1 over the classes that occur in the truth or in the predictions.

    F1 of a class is 2 TP / (2 TP + FP + FN), which equals 2 TP / (true count + predicted count).
    """
    true_count = np.bincount(truth, minlength=n_classes)
    predicted_count = np.bincount(predicted, minlength=n_classes)
    hits = np.bincount(truth[truth == predicted], minlength=n_classes)
    total = true_count + predicted_count
    present = total > 0
    return float((2 * hits[present] / total[present]).mean())


def quartile_groups(values: np.ndarray) -> list[str]:
    """Label every value with its quartile of the values, such as ``Q2: 98.0 to 120.5``."""
    low, middle, high = (float(edge) for edge in np.quantile(values, QUARTILES))
    names = [
        f"Q1: below {low:.1f}",
        f"Q2: {low:.1f} to {middle:.1f}",
        f"Q3: {middle:.1f} to {high:.1f}",
        f"Q4: {high:.1f} and above",
    ]
    return [names[int(index)] for index in np.digitize(values, [low, middle, high])]


def bootstrap_interval(
    truth: np.ndarray, predicted: np.ndarray, n_classes: int, repeats: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """95 percent bootstrap intervals for accuracy and macro F1 by resampling images."""
    draws = rng.integers(0, len(truth), size=(repeats, len(truth)))
    accuracy = [float((truth[rows] == predicted[rows]).mean()) for rows in draws]
    f1 = [macro_f1(truth[rows], predicted[rows], n_classes) for rows in draws]
    return {
        "accuracy": (float(np.percentile(accuracy, 2.5)), float(np.percentile(accuracy, 97.5))),
        "macro_f1": (float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))),
    }


def subgroup_row(
    measure: str,
    group: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    n_classes: int,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return size, accuracy, macro F1, their intervals and the small-group flag of a subgroup."""
    interval = bootstrap_interval(truth, predicted, n_classes, repeats, rng)
    return {
        "measure": measure,
        "group": group,
        "n": len(truth),
        "accuracy": float((truth == predicted).mean()),
        "accuracy_low": interval["accuracy"][0],
        "accuracy_high": interval["accuracy"][1],
        "macro_f1": macro_f1(truth, predicted, n_classes),
        "macro_f1_low": interval["macro_f1"][0],
        "macro_f1_high": interval["macro_f1"][1],
        "low_n": len(truth) < MIN_GROUP_SIZE,
    }


def analyse(
    data: dict[str, np.ndarray],
    groupings: dict[str, list[str]],
    repeats: int = DEFAULT_BOOTSTRAPS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return one row per subgroup of every grouping, in a fixed order, with a seeded bootstrap."""
    truth = np.asarray(data["y_true"], dtype=np.int64)
    predicted = data["y_proba"].argmax(axis=1)
    n_classes = len(data["class_names"])
    rng = np.random.default_rng(seed)
    rows = []
    for measure, labels in groupings.items():
        labels_array = np.asarray(labels)
        for group in sorted(set(labels)):
            chosen = labels_array == group
            rows.append(
                subgroup_row(
                    measure, group, truth[chosen], predicted[chosen], n_classes, repeats, rng
                )
            )
    return rows


def measure_images(paths: list[str], image_root: Path) -> dict[str, np.ndarray]:
    """Measure brightness and blur (variance of the Laplacian) of each test image."""
    brightness, blur = [], []
    for path in paths:
        try:
            stats = measure_image(image_root / path)
        except READ_ERRORS as error:
            raise ValueError(f"Cannot read image {image_root / path}: {error}") from error
        brightness.append(stats.brightness)
        blur.append(stats.blur_score)
    return {"brightness": np.asarray(brightness), "blur": np.asarray(blur)}


def detected_from_table(paths: list[str], table: Path) -> list[str]:
    """Label each path ``hand detected`` or ``no hand`` using the extraction detection table."""
    with table.open(newline="", encoding="utf-8") as handle:
        found = {
            row["path"]: row["detected"].strip().lower() == "true" for row in csv.DictReader(handle)
        }
    missing = [path for path in paths if path not in found]
    if missing:
        raise ValueError(f"{len(missing)} test images are not in {table}, for example {missing[0]}")
    return ["hand detected" if found[path] else "no hand" for path in paths]


def detected_from_predictions(paths: list[str], landmark: dict[str, np.ndarray]) -> list[str]:
    """Label each path using the landmark prediction file, which only holds detected images."""
    detected = {str(path) for path in landmark["paths"]}
    return ["hand detected" if path in detected else "no hand" for path in paths]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the subgroup rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, Any]], title: str, path: Path) -> None:
    """Save accuracy per subgroup with its 95 percent interval. Small groups are grey."""
    labels = [f"{row['measure']}: {row['group']} (n={row['n']})" for row in rows]
    accuracy = np.array([row["accuracy"] for row in rows])
    low = accuracy - np.array([row["accuracy_low"] for row in rows])
    high = np.array([row["accuracy_high"] for row in rows]) - accuracy
    colours = ["lightgrey" if row["low_n"] else "tab:blue" for row in rows]
    figure, axis = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.5))
    axis.barh(range(len(rows)), accuracy, xerr=[low, high], color=colours, capsize=3)
    axis.set_yticks(range(len(rows)), labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Accuracy (bars show the bootstrap 95% interval; grey = fewer than 30 images)")
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(
    predictions: Path,
    image_root: Path,
    out_dir: Path,
    name: str,
    detection_table: Path | None,
    landmark_predictions: Path | None,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Measure the test images, build the subgroups, and write ``<name>_subgroups.csv/.png``."""
    data = load_predictions(predictions)
    paths = [str(path) for path in data["paths"]]
    measures = measure_images(paths, image_root)
    groupings = {key: quartile_groups(values) for key, values in measures.items()}
    if detection_table is not None:
        groupings["hand"] = detected_from_table(paths, detection_table)
    elif landmark_predictions is not None:
        groupings["hand"] = detected_from_predictions(paths, load_predictions(landmark_predictions))
    else:
        logger.warning("No detection source given; the hand detected check is skipped")
    rows = analyse(data, groupings, repeats, seed)
    write_rows(out_dir / f"{name}_subgroups.csv", rows)
    plot_rows(rows, f"Accuracy by subgroup: {name}", out_dir / f"{name}_subgroups.png")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Accuracy and macro F1 by test-image subgroup.")
    parser.add_argument("--predictions", type=Path, required=True, help="A test .npz file.")
    parser.add_argument("--name", required=True, help="File name prefix, for example cnn_test.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for the output files.")
    parser.add_argument("--image-root", type=Path, help="ASL train folder (default: from config).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--detections", type=Path, help="Extraction table: path,label,split,detected."
    )
    source.add_argument(
        "--landmark-predictions", type=Path, help="Landmark .npz (detected images)."
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, help="Bootstrap seed (default: seed in the config).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    setup_logging()
    try:
        config = load_config(args.config)
        rows = run(
            args.predictions,
            args.image_root or default_image_root(config),
            args.out,
            args.name,
            args.detections,
            args.landmark_predictions,
            args.bootstraps,
            args.seed if args.seed is not None else int(config["seed"]),
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    flagged = sum(1 for row in rows if row["low_n"])
    logger.info(
        "%d subgroups written to %s (%d under %d images)",
        len(rows),
        args.out,
        flagged,
        MIN_GROUP_SIZE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

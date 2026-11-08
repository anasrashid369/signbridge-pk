"""Compute and save the classification metrics for one set of predictions.

The inputs follow the prediction files written by ``signbridge.training.predictions``:
``y_true`` holds class indices and ``y_proba`` holds one probability row per image, with the
columns in the order of ``class_names``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PROBABILITY_TOLERANCE = 1e-3
PER_CLASS_COLUMNS = ["label", "precision", "recall", "f1", "support"]


def validate_inputs(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Check shapes, label range and probabilities. Returns y_true and y_proba as clean arrays.

    Raises:
        ValueError: If the shapes do not match, a label is out of range, a probability is not
            finite, a row does not sum to 1, or a class has no sample in y_true.
    """
    truth = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=np.float64)
    count = len(class_names)
    if truth.ndim != 1 or proba.ndim != 2 or len(truth) != len(proba) or len(truth) == 0:
        raise ValueError("y_true must be 1-D and y_proba 2-D with the same, non-zero length")
    if proba.shape[1] != count:
        raise ValueError(f"y_proba has {proba.shape[1]} columns but there are {count} classes")
    if truth.min() < 0 or truth.max() >= count:
        raise ValueError(f"y_true must hold class indices from 0 to {count - 1}")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("y_proba must hold finite, non-negative probabilities")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE):
        raise ValueError("Every row of y_proba must sum to 1")
    missing = [class_names[i] for i in range(count) if i not in set(truth.tolist())]
    if missing:
        raise ValueError(f"No sample in y_true for classes {missing}; AUC-ROC is undefined")
    return truth.astype(np.int64), proba / proba.sum(axis=1, keepdims=True)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, macro AUC-ROC and the confusion matrix.

    Args:
        y_true: True class indices, shape (n,).
        y_proba: Predicted probabilities, shape (n, number of classes).
        class_names: Class names in column order.

    Returns:
        A dictionary with ``accuracy``, ``macro_precision``, ``macro_recall``, ``macro_f1``,
        ``macro_auc_roc`` (one-vs-rest), ``per_class`` (precision, recall, f1 and support per
        letter), ``confusion_matrix`` (rows are true classes) and ``n_samples``.

    Raises:
        ValueError: If the inputs are invalid (see ``validate_inputs``).
    """
    truth, proba = validate_inputs(y_true, y_proba, class_names)
    labels = list(range(len(class_names)))
    predicted = proba.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "n_samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auc_roc": float(
            roc_auc_score(truth, proba, multi_class="ovr", average="macro", labels=labels)
        ),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def plot_confusion_matrix(
    matrix: list[list[int]], class_names: list[str], title: str, path: Path
) -> None:
    """Save a confusion matrix figure. Colours show the share of each true class."""
    counts = np.asarray(matrix)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = range(len(class_names))
    axis.set_xticks(ticks, class_names)
    axis.set_yticks(ticks, class_names)
    axis.set_xlabel("Predicted letter")
    axis.set_ylabel("True letter")
    axis.set_title(title)
    for row, col in zip(*np.nonzero(counts), strict=True):
        colour = "white" if shares[row, col] > 0.5 else "black"
        axis.text(
            col, row, str(counts[row, col]), ha="center", va="center", fontsize=6, color=colour
        )
    figure.colorbar(image, ax=axis, label="Share of the true class")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_report(
    metrics: dict[str, Any], class_names: list[str], output_dir: Path, name: str, title: str
) -> list[Path]:
    """Write ``<name>_metrics.json``, ``<name>_per_class.csv`` and ``<name>_confusion.png``.

    Returns:
        The three paths that were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}_metrics.json"
    csv_path = output_dir / f"{name}_per_class.csv"
    figure_path = output_dir / f"{name}_confusion.png"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_CLASS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"label": label, **values})
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, title, figure_path)
    logger.info("Saved metrics for %s to %s", name, output_dir)
    return [json_path, csv_path, figure_path]

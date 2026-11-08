"""Command-line entry point that runs ``metrics.evaluate`` on one prediction file.

Run from the repository root:
    python -m signbridge.evaluation.run_metrics --predictions reports/predictions/cnn_test.npz \
        --name cnn_test --out reports/metrics
Add ``--restrict-to reports/predictions/landmark_test.npz`` to score only the images that
appear in both files (the common set).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from signbridge.evaluation.common_set import restrict_to_common
from signbridge.evaluation.metrics import evaluate, save_report
from signbridge.logging_setup import setup_logging
from signbridge.training.predictions import load_predictions

logger = logging.getLogger(__name__)


def run(predictions: Path, out_dir: Path, name: str, restrict_to: Path | None) -> dict[str, Any]:
    """Evaluate one prediction file, optionally on the images it shares with another file.

    Returns:
        The metrics dictionary from ``evaluate``.

    Raises:
        FileNotFoundError: If a prediction file is missing.
        ValueError: If the files cannot be compared or the predictions are invalid.
    """
    data = load_predictions(predictions)
    if restrict_to is not None:
        data, _ = restrict_to_common(data, load_predictions(restrict_to))
    names = [str(label) for label in data["class_names"]]
    metrics = evaluate(data["y_true"], data["y_proba"], names)
    save_report(metrics, names, out_dir, name, f"Confusion matrix: {name}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute the metrics of a prediction file.")
    parser.add_argument("--predictions", type=Path, required=True, help="A .npz prediction file.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for the output files.")
    parser.add_argument("--name", required=True, help="File name prefix, for example cnn_test.")
    parser.add_argument("--restrict-to", type=Path, help="Keep only images also in this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a data error."""
    args = parse_args(argv)
    setup_logging()
    try:
        metrics = run(args.predictions, args.out, args.name, args.restrict_to)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "%s: %d images, accuracy %.4f, macro F1 %.4f, macro AUC-ROC %.4f",
        args.name,
        metrics["n_samples"],
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["macro_auc_roc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

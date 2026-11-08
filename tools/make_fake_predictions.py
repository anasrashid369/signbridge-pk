"""Write a clearly fake prediction file in the team contract format.

The evaluation modules (error analysis, bias check) need a prediction file
before the real CNN predictions exist. This command writes one with the same
four arrays as ``reports/predictions/<model>_<split>.npz``: ``y_true`` (class
indices), ``y_proba`` (rows of softmax probabilities), ``paths`` and
``class_names``. Every path starts with ``fake/`` and the default file name says
``fake``.

The numbers are synthetic. Never commit the output and never report metrics
computed from it.

Run from the repository root, writing outside the repository:
    python -m tools.make_fake_predictions --output PATH/OUTSIDE/THE/REPO/fake_cnn_test.npz
"""

from __future__ import annotations

import argparse
import logging
import string
import tempfile
from pathlib import Path

import numpy as np

from signbridge.config import REPO_ROOT
from signbridge.training.predictions import save_predictions

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "signbridge_fake_predictions.npz"
DEFAULT_ROWS = 200
DEFAULT_CLASSES = 26
DEFAULT_SEED = 42
DEFAULT_ACCURACY = 0.8
DIRICHLET_CONCENTRATION = 0.3
PREDICTED_MASS = 0.6


def class_names(count: int) -> list[str]:
    """Return class names in index order: A-Z, then ``class_26`` onwards.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"classes must be positive, got {count}")
    letters = list(string.ascii_uppercase)
    return [letters[index] if index < len(letters) else f"class_{index}" for index in range(count)]


def make_fake_predictions(
    rows: int, num_classes: int, seed: int, accuracy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic truth, probabilities and predicted classes.

    Args:
        rows: Number of fake images.
        num_classes: Number of classes (26 for A-Z).
        seed: Seed for NumPy's generator, so the file is reproducible.
        accuracy: Share of rows whose most likely class is the true class.

    Returns:
        ``y_true`` (int64, shape (rows,)), ``y_proba`` (float64, shape
        (rows, num_classes), rows summing to 1) and the predicted class index
        per row.

    Raises:
        ValueError: If rows, classes or accuracy are out of range.
    """
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    if num_classes <= 0:
        raise ValueError(f"classes must be positive, got {num_classes}")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be between 0 and 1, got {accuracy}")
    generator = np.random.default_rng(seed)
    y_true = generator.integers(0, num_classes, size=rows)
    correct = generator.random(rows) < accuracy
    shift = generator.integers(1, num_classes, size=rows)
    predicted = np.where(correct, y_true, (y_true + shift) % num_classes)
    y_proba = generator.dirichlet(np.full(num_classes, DIRICHLET_CONCENTRATION), size=rows)
    y_proba *= 1.0 - PREDICTED_MASS
    y_proba[np.arange(rows), predicted] += PREDICTED_MASS
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    return y_true.astype(np.int64), y_proba, predicted.astype(np.int64)


def fake_paths(y_true: np.ndarray, names: list[str]) -> list[str]:
    """Return one clearly fake path per row, using the true class name."""
    return [f"fake/{names[label_index]}/{row:05d}.jpg" for row, label_index in enumerate(y_true)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write a FAKE prediction file in the team contract format."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .npz outside the repository (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of fake rows.")
    parser.add_argument("--classes", type=int, default=DEFAULT_CLASSES, help="Number of classes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for the fake data.")
    parser.add_argument(
        "--accuracy",
        type=float,
        default=DEFAULT_ACCURACY,
        help="Share of rows where the top class is the true class (0-1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a bad path or argument."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        output = args.output.resolve()
        if output.is_relative_to(REPO_ROOT.resolve()):
            raise ValueError(
                f"Refusing to write inside the repository ({REPO_ROOT}); choose a path outside it."
            )
        names = class_names(args.classes)
        y_true, y_proba, _ = make_fake_predictions(
            args.rows, args.classes, args.seed, args.accuracy
        )
        save_predictions(output, y_true, y_proba, fake_paths(y_true, names), names)
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Wrote FAKE predictions to %s", output)
    logger.info("Synthetic numbers: never commit this file or report metrics from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Save and load prediction files in the team contract format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PREDICTION_KEYS = ("y_true", "y_proba", "paths", "class_names")


def save_predictions(
    path: Path,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    paths: list[str],
    class_names: list[str],
) -> Path:
    """Write y_true, y_proba, paths and class_names to a compressed .npz file.

    Args:
        path: Destination file, for example ``reports/predictions/cnn_val.npz``.
        y_true: Integer class indices, shape (n,).
        y_proba: Softmax probabilities, shape (n, num_classes).
        paths: Split paths in the same order as y_true.
        class_names: Class names in index order.

    Returns:
        The destination path.

    Raises:
        ValueError: If the array lengths or probability shape do not match.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_proba, dtype=np.float32)
    if truth.ndim != 1 or probability.ndim != 2:
        raise ValueError("y_true must be 1-D and y_proba must be 2-D")
    if len(truth) != len(probability) or len(truth) != len(paths):
        raise ValueError("y_true, y_proba and paths must have the same length")
    if probability.shape[1] != len(class_names):
        raise ValueError("y_proba columns must equal the number of class names")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=truth,
        y_proba=probability,
        paths=np.asarray(paths, dtype=str),
        class_names=np.asarray(class_names, dtype=str),
    )
    return path


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Load a prediction file and return its four arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If one of the contract keys is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in PREDICTION_KEYS if key not in data]
        if missing:
            raise ValueError(f"Prediction file is missing keys: {missing}")
        return {key: data[key] for key in PREDICTION_KEYS}

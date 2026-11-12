"""Shared prediction types for every inference backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """One class label and its probability.

    Attributes:
        label: Class name, for example ``"A"``.
        probability: Probability of the class, in the range 0 to 1.
    """

    label: str
    probability: float


class Predictor(Protocol):
    """Interface implemented by the CNN and the landmark predictors.

    ``predict`` receives an RGB image as a uint8 array of shape
    (height, width, 3). The probabilities are computed over all classes, sum
    to 1 before truncation, and the returned list is sorted from highest to
    lowest probability.
    """

    name: str

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        """Return the top ``top_k`` predictions for one RGB image."""
        ...


def load_labels(label_map_path: str | Path) -> list[str]:
    """Return the class names in model order from ``label_map.json``.

    The file maps each class name to its integer index. The indices must be
    exactly ``0..n-1``; otherwise the model output columns would not match.

    Args:
        label_map_path: Path to the JSON file, for example
            ``data/processed/label_map.json``.

    Returns:
        The class names ordered by their index.

    Raises:
        FileNotFoundError: If the label map does not exist.
        ValueError: If the file is not a non-empty mapping, an index is not an
            integer, or the indices are not exactly ``0..n-1``.
    """
    path = Path(label_map_path)
    if not path.is_file():
        raise FileNotFoundError(f"Label map not found: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Label map must be a non-empty mapping: {path}")
    if not all(isinstance(index, int) for index in mapping.values()):
        raise ValueError(f"Label map indices must be integers: {path}")
    pairs = sorted(mapping.items(), key=lambda item: item[1])
    indices = [index for _, index in pairs]
    if indices != list(range(len(pairs))):
        raise ValueError(f"Label map indices must be 0..{len(pairs) - 1} without gaps: {path}")
    labels = [str(name) for name, _ in pairs]
    if any(not label for label in labels):
        raise ValueError(f"Label map contains an empty class name: {path}")
    return labels

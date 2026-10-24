"""Helpers that create small synthetic images for the tests."""

from pathlib import Path

import numpy as np
from PIL import Image


def make_pattern(path: Path, seed: int, noise: float = 3.0, low: int = 0, high: int = 256) -> Path:
    """Save a sharp 80x90 block pattern. The same seed gives the same pattern."""
    blocks = np.random.default_rng(seed).integers(low, high, size=(8, 9))
    image = np.kron(blocks, np.ones((10, 10)))
    image = image + np.random.default_rng(seed + 1000).normal(0, noise, image.shape)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
    return path

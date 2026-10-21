"""Image resizing and pixel scaling for the CNN feature set (feature set A)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def get_image_size(config: dict[str, Any]) -> int:
    """Return the target side length in pixels from ``image.size`` in the config."""
    return int(config["image"]["size"])


def load_and_preprocess(path: Path, size: int) -> np.ndarray:
    """Load an image, resize it to size x size, and scale pixels to the range 0 to 1.

    Args:
        path: Path to the image file.
        size: Target side length in pixels (from the config).

    Returns:
        A float32 array of shape (size, size, 3).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file is corrupt or not an image.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size))
        pixels = np.asarray(rgb, dtype=np.float32)
    return pixels / 255.0

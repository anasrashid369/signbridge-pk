"""Per-image measurements and quality flags used by the cleaning scripts."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

HASH_WIDTH = 9  # one column more than the hash width, so each row gives 8 left-right comparisons
HASH_HEIGHT = 8
READ_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)


@dataclass(frozen=True)
class ImageStats:
    """Measurements of one readable image."""

    width: int
    height: int
    sha256: str
    blur_score: float
    brightness: float
    dhash: int


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file's bytes, used to find exact duplicates."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(gray: np.ndarray) -> int:
    """Return a 64-bit hash that stays almost the same for near-identical images.

    The image is shrunk to 9x8 pixels and each pixel is compared with its right neighbour.
    """
    small = cv2.resize(gray, (HASH_WIDTH, HASH_HEIGHT), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def measure_image(path: Path) -> ImageStats:
    """Read an image and measure its size, hash, sharpness and brightness.

    Raises:
        OSError, ValueError, SyntaxError: If the file is empty, corrupt or not an image.
    """
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        gray = np.asarray(image.convert("L"))
    return ImageStats(
        width=width,
        height=height,
        sha256=file_sha256(path),
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        dhash=difference_hash(gray),
    )


def quality_flags(
    width: int, height: int, blur_score: float, brightness: float, thresholds: dict[str, Any]
) -> dict[str, int]:
    """Flag an image as too small, blurry, too dark or too bright using the config thresholds.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        blur_score: Variance of the Laplacian; low values mean a blurry image.
        brightness: Mean grayscale value from 0 to 255.
        thresholds: The ``quality`` section of configs/data.yaml.

    Returns:
        A mapping of flag name to 1 (flagged) or 0.
    """
    return {
        "too_small": int(min(width, height) < thresholds["min_side"]),
        "blurry": int(blur_score < thresholds["blur_threshold"]),
        "too_dark": int(brightness < thresholds["brightness_min"]),
        "too_bright": int(brightness > thresholds["brightness_max"]),
    }

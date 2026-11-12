"""Turn uploads into the RGB uint8 arrays the predictors expect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from signbridge.inference.errors import InvalidImage

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MiB, the app's upload limit
MAX_SIDE = 4000  # pixels; larger images are rejected instead of silently shrunk
_ALLOWED_MODES = frozenset({"RGB", "RGBA", "L", "P"})


def load_rgb_array(source: Image.Image | str | Path | np.ndarray) -> np.ndarray:
    """Return ``source`` as a uint8 RGB array of shape (height, width, 3).

    Args:
        source: A PIL image, a path to an image file, or an RGB uint8 array.
            Gradio's webcam gives an RGB array; OpenCV gives BGR, so convert
            OpenCV frames before calling this helper.

    Returns:
        A contiguous uint8 array with three channels.

    Raises:
        InvalidImage: If the file is missing or unreadable, larger than
            ``MAX_FILE_BYTES``, has a side longer than ``MAX_SIDE``, has a
            mode that cannot be converted to RGB, or is not a uint8 RGB array.
    """
    if isinstance(source, np.ndarray):
        return _check_array(source)
    if isinstance(source, Image.Image):
        return _from_pil(source)
    path = Path(source)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InvalidImage(f"Could not read the image file: {path}") from error
    if size > MAX_FILE_BYTES:
        raise InvalidImage(
            f"Image is {size / (1024 * 1024):.1f} MB; the limit is "
            f"{MAX_FILE_BYTES / (1024 * 1024):.0f} MB."
        )
    try:
        with Image.open(path) as image:
            image.load()
            return _from_pil(image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImage(f"Could not read the image file: {path}") from error


def _from_pil(image: Image.Image) -> np.ndarray:
    """Validate a PIL image, apply EXIF orientation and convert it to RGB."""
    width, height = image.size
    if width < 1 or height < 1:
        raise InvalidImage("The image is empty.")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise InvalidImage(
            f"Image is {width}x{height} px; each side must be at most {MAX_SIDE} px."
        )
    if image.mode not in _ALLOWED_MODES:
        raise InvalidImage(f"Unsupported image mode {image.mode!r}; use an RGB image.")
    image = ImageOps.exif_transpose(image)
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _check_array(image: np.ndarray) -> np.ndarray:
    """Validate an array from Gradio or OpenCV and return it as uint8 RGB."""
    if image.dtype != np.uint8:
        raise InvalidImage(f"Image array must be uint8, got {image.dtype}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImage(f"Image array must have shape (height, width, 3), got {image.shape}.")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise InvalidImage("The image is empty.")
    if height > MAX_SIDE or width > MAX_SIDE:
        raise InvalidImage(
            f"Image is {width}x{height} px; each side must be at most {MAX_SIDE} px."
        )
    return np.ascontiguousarray(image)

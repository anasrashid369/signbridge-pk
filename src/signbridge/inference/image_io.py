"""Turn uploads into the RGB uint8 arrays the predictors expect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from signbridge.inference.errors import InvalidImage

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MiB, the app's upload limit
MAX_SIDE = 4000  # pixels; larger images are rejected instead of silently shrunk
_ALLOWED_MODES = frozenset({"RGB", "RGBA", "L", "P"})


def load_rgb_array(
    source: Image.Image | str | Path | np.ndarray,
    max_bytes: int | None = None,
    max_side: int | None = None,
) -> np.ndarray:
    """Return ``source`` as a uint8 RGB array of shape (height, width, 3).

    Args:
        source: A PIL image, a path to an image file, or an RGB uint8 array.
            Gradio's webcam gives an RGB array; OpenCV gives BGR, so convert
            OpenCV frames before calling this helper.
        max_bytes: Largest allowed file size in bytes. Defaults to
            ``MAX_FILE_BYTES``; the app passes its configured limit.
        max_side: Largest allowed width or height in pixels. Defaults to
            ``MAX_SIDE``.

    Returns:
        A contiguous uint8 array with three channels.

    Raises:
        InvalidImage: If the file is missing or unreadable, larger than
            ``max_bytes``, has a side longer than ``max_side``, has a mode
            that cannot be converted to RGB, or is not a uint8 RGB array.
    """
    max_bytes = MAX_FILE_BYTES if max_bytes is None else max_bytes
    max_side = MAX_SIDE if max_side is None else max_side
    if isinstance(source, np.ndarray):
        return _check_array(source, max_side)
    if isinstance(source, Image.Image):
        return _from_pil(source, max_side)
    path = Path(source)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InvalidImage(f"Could not read the image file: {path}") from error
    if size > max_bytes:
        raise InvalidImage(
            f"Image is {size / (1024 * 1024):.1f} MB; the limit is "
            f"{max_bytes / (1024 * 1024):.0f} MB."
        )
    try:
        with Image.open(path) as image:
            image.load()
            return _from_pil(image, max_side)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImage(f"Could not read the image file: {path}") from error


def _from_pil(image: Image.Image, max_side: int) -> np.ndarray:
    """Validate a PIL image, apply EXIF orientation and convert it to RGB."""
    width, height = image.size
    if width < 1 or height < 1:
        raise InvalidImage("The image is empty.")
    if width > max_side or height > max_side:
        raise InvalidImage(
            f"Image is {width}x{height} px; each side must be at most {max_side} px."
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


def _check_array(image: np.ndarray, max_side: int) -> np.ndarray:
    """Validate an array from Gradio or OpenCV and return it as uint8 RGB."""
    if image.dtype != np.uint8:
        raise InvalidImage(f"Image array must be uint8, got {image.dtype}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImage(f"Image array must have shape (height, width, 3), got {image.shape}.")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise InvalidImage("The image is empty.")
    if height > max_side or width > max_side:
        raise InvalidImage(
            f"Image is {width}x{height} px; each side must be at most {max_side} px."
        )
    return np.ascontiguousarray(image)

"""Draw MediaPipe hand landmarks on an image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from signbridge.inference.image_io import load_rgb_array

NUM_LANDMARKS = 21  # MediaPipe gives 21 points per hand
NUM_COORDS = 3  # each point has x, y, z

# The 21 bones of the MediaPipe hand: five palm links, then three per finger.
HAND_CONNECTIONS = (
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),  # palm
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (5, 6),
    (6, 7),
    (7, 8),  # index finger
    (9, 10),
    (10, 11),
    (11, 12),  # middle finger
    (13, 14),
    (14, 15),
    (15, 16),  # ring finger
    (17, 18),
    (18, 19),
    (19, 20),  # little finger
)

LINE_COLOR = (0, 200, 0)  # green bones
POINT_COLOR = (255, 0, 0)  # red joints


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray | None:
    """Return a copy of ``image`` with the 21 hand landmarks drawn on it.

    Args:
        image: uint8 RGB array of shape (height, width, 3). It is never
            changed; the drawing is made on a copy.
        landmarks: Raw MediaPipe landmarks of shape (21, 3), as returned by
            ``LandmarkPredictor.detect_landmarks``. x and y are fractions of
            the image width and height. z is not used. None means no hand
            was found.

    Returns:
        A new uint8 RGB array with the same shape as ``image``, or None if
        ``landmarks`` is None.

    Raises:
        InvalidImage: If ``image`` is not a uint8 RGB array within limits.
        ValueError: If ``landmarks`` does not have shape (21, 3) or holds
            values that are not finite.
    """
    array = load_rgb_array(image)
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(
            f"Expected landmarks of shape {(NUM_LANDMARKS, NUM_COORDS)}, got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks must be finite numbers.")

    height, width = array.shape[:2]
    # load_rgb_array can return the caller's own array, so copy before drawing.
    canvas = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(canvas)
    pixels = [(round(x * width), round(y * height)) for x, y, _ in points]

    scale = min(height, width)
    line_width = max(1, round(scale / 200))
    radius = max(2, round(scale / 100))
    for start, end in HAND_CONNECTIONS:
        draw.line([pixels[start], pixels[end]], fill=LINE_COLOR, width=line_width)
    for x, y in pixels:  # joints last, so they sit on top of the bones
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=POINT_COLOR)
    return np.array(canvas, dtype=np.uint8)

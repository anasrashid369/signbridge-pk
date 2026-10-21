"""Landmark normalization: make hand features independent of position and size."""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21  # MediaPipe gives 21 points per hand
NUM_COORDS = 3  # each point has x, y, z
WRIST_INDEX = 0  # landmark 0 is the wrist
MIDDLE_FINGER_BASE_INDEX = 9  # landmark 9 is the base of the middle finger
MIN_HAND_SIZE = 1e-8  # below this the hand size is treated as zero


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize one hand's landmarks so position and size no longer matter.

    Step 1: subtract the wrist, so the wrist becomes (0, 0, 0).
    Step 2: divide by the wrist-to-middle-finger-base distance (hand size).

    Args:
        landmarks: Array of shape (21, 3) holding x, y, z for each landmark.

    Returns:
        A flat array of 63 numbers (21 landmarks x 3 coordinates).

    Raises:
        ValueError: If the shape is wrong or the hand size is zero.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(f"Expected shape {(NUM_LANDMARKS, NUM_COORDS)}, got {points.shape}")

    shifted = points - points[WRIST_INDEX]
    hand_size = float(np.linalg.norm(shifted[MIDDLE_FINGER_BASE_INDEX]))
    if hand_size < MIN_HAND_SIZE:
        raise ValueError("Hand size is zero; cannot normalize this hand.")

    return (shifted / hand_size).flatten()

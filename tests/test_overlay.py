"""Tests for the landmark overlay."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.inference import InvalidImage
from signbridge.inference.overlay import (
    HAND_CONNECTIONS,
    NUM_LANDMARKS,
    POINT_COLOR,
    draw_landmarks,
)

HEIGHT, WIDTH = 100, 200


@pytest.fixture
def image() -> np.ndarray:
    """A black image, so any drawn pixel is easy to see."""
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def landmarks() -> np.ndarray:
    """Landmark i sits at x = 0.1 + 0.04 i and y = 0.2 + 0.03 i."""
    index = np.arange(NUM_LANDMARKS)
    return np.stack([0.1 + 0.04 * index, 0.2 + 0.03 * index, np.zeros(NUM_LANDMARKS)], axis=1)


def test_connections_are_valid() -> None:
    assert len(HAND_CONNECTIONS) == 21
    assert len(set(HAND_CONNECTIONS)) == 21
    assert all(0 <= a < NUM_LANDMARKS and 0 <= b < NUM_LANDMARKS for a, b in HAND_CONNECTIONS)


def test_no_landmarks_returns_none(image: np.ndarray) -> None:
    assert draw_landmarks(image, None) is None


def test_output_has_same_shape_and_dtype(image: np.ndarray, landmarks: np.ndarray) -> None:
    result = draw_landmarks(image, landmarks)
    assert result is not None
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_pixels_change(image: np.ndarray, landmarks: np.ndarray) -> None:
    result = draw_landmarks(image, landmarks)
    assert result is not None
    assert result.any()


def test_input_is_not_changed(image: np.ndarray, landmarks: np.ndarray) -> None:
    before = image.copy()
    result = draw_landmarks(image, landmarks)
    assert result is not None
    assert np.array_equal(image, before)
    assert not np.shares_memory(result, image)


def test_joint_is_drawn_at_its_pixel(image: np.ndarray, landmarks: np.ndarray) -> None:
    result = draw_landmarks(image, landmarks)
    assert result is not None
    # Landmark 8: x = 0.42 * 200 = 84, y = 0.44 * 100 = 44. Array order is [row, column].
    assert tuple(result[44, 84]) == POINT_COLOR


def test_points_outside_the_image_do_not_crash(image: np.ndarray, landmarks: np.ndarray) -> None:
    landmarks[3] = (1.4, -0.3, 0.0)
    assert draw_landmarks(image, landmarks) is not None


def test_rejects_wrong_landmark_shape(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        draw_landmarks(image, np.zeros((20, 3)))


def test_rejects_non_finite_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    landmarks[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        draw_landmarks(image, landmarks)


def test_rejects_non_uint8_image(landmarks: np.ndarray) -> None:
    with pytest.raises(InvalidImage, match="uint8"):
        draw_landmarks(np.zeros((32, 32, 3), dtype=np.float32), landmarks)

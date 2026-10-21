"""Tests for landmark normalization."""

import numpy as np
import pytest

from data_scripts.features.landmarks import normalize_landmarks


def make_fake_hand() -> np.ndarray:
    """Return a reproducible random hand of shape (21, 3)."""
    rng = np.random.default_rng(0)
    return rng.random((21, 3))


def test_output_has_63_values() -> None:
    assert normalize_landmarks(make_fake_hand()).shape == (63,)


def test_wrist_becomes_origin() -> None:
    result = normalize_landmarks(make_fake_hand()).reshape(21, 3)
    assert np.allclose(result[0], 0.0)


def test_shifted_hand_gives_same_features() -> None:
    hand = make_fake_hand()
    moved = hand + np.array([0.3, -0.2, 0.1])
    assert np.allclose(normalize_landmarks(hand), normalize_landmarks(moved))


def test_scaled_hand_gives_same_features() -> None:
    hand = make_fake_hand()
    assert np.allclose(normalize_landmarks(hand), normalize_landmarks(hand * 2.5))


def test_shifted_and_scaled_hand_gives_same_features() -> None:
    hand = make_fake_hand()
    changed = hand * 0.4 + np.array([5.0, 1.0, -2.0])
    assert np.allclose(normalize_landmarks(hand), normalize_landmarks(changed))


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3)))


def test_zero_hand_size_raises() -> None:
    with pytest.raises(ValueError):
        normalize_landmarks(np.ones((21, 3)))

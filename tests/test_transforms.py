"""Tests for the CNN image transforms."""


import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image

from signbridge.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def test_train_transform_has_no_horizontal_flip() -> None:
    names = [type(step).__name__ for step in build_train_transform(32).transforms]
    assert "RandomHorizontalFlip" not in names
    assert "RandomRotation" in names
    assert "ColorJitter" in names


def test_eval_transform_is_deterministic() -> None:
    names = [type(step).__name__ for step in build_eval_transform(32).transforms]
    assert "RandomRotation" not in names
    assert "ColorJitter" not in names
    image = Image.fromarray(np.full((40, 40, 3), 128, dtype=np.uint8))
    transform = build_eval_transform(32)
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 32, 32)
    assert first.dtype.is_floating_point
    assert np.allclose(first.numpy(), second.numpy())
    expected = (128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert np.isclose(float(first[0, 0, 0]), expected)

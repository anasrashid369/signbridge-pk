"""Tests for image resizing and pixel scaling."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_scripts.features.images import get_image_size, load_and_preprocess


def make_image(path: Path, width: int, height: int) -> Path:
    """Save a plain grey test image and return its path."""
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(path)
    return path


def test_output_shape_and_type(tmp_path: Path) -> None:
    result = load_and_preprocess(make_image(tmp_path / "a.png", 200, 200), 64)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.float32


def test_non_square_image_is_resized(tmp_path: Path) -> None:
    result = load_and_preprocess(make_image(tmp_path / "b.png", 300, 100), 64)
    assert result.shape == (64, 64, 3)


def test_pixels_are_scaled_to_0_1(tmp_path: Path) -> None:
    result = load_and_preprocess(make_image(tmp_path / "c.png", 50, 50), 32)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert np.allclose(result, 128 / 255)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_preprocess(tmp_path / "missing.png", 64)


def test_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_text("this is not an image")
    with pytest.raises(OSError):
        load_and_preprocess(bad, 64)


def test_get_image_size_reads_config() -> None:
    assert get_image_size({"image": {"size": 224}}) == 224

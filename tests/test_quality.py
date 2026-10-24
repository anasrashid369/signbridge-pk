"""Tests for data_scripts/clean/quality.py."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_scripts.clean.quality import difference_hash, file_sha256, measure_image, quality_flags
from tests.image_helpers import make_pattern

THRESHOLDS = {"blur_threshold": 100.0, "brightness_min": 40, "brightness_max": 220, "min_side": 64}


def test_measure_image_reports_size_and_scores(tmp_path: Path) -> None:
    stats = measure_image(make_pattern(tmp_path / "a.png", seed=1))
    assert (stats.width, stats.height) == (90, 80)
    assert stats.blur_score > THRESHOLDS["blur_threshold"]
    assert THRESHOLDS["brightness_min"] < stats.brightness < THRESHOLDS["brightness_max"]
    assert len(stats.sha256) == 64


def test_measure_image_flat_image_has_zero_blur_score(tmp_path: Path) -> None:
    path = tmp_path / "flat.png"
    Image.fromarray(np.full((80, 90), 128, dtype=np.uint8)).save(path)
    assert measure_image(path).blur_score == 0.0


def test_measure_image_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    with pytest.raises(OSError):
        measure_image(path)


def test_measure_image_corrupt_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(OSError):
        measure_image(path)


def test_same_file_content_gives_same_sha256(tmp_path: Path) -> None:
    first = make_pattern(tmp_path / "a.png", seed=1)
    second = tmp_path / "b.png"
    second.write_bytes(first.read_bytes())
    assert file_sha256(first) == file_sha256(second)


def test_difference_hash_is_stable_for_small_noise(tmp_path: Path) -> None:
    first = measure_image(make_pattern(tmp_path / "a.png", seed=1, noise=2.0))
    second = measure_image(make_pattern(tmp_path / "b.png", seed=1, noise=3.0))
    assert bin(first.dhash ^ second.dhash).count("1") <= 4


def test_difference_hash_differs_for_different_images(tmp_path: Path) -> None:
    first = measure_image(make_pattern(tmp_path / "a.png", seed=1))
    second = measure_image(make_pattern(tmp_path / "b.png", seed=2))
    assert bin(first.dhash ^ second.dhash).count("1") > 4


def test_difference_hash_of_flat_image_is_zero() -> None:
    assert difference_hash(np.full((50, 50), 7, dtype=np.uint8)) == 0


def test_quality_flags_good_image_has_no_flags() -> None:
    flags = quality_flags(200, 200, 500.0, 120.0, THRESHOLDS)
    assert sum(flags.values()) == 0


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        ((30, 200, 500.0, 120.0), "too_small"),
        ((200, 30, 500.0, 120.0), "too_small"),
        ((200, 200, 20.0, 120.0), "blurry"),
        ((200, 200, 500.0, 10.0), "too_dark"),
        ((200, 200, 500.0, 250.0), "too_bright"),
    ],
)
def test_quality_flags_flag_each_problem(args: tuple, flag: str) -> None:
    flags = quality_flags(*args, THRESHOLDS)
    assert flags[flag] == 1
    assert sum(flags.values()) == 1

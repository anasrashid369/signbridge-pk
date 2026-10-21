"""Tests for the landmark extraction helpers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from data_scripts.features.extract import (
    create_landmarker,
    extract_features,
    find_images,
    landmarks_to_array,
    save_cache,
    save_detections,
)
from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path

LABEL_MAP = {"A": 0, "B": 1}


def touch(path: Path) -> None:
    """Create an empty file, including its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_landmarks_to_array_shape() -> None:
    points = [SimpleNamespace(x=0.1, y=0.2, z=0.3) for _ in range(21)]
    assert landmarks_to_array(points).shape == (21, 3)


def test_find_images_uses_folder_name_as_label(tmp_path: Path) -> None:
    touch(tmp_path / "A" / "1.jpg")
    touch(tmp_path / "A" / "2.JPG")
    touch(tmp_path / "b" / "1.png")
    touch(tmp_path / "junk" / "1.jpg")
    touch(tmp_path / "A" / "notes.txt")
    items, skipped = find_images(tmp_path, LABEL_MAP, max_per_class=0, seed=0)
    assert sorted(label for _, label in items) == ["A", "A", "B"]
    assert skipped == 1


def test_find_images_limits_per_class(tmp_path: Path) -> None:
    for number in range(5):
        touch(tmp_path / "A" / f"{number}.jpg")
    for number in range(3):
        touch(tmp_path / "B" / f"{number}.jpg")
    items, _ = find_images(tmp_path, LABEL_MAP, max_per_class=2, seed=0)
    labels = [label for _, label in items]
    assert labels.count("A") == 2
    assert labels.count("B") == 2


def test_find_images_sampling_is_reproducible(tmp_path: Path) -> None:
    for number in range(10):
        touch(tmp_path / "A" / f"{number}.jpg")
    first, _ = find_images(tmp_path, LABEL_MAP, max_per_class=3, seed=7)
    second, _ = find_images(tmp_path, LABEL_MAP, max_per_class=3, seed=7)
    assert first == second


def test_save_detections_writes_rows(tmp_path: Path) -> None:
    destination = save_detections(
        tmp_path, [("A/1.jpg", "A", True), ("A/2.jpg", "A", False)]
    )
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,label,detected"
    assert lines[2] == "A/2.jpg,A,False"


def test_save_cache_round_trip(tmp_path: Path) -> None:
    vectors = [np.zeros(63), np.ones(63)]
    destination = save_cache(tmp_path, vectors, [0, 1], ["A/1.jpg", "B/1.jpg"])
    data = np.load(destination)
    assert data["X"].shape == (2, 63)
    assert list(data["y"]) == [0, 1]
    assert list(data["paths"]) == ["A/1.jpg", "B/1.jpg"]


def test_save_cache_with_no_images(tmp_path: Path) -> None:
    data = np.load(save_cache(tmp_path, [], [], []))
    assert data["X"].shape == (0, 63)


def test_blank_image_has_no_hand(tmp_path: Path) -> None:
    pytest.importorskip("mediapipe")
    model = resolve_path(load_config(DEFAULT_CONFIG_PATH), "hand_model")
    if not model.is_file():
        pytest.skip("hand landmarker model not downloaded")
    blank = tmp_path / "blank.png"
    Image.new("RGB", (224, 224), color=(200, 200, 200)).save(blank)
    with create_landmarker(model) as landmarker:
        assert extract_features(landmarker, blank) is None

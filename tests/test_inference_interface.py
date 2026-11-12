"""Tests for the shared inference interface. These tests do not need torch."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import signbridge.inference.image_io as image_io
from signbridge.inference import (
    InvalidImage,
    Prediction,
    Predictor,
    load_labels,
    load_rgb_array,
)


def write_label_map(path: Path, mapping: dict[str, int]) -> Path:
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_prediction_is_frozen() -> None:
    prediction = Prediction(label="A", probability=0.75)
    assert prediction.label == "A"
    assert prediction.probability == 0.75
    with pytest.raises(FrozenInstanceError):
        prediction.label = "B"  # type: ignore[misc]


def test_prediction_equality() -> None:
    assert Prediction("A", 0.5) == Prediction("A", 0.5)
    assert Prediction("A", 0.5) != Prediction("B", 0.5)


class FakePredictor:
    name = "fake"

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        return [Prediction("A", 0.6), Prediction("B", 0.4)][:top_k]


def test_protocol_accepts_a_duck_typed_class() -> None:
    predictor: Predictor = FakePredictor()
    result = predictor.predict(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result == [Prediction("A", 0.6), Prediction("B", 0.4)]


def test_load_labels_orders_by_index(tmp_path: Path) -> None:
    path = write_label_map(tmp_path / "label_map.json", {"B": 1, "A": 0, "C": 2})
    assert load_labels(path) == ["A", "B", "C"]


def test_load_labels_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "missing.json")


def test_load_labels_rejects_gaps(tmp_path: Path) -> None:
    path = write_label_map(tmp_path / "label_map.json", {"A": 0, "C": 2})
    with pytest.raises(ValueError):
        load_labels(path)


def test_load_labels_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "label_map.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labels(path)


def test_load_labels_rejects_non_integer_indices(tmp_path: Path) -> None:
    path = write_label_map(tmp_path / "label_map.json", {"A": "0"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        load_labels(path)


def test_load_rgb_array_from_path(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(path)
    array = load_rgb_array(path)
    assert array.shape == (6, 8, 3)
    assert array.dtype == np.uint8
    assert tuple(array[0, 0]) == (10, 20, 30)


def test_load_rgb_array_from_pil_image() -> None:
    array = load_rgb_array(Image.new("RGB", (4, 3), (1, 2, 3)))
    assert array.shape == (3, 4, 3)
    assert tuple(array[0, 0]) == (1, 2, 3)


def test_load_rgb_array_from_array() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    source[1, 2] = (9, 8, 7)
    array = load_rgb_array(source)
    assert array.flags["C_CONTIGUOUS"]
    assert tuple(array[1, 2]) == (9, 8, 7)


def test_load_rgb_array_composites_transparency_on_white() -> None:
    image = Image.new("RGBA", (2, 1))
    image.putpixel((0, 0), (255, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 255, 255))
    array = load_rgb_array(image)
    assert tuple(array[0, 0]) == (255, 255, 255)
    assert tuple(array[0, 1]) == (0, 0, 255)


def test_load_rgb_array_converts_grayscale() -> None:
    array = load_rgb_array(Image.new("L", (2, 2), 128))
    assert array.shape == (2, 2, 3)
    assert tuple(array[0, 0]) == (128, 128, 128)


def test_load_rgb_array_converts_palette() -> None:
    image = Image.new("P", (1, 1))
    image.putpalette([255, 0, 0] + [0] * (768 - 3))
    array = load_rgb_array(image)
    assert tuple(array[0, 0]) == (255, 0, 0)


def test_load_rgb_array_rejects_unsupported_mode() -> None:
    with pytest.raises(InvalidImage, match="mode"):
        load_rgb_array(Image.new("CMYK", (2, 2)))


def test_load_rgb_array_rejects_oversized_side() -> None:
    with pytest.raises(InvalidImage, match="at most"):
        load_rgb_array(Image.new("RGB", (4001, 1)))


def test_load_rgb_array_rejects_large_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_io, "MAX_FILE_BYTES", 10)
    path = tmp_path / "big.bin"
    path.write_bytes(b"0" * 11)
    with pytest.raises(InvalidImage, match="limit"):
        load_rgb_array(path)


def test_load_rgb_array_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidImage):
        load_rgb_array(tmp_path / "missing.png")


def test_load_rgb_array_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(InvalidImage):
        load_rgb_array(path)


def test_load_rgb_array_rejects_non_uint8_array() -> None:
    with pytest.raises(InvalidImage, match="uint8"):
        load_rgb_array(np.zeros((2, 2, 3), dtype=np.float32))


def test_load_rgb_array_rejects_wrong_shape() -> None:
    with pytest.raises(InvalidImage, match="shape"):
        load_rgb_array(np.zeros((2, 2), dtype=np.uint8))


def test_load_rgb_array_rejects_empty_array() -> None:
    with pytest.raises(InvalidImage, match="empty"):
        load_rgb_array(np.zeros((0, 2, 3), dtype=np.uint8))


def test_load_rgb_array_rejects_oversized_array() -> None:
    with pytest.raises(InvalidImage, match="at most"):
        load_rgb_array(np.zeros((4001, 1, 3), dtype=np.uint8))

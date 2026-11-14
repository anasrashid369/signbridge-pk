"""Tests for the app service. They use fakes, so they need no torch or Gradio."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import signbridge.app_service as service
from signbridge.app_config import load_app_settings
from signbridge.inference import ModelNotFound, NoHandDetected, Prediction


class FakePredictor:
    """Predictor that records its calls and returns fixed predictions."""

    name = "fake"

    def __init__(self, predictions: list[Prediction] | None = None) -> None:
        self.predictions = predictions or [Prediction("A", 0.8), Prediction("B", 0.15)]
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        self.calls.append((image.shape, top_k))
        return self.predictions[:top_k]


class LandmarkFake(FakePredictor):
    """Landmark predictor that also returns the raw points."""

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        return np.zeros((21, 3), dtype=np.float32)


@pytest.fixture(autouse=True)
def clear_predictors() -> None:
    service.clear_cache()
    yield
    service.clear_cache()


def test_no_image() -> None:
    result = service.predict_image(None, "cnn", 3, load_app_settings())
    assert "Upload an image" in result.status
    assert result.rows == []
    assert result.overlay is None


def test_invalid_image() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float32)
    result = service.predict_image(image, "cnn", 3, load_app_settings())
    assert "Invalid image" in result.status
    assert result.rows == []


def test_cnn_result(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = FakePredictor()
    monkeypatch.setattr(service, "get_predictor", lambda name, settings: predictor)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = service.predict_image(image, "cnn", 3, load_app_settings())
    assert result.status == "**cnn** predicts **A** (80.0%)"
    assert result.rows == [[1, "A", 0.8], [2, "B", 0.15]]
    assert result.overlay is None
    assert predictor.calls == [((16, 16, 3), 3)]


def test_slider_float_is_cast_to_int(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = FakePredictor()
    monkeypatch.setattr(service, "get_predictor", lambda name, settings: predictor)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    service.predict_image(image, "cnn", 1.0, load_app_settings())
    assert predictor.calls[0][1] == 1


def test_unavailable_model_becomes_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing(FakePredictor):
        def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
            raise ModelNotFound("weights missing")

    monkeypatch.setattr(service, "get_predictor", lambda name, settings: Failing())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert "is not available" in result.status
    assert "weights missing" in result.status
    assert result.rows == []


def test_no_hand_becomes_its_own_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoHand(FakePredictor):
        def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
            raise NoHandDetected("No hand was found in the image.")

    monkeypatch.setattr(service, "get_predictor", lambda name, settings: NoHand())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert result.status == (
        "**No hand detected:** the landmark model needs a visible hand. "
        "Switch to the CNN, which does not need one."
    )
    assert result.rows == []


def test_predictor_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    def builder(settings):
        built.append("cnn")
        return FakePredictor()

    monkeypatch.setitem(service.BUILDERS, "cnn", builder)
    settings = load_app_settings()
    first = service.get_predictor("cnn", settings)
    second = service.get_predictor("cnn", settings)
    assert first is second
    assert built == ["cnn"]


def test_landmark_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    overlay_module = types.ModuleType("signbridge.inference.overlay")
    overlay_module.draw_landmarks = lambda image, points: np.full_like(image, 7)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "signbridge.inference.overlay", overlay_module)
    monkeypatch.setattr(
        service, "get_predictor", lambda name, settings: LandmarkFake()
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert result.overlay is not None
    assert (result.overlay == 7).all()


def test_landmark_without_overlay_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "signbridge.inference.overlay", None)
    monkeypatch.setattr(service, "get_predictor", lambda name, settings: LandmarkFake())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert result.overlay is None


def test_landmark_without_detection_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "get_predictor", lambda name, settings: FakePredictor())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert result.overlay is None


def test_landmark_without_a_hand_has_no_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoHand(LandmarkFake):
        def detect_landmarks(self, image: np.ndarray) -> None:
            return None

    monkeypatch.setattr(service, "get_predictor", lambda name, settings: NoHand())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_image(image, "landmark", 3, load_app_settings())
    assert result.overlay is None


def test_both_models_are_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    cnn = FakePredictor()
    landmark = FakePredictor()
    monkeypatch.setattr(
        service,
        "get_predictor",
        lambda name, settings: cnn if name == "cnn" else landmark,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_both(image, 2, load_app_settings())
    assert "**cnn** predicts **A** (80.0%)" in result.status
    assert "**landmark** predicts **A** (80.0%)" in result.status
    assert result.rows == [
        ["cnn", 1, "A", 0.8],
        ["cnn", 2, "B", 0.15],
        ["landmark", 1, "A", 0.8],
        ["landmark", 2, "B", 0.15],
    ]
    assert cnn.calls == [((8, 8, 3), 2)]
    assert landmark.calls == [((8, 8, 3), 2)]


def test_both_keeps_the_cnn_when_the_landmark_finds_no_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoHand(FakePredictor):
        def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
            raise NoHandDetected("No hand was found in the image.")

    monkeypatch.setattr(
        service,
        "get_predictor",
        lambda name, settings: FakePredictor() if name == "cnn" else NoHand(),
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = service.predict_both(image, 2, load_app_settings())
    assert "**cnn** predicts **A**" in result.status
    assert "No hand detected" in result.status
    assert result.rows == [["cnn", 1, "A", 0.8], ["cnn", 2, "B", 0.15]]


def test_both_without_an_image() -> None:
    result = service.predict_both(None, 3, load_app_settings())
    assert "Upload an image" in result.status
    assert result.rows == []
    assert result.overlay is None

"""Tests for the predict CLI. They use fakes, so they do not need torch."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import signbridge.predict as predict_cli
from signbridge.app_config import load_app_settings
from signbridge.inference import (
    ModelNotFound,
    NoHandDetected,
    Prediction,
    PredictorUnavailable,
)


class FakePredictor:
    """Predictor that records its calls and returns fixed predictions."""

    name = "fake"

    def __init__(self, predictions: list[Prediction] | None = None) -> None:
        self.predictions = predictions or [
            Prediction("A", 0.8),
            Prediction("B", 0.15),
            Prediction("C", 0.05),
        ]
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        self.calls.append((image.shape, top_k))
        return self.predictions[:top_k]


class FailingPredictor:
    """Predictor that raises the given error."""

    name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        raise self.error


def make_image(path: Path) -> Path:
    """Save a small RGB image and return its path."""
    Image.fromarray(np.full((24, 32, 3), 128, dtype=np.uint8)).save(path)
    return path


def test_format_predictions() -> None:
    text = predict_cli.format_predictions(
        "cnn", Path("hand.jpg"), [Prediction("A", 0.81234), Prediction("B", 0.1)]
    )
    lines = text.splitlines()
    assert lines[0] == "cnn: hand.jpg"
    assert lines[1].split() == ["rank", "letter", "probability"]
    assert lines[2].split() == ["1", "A", "0.8123"]
    assert lines[3].split() == ["2", "B", "0.1000"]


def test_default_model_is_cnn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    image = make_image(tmp_path / "hand.png")
    predictor = FakePredictor()
    monkeypatch.setattr(predict_cli, "build_cnn_predictor", lambda settings: predictor)
    assert predict_cli.main(["--image", str(image)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "cnn:" in captured.out
    assert "0.8000" in captured.out
    assert predictor.calls == [((24, 32, 3), 3)]


def test_top_k_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    image = make_image(tmp_path / "hand.png")
    predictor = FakePredictor()
    monkeypatch.setattr(predict_cli, "build_cnn_predictor", lambda settings: predictor)
    assert predict_cli.main(["--image", str(image), "--top-k", "2"]) == 0
    assert predictor.calls[0][1] == 2
    assert len(capsys.readouterr().out.splitlines()) == 4


@pytest.mark.parametrize("top_k", ["0", "27"])
def test_rejects_bad_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys, top_k: str
) -> None:
    image = make_image(tmp_path / "hand.png")
    monkeypatch.setattr(predict_cli, "build_cnn_predictor", lambda settings: FakePredictor())
    assert predict_cli.main(["--image", str(image), "--top-k", top_k]) == 1
    assert "--top-k" in capsys.readouterr().err


def test_missing_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(predict_cli, "build_cnn_predictor", lambda settings: FakePredictor())
    assert predict_cli.main(["--image", str(tmp_path / "missing.png")]) == 1
    assert "error:" in capsys.readouterr().err


def test_both_runs_cnn_and_reports_missing_landmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    image = make_image(tmp_path / "hand.png")
    monkeypatch.setattr(predict_cli, "build_cnn_predictor", lambda settings: FakePredictor())

    def unavailable(settings):
        raise PredictorUnavailable("module missing")

    monkeypatch.setattr(predict_cli, "build_landmark_predictor", unavailable)
    assert predict_cli.main(["--image", str(image), "--model", "both"]) == 1
    captured = capsys.readouterr()
    assert "cnn:" in captured.out
    assert "landmark: error: module missing" in captured.err


def test_landmark_without_a_hand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    image = make_image(tmp_path / "hand.png")
    monkeypatch.setattr(
        predict_cli,
        "build_landmark_predictor",
        lambda settings: FailingPredictor(NoHandDetected("No hand detected.")),
    )
    assert predict_cli.main(["--image", str(image), "--model", "landmark"]) == 1
    assert "no hand detected" in capsys.readouterr().err.lower()


def test_missing_cnn_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    image = make_image(tmp_path / "hand.png")
    monkeypatch.setattr(
        predict_cli,
        "build_cnn_predictor",
        lambda settings: FailingPredictor(ModelNotFound("checkpoint missing")),
    )
    assert predict_cli.main(["--image", str(image)]) == 1
    assert "checkpoint missing" in capsys.readouterr().err


def test_missing_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    image = make_image(tmp_path / "hand.png")
    assert predict_cli.main(["--image", str(image), "--config", str(tmp_path / "no.yaml")]) == 1
    assert "error:" in capsys.readouterr().err


def test_landmark_builder_passes_only_declared_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("signbridge.inference.landmark_predictor")
    received: dict[str, object] = {}

    class LandmarkPredictor:
        def __init__(self, model_dir: Path, label_map_path: Path) -> None:
            received["model_dir"] = model_dir
            received["label_map_path"] = label_map_path

    module.LandmarkPredictor = LandmarkPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "signbridge.inference.landmark_predictor", module)
    settings = load_app_settings()
    predictor = predict_cli.build_landmark_predictor(settings)
    assert isinstance(predictor, LandmarkPredictor)
    assert received == {
        "model_dir": settings.landmark_model_dir,
        "label_map_path": settings.label_map,
    }


def test_landmark_builder_reports_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "signbridge.inference.landmark_predictor", None)
    with pytest.raises(PredictorUnavailable, match="not available"):
        predict_cli.build_landmark_predictor(load_app_settings())


def test_landmark_builder_reports_a_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("signbridge.inference.landmark_predictor")

    class LandmarkPredictor:
        def __init__(self) -> None:
            raise OSError("hand model missing")

    module.LandmarkPredictor = LandmarkPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "signbridge.inference.landmark_predictor", module)
    with pytest.raises(PredictorUnavailable, match="hand model missing"):
        predict_cli.build_landmark_predictor(load_app_settings())

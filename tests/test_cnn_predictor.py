"""Tests for the CNN predictor. They need torch and are skipped without it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signbridge.inference import InvalidImage, ModelNotFound, Prediction  # noqa: E402
from signbridge.inference.cnn_predictor import CNNPredictor  # noqa: E402
from signbridge.models.cnn import build_mobilenet_v2  # noqa: E402

NUM_CLASSES = 26


@pytest.fixture(scope="module")
def label_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write an A-Z label map once for every test in this module."""
    path = tmp_path_factory.mktemp("labels") / "label_map.json"
    mapping = {chr(ord("A") + index): index for index in range(NUM_CLASSES)}
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a small random checkpoint with the real architecture."""
    model = build_mobilenet_v2(
        num_classes=NUM_CLASSES, dropout=0.2, pretrained=False, unfreeze_blocks=0
    )
    path = tmp_path_factory.mktemp("checkpoint") / "cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_macro_f1": 0.0,
            "val_acc": 0.0,
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def predictor(checkpoint: Path, label_map: Path) -> CNNPredictor:
    """Build one predictor and reuse it, because loading MobileNetV2 is slow."""
    return CNNPredictor(
        checkpoint_path=checkpoint, label_map_path=label_map, image_size=64, device="cpu"
    )


@pytest.fixture
def image() -> np.ndarray:
    """Return a deterministic 96x96 RGB test image."""
    return np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)


def test_name(predictor: CNNPredictor) -> None:
    assert predictor.name == "cnn"


def test_predict_returns_sorted_top_k(predictor: CNNPredictor, image: np.ndarray) -> None:
    predictions = predictor.predict(image)
    assert len(predictions) == 3
    assert all(isinstance(prediction, Prediction) for prediction in predictions)
    assert all(prediction.label in predictor.labels for prediction in predictions)
    probabilities = [prediction.probability for prediction in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_full_distribution_sums_to_one(predictor: CNNPredictor, image: np.ndarray) -> None:
    predictions = predictor.predict(image, top_k=NUM_CLASSES)
    assert len(predictions) == NUM_CLASSES
    assert {prediction.label for prediction in predictions} == set(predictor.labels)
    assert sum(prediction.probability for prediction in predictions) == pytest.approx(1.0, abs=1e-5)


def test_predict_is_deterministic(predictor: CNNPredictor, image: np.ndarray) -> None:
    assert predictor.predict(image) == predictor.predict(image)


def test_top_k_is_respected(predictor: CNNPredictor, image: np.ndarray) -> None:
    assert len(predictor.predict(image, top_k=1)) == 1
    assert len(predictor.predict(image, top_k=10)) == 10


@pytest.mark.parametrize("top_k", [0, -1, NUM_CLASSES + 1])
def test_rejects_top_k_out_of_range(
    predictor: CNNPredictor, image: np.ndarray, top_k: int
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        predictor.predict(image, top_k=top_k)


def test_rejects_non_uint8_image(predictor: CNNPredictor) -> None:
    with pytest.raises(InvalidImage, match="uint8"):
        predictor.predict(np.zeros((32, 32, 3), dtype=np.float32))


def test_rejects_wrong_shape(predictor: CNNPredictor) -> None:
    with pytest.raises(InvalidImage, match="shape"):
        predictor.predict(np.zeros((32, 32), dtype=np.uint8))


def test_missing_checkpoint_mentions_the_readme(label_map: Path, tmp_path: Path) -> None:
    with pytest.raises(ModelNotFound, match="README.md"):
        CNNPredictor(checkpoint_path=tmp_path / "missing.pt", label_map_path=label_map)


def test_checkpoint_without_state_dict(label_map: Path, tmp_path: Path) -> None:
    path = tmp_path / "empty.pt"
    torch.save({"epoch": 1}, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        CNNPredictor(checkpoint_path=path, label_map_path=label_map)


def test_checkpoint_with_wrong_class_count(label_map: Path, tmp_path: Path) -> None:
    model = build_mobilenet_v2(num_classes=2, pretrained=False, unfreeze_blocks=0)
    path = tmp_path / "two.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="26 classes"):
        CNNPredictor(checkpoint_path=path, label_map_path=label_map)

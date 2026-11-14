"""Tests for the landmark predictor. They use a fake MediaPipe landmarker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from data_scripts.features.landmarks import (  # noqa: E402
    normalize_landmarks as training_normalize,
)
from signbridge.app_config import load_app_settings  # noqa: E402
from signbridge.inference import (  # noqa: E402
    InvalidImage,
    ModelNotFound,
    NoHandDetected,
    Prediction,
)
from signbridge.inference import landmark_predictor as module  # noqa: E402
from signbridge.inference.landmark_predictor import (  # noqa: E402
    LandmarkPredictor,
    normalize_landmarks,
)

NUM_CLASSES = 26


class FakeLandmarker:
    """Stands in for MediaPipe: returns fixed landmarks and records its input."""

    def __init__(self, points: np.ndarray | None) -> None:
        self.points = points
        self.seen: list[np.ndarray] = []
        self.closed = False

    def detect(self, mp_image):
        self.seen.append(mp_image.numpy_view().copy())
        if self.points is None:
            return SimpleNamespace(hand_landmarks=[])
        hand = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in self.points]
        return SimpleNamespace(hand_landmarks=[hand])

    def close(self) -> None:
        self.closed = True


class OneHotModel:
    """Fake classifier: 70% for class 5 (F), 30% for class 2 (C)."""

    classes_ = np.arange(NUM_CLASSES)

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        self.seen.append(rows)
        out = np.zeros((1, NUM_CLASSES))
        out[0, 5], out[0, 2] = 0.7, 0.3
        return out


@pytest.fixture(scope="module")
def points() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(21, 3))


@pytest.fixture(scope="module")
def label_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("labels") / "label_map.json"
    mapping = {chr(ord("A") + index): index for index in range(NUM_CLASSES)}
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a small random scaler + classifier pipeline with classes 0..25."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(2 * NUM_CLASSES, 63))
    labels = np.tile(np.arange(NUM_CLASSES), 2)
    pipeline = Pipeline([("scale", StandardScaler()), ("svm", LogisticRegression(max_iter=500))])
    pipeline.fit(features, labels)
    folder = tmp_path_factory.mktemp("landmark")
    joblib.dump(pipeline, folder / "landmark_svm.pkl")
    return folder


@pytest.fixture
def hand_model(tmp_path: Path) -> Path:
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(b"fake")  # the real landmarker is replaced by FakeLandmarker
    return path


def build(monkeypatch, model_dir, label_map, hand_model, fake) -> LandmarkPredictor:
    monkeypatch.setattr(module, "create_landmarker", lambda path: fake)
    return LandmarkPredictor(
        model_dir=model_dir, label_map_path=label_map, hand_model=hand_model
    )


@pytest.fixture
def fake(points: np.ndarray) -> FakeLandmarker:
    return FakeLandmarker(points)


@pytest.fixture
def predictor(monkeypatch, model_dir, label_map, hand_model, fake) -> LandmarkPredictor:
    return build(monkeypatch, model_dir, label_map, hand_model, fake)


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(2).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


# --- normalisation ------------------------------------------------------


def test_normalize_matches_training_code(points: np.ndarray) -> None:
    assert np.array_equal(normalize_landmarks(points), training_normalize(points))


def test_normalize_properties(points: np.ndarray) -> None:
    features = normalize_landmarks(points)
    assert features.shape == (63,)
    assert np.allclose(features[:3], 0.0)  # the wrist is the origin
    moved = normalize_landmarks(points + np.array([0.3, -0.2, 0.1]))
    scaled = normalize_landmarks(points * 2.5)
    assert np.allclose(moved, features)  # position does not matter
    assert np.allclose(scaled, features)  # size does not matter


def test_normalize_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_landmarks(np.zeros((20, 3)))
    with pytest.raises(ValueError, match="zero"):
        normalize_landmarks(np.zeros((21, 3)))


# --- predict ------------------------------------------------------------


def test_name(predictor: LandmarkPredictor) -> None:
    assert predictor.name == "landmark"


def test_predict_returns_sorted_top_k(predictor: LandmarkPredictor, image: np.ndarray) -> None:
    predictions = predictor.predict(image)
    assert len(predictions) == 3
    assert all(isinstance(item, Prediction) for item in predictions)
    probabilities = [item.probability for item in predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_full_distribution_sums_to_one(predictor: LandmarkPredictor, image: np.ndarray) -> None:
    predictions = predictor.predict(image, top_k=NUM_CLASSES)
    assert {item.label for item in predictions} == set(predictor.labels)
    assert sum(item.probability for item in predictions) == pytest.approx(1.0, abs=1e-6)


def test_labels_follow_class_index(predictor: LandmarkPredictor, image: np.ndarray) -> None:
    predictor._model = OneHotModel()
    assert predictor.predict(image, top_k=2) == [Prediction("F", 0.7), Prediction("C", 0.3)]


def test_model_receives_training_features(
    predictor: LandmarkPredictor, image: np.ndarray, points: np.ndarray
) -> None:
    model = OneHotModel()
    predictor._model = model
    predictor.predict(image)
    expected = training_normalize(points).astype(np.float32).reshape(1, -1)
    assert model.seen[0].dtype == np.float32
    assert np.array_equal(model.seen[0], expected)


def test_image_reaches_mediapipe_unchanged(
    predictor: LandmarkPredictor, fake: FakeLandmarker, image: np.ndarray
) -> None:
    predictor.predict(image)
    assert np.array_equal(fake.seen[0], image)  # no colour conversion inside


def test_detect_landmarks_returns_raw_points(
    predictor: LandmarkPredictor, image: np.ndarray, points: np.ndarray
) -> None:
    assert np.allclose(predictor.detect_landmarks(image), points)


# --- no hand ------------------------------------------------------------


def test_no_hand_raises(monkeypatch, model_dir, label_map, hand_model, image) -> None:
    predictor = build(monkeypatch, model_dir, label_map, hand_model, FakeLandmarker(None))
    assert predictor.detect_landmarks(image) is None
    with pytest.raises(NoHandDetected):
        predictor.predict(image)


def test_non_finite_landmarks_count_as_no_hand(
    monkeypatch, model_dir, label_map, hand_model, image, points
) -> None:
    broken = points.copy()
    broken[3, 0] = np.nan
    predictor = build(monkeypatch, model_dir, label_map, hand_model, FakeLandmarker(broken))
    with pytest.raises(NoHandDetected):
        predictor.predict(image)


def test_zero_size_hand_counts_as_no_hand(
    monkeypatch, model_dir, label_map, hand_model, image
) -> None:
    flat = FakeLandmarker(np.full((21, 3), 0.5))
    predictor = build(monkeypatch, model_dir, label_map, hand_model, flat)
    with pytest.raises(NoHandDetected):
        predictor.predict(image)


# --- bad input ----------------------------------------------------------


def test_rejects_non_uint8_image(predictor: LandmarkPredictor) -> None:
    with pytest.raises(InvalidImage, match="uint8"):
        predictor.predict(np.zeros((32, 32, 3), dtype=np.float32))


def test_rejects_wrong_shape(predictor: LandmarkPredictor) -> None:
    with pytest.raises(InvalidImage, match="shape"):
        predictor.predict(np.zeros((32, 32), dtype=np.uint8))


@pytest.mark.parametrize("top_k", [0, -1, NUM_CLASSES + 1])
def test_rejects_top_k_out_of_range(
    predictor: LandmarkPredictor, image: np.ndarray, top_k: int
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        predictor.predict(image, top_k=top_k)


# --- loading ------------------------------------------------------------


def test_missing_svm(monkeypatch, label_map, hand_model, tmp_path) -> None:
    monkeypatch.setattr(module, "create_landmarker", lambda path: FakeLandmarker(None))
    with pytest.raises(ModelNotFound, match="landmark_svm.pkl"):
        LandmarkPredictor(model_dir=tmp_path, label_map_path=label_map, hand_model=hand_model)


def test_missing_hand_model(monkeypatch, model_dir, label_map, tmp_path) -> None:
    monkeypatch.setattr(module, "create_landmarker", lambda path: FakeLandmarker(None))
    with pytest.raises(ModelNotFound, match="hand model"):
        LandmarkPredictor(
            model_dir=model_dir, label_map_path=label_map, hand_model=tmp_path / "missing.task"
        )


def test_svm_with_wrong_class_count(monkeypatch, label_map, hand_model, tmp_path) -> None:
    rng = np.random.default_rng(3)
    pipeline = Pipeline([("scale", StandardScaler()), ("svm", LogisticRegression())])
    pipeline.fit(rng.normal(size=(10, 63)), np.array([0, 1] * 5))
    joblib.dump(pipeline, tmp_path / "landmark_svm.pkl")
    monkeypatch.setattr(module, "create_landmarker", lambda path: FakeLandmarker(None))
    with pytest.raises(ValueError, match="26"):
        LandmarkPredictor(model_dir=tmp_path, label_map_path=label_map, hand_model=hand_model)


def test_close_releases_landmarker(predictor: LandmarkPredictor, fake: FakeLandmarker) -> None:
    with predictor:
        pass
    assert fake.closed


# --- real models (skipped when they are not downloaded) -------------------

def _real_models_present() -> bool:
    settings = load_app_settings()
    svm = settings.landmark_model_dir / "landmark_svm.pkl"
    return svm.is_file() and settings.hand_model.is_file()

@pytest.mark.skipif(not _real_models_present(), reason="models are not in this checkout")
def test_real_models_report_no_hand_on_blank_image() -> None:
    with LandmarkPredictor() as predictor:  # paths come from configs/app.yaml
        with pytest.raises(NoHandDetected):
            predictor.predict(np.full((224, 224, 3), 255, dtype=np.uint8))

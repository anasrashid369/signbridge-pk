"""Hand-landmark predictor: MediaPipe, 63 normalised features, calibrated SVM."""

from __future__ import annotations

import threading
from pathlib import Path
from types import TracebackType
from typing import Any

import joblib
import mediapipe as mp
import numpy as np

from signbridge.app_config import load_app_settings
from signbridge.inference.errors import ModelNotFound, NoHandDetected
from signbridge.inference.image_io import load_rgb_array
from signbridge.inference.types import Prediction, load_labels

SVM_FILE = "landmark_svm.pkl"
NUM_LANDMARKS = 21  # MediaPipe gives 21 points per hand
NUM_COORDS = 3  # each point has x, y, z
WRIST_INDEX = 0
MIDDLE_FINGER_BASE_INDEX = 9
MIN_HAND_SIZE = 1e-8


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Return the 63 features the SVM was trained on.

    This is a copy of ``data_scripts/features/landmarks.py``; a test checks
    that both give identical output. Step 1 subtracts the wrist, so position
    no longer matters. Step 2 divides by the wrist-to-middle-finger-base
    distance, so hand size no longer matters.

    Args:
        landmarks: Raw MediaPipe landmarks, shape (21, 3).

    Returns:
        A flat float64 array of 63 numbers.

    Raises:
        ValueError: If the shape is wrong or the hand size is zero.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(f"Expected shape {(NUM_LANDMARKS, NUM_COORDS)}, got {points.shape}")
    shifted = points - points[WRIST_INDEX]
    hand_size = float(np.linalg.norm(shifted[MIDDLE_FINGER_BASE_INDEX]))
    if hand_size < MIN_HAND_SIZE:
        raise ValueError("Hand size is zero; cannot normalize this hand.")
    return (shifted / hand_size).flatten()


def create_landmarker(model_path: Path) -> Any:
    """Create a MediaPipe hand landmarker in image mode.

    The settings match ``data_scripts/features/extract.py`` (image mode, one
    hand, default thresholds), so inference sees the hands the way training did.
    """
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


class LandmarkPredictor:
    """MediaPipe hand landmarks followed by the Deliverable 2 SVM.

    Both models are loaded once and reused for every image. The input must be
    an RGB array: this class never converts colours. OpenCV frames are BGR and
    must be converted by the caller first.

    Attributes:
        name: ``"landmark"``, so the app can label the result.
        labels: Class names in model output order.
    """

    name = "landmark"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        label_map_path: str | Path | None = None,
        hand_model: str | Path | None = None,
    ) -> None:
        """Load the labels, the SVM and the MediaPipe hand model.

        Any argument left as None is read from ``configs/app.yaml``.

        Args:
            model_dir: Folder containing ``landmark_svm.pkl``.
            label_map_path: ``label_map.json``; its indices must be 0..25.
            hand_model: MediaPipe ``hand_landmarker.task`` file.

        Raises:
            ModelNotFound: If the SVM file or the hand model file is missing.
            FileNotFoundError: If the label map is missing.
            ValueError: If the SVM cannot give probabilities or its classes
                do not match the label map.
        """
        if model_dir is None or label_map_path is None or hand_model is None:
            settings = load_app_settings()
            model_dir = settings.landmark_model_dir if model_dir is None else model_dir
            label_map_path = settings.label_map if label_map_path is None else label_map_path
            hand_model = settings.hand_model if hand_model is None else hand_model
        self.model_dir = Path(model_dir)
        self.hand_model = Path(hand_model)
        self.labels = load_labels(label_map_path)
        self._model = self._load_svm(self.model_dir / SVM_FILE)
        if not self.hand_model.is_file():
            raise ModelNotFound(
                f"MediaPipe hand model not found: {self.hand_model}\n"
                "Download it as described in README.md."
            )
        self._lock = threading.Lock()  # one MediaPipe call at a time
        self._landmarker = create_landmarker(self.hand_model)

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray | None:
        """Return the raw landmarks of the hand in an RGB image, or None.

        Args:
            image: uint8 RGB array of shape (height, width, 3).

        Returns:
            Array of shape (21, 3) with x and y as fractions of the image
            width and height, or None if no usable hand was found.

        Raises:
            InvalidImage: If ``image`` is not a uint8 RGB array within limits.
        """
        array = load_rgb_array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        raw = np.array(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float64,
        )
        return raw if np.all(np.isfinite(raw)) else None

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        """Return the ``top_k`` most likely letters for one RGB image.

        Args:
            image: uint8 RGB array of shape (height, width, 3).
            top_k: Number of predictions to return, between 1 and 26.

        Returns:
            Predictions sorted from highest to lowest probability. The
            probabilities cover all 26 classes and sum to 1 before truncation.

        Raises:
            InvalidImage: If ``image`` is not a uint8 RGB array within limits.
            NoHandDetected: If MediaPipe finds no usable hand.
            ValueError: If ``top_k`` is outside 1..26.
        """
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(f"top_k must be between 1 and {len(self.labels)}, got {top_k}")
        raw = self.detect_landmarks(image)
        if raw is None:
            raise NoHandDetected("No hand was found in the image.")
        try:
            features = normalize_landmarks(raw)
        except ValueError as error:
            raise NoHandDetected("The detected hand has no usable size.") from error
        # Training cached the features as float32, so the SVM sees float32 here too.
        row = features.astype(np.float32).reshape(1, -1)
        probabilities = self._model.predict_proba(row)[0]
        order = np.argsort(-probabilities, kind="stable")[:top_k]
        return [
            Prediction(label=self.labels[index], probability=float(probabilities[index]))
            for index in order
        ]

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        self._landmarker.close()

    def __enter__(self) -> LandmarkPredictor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _load_svm(self, path: Path) -> Any:
        """Load the pickled pipeline and check it matches the label map."""
        if not path.is_file():
            raise ModelNotFound(
                f"Landmark SVM not found: {path}\n"
                "Download models/landmark/ as described in README.md "
                "(section 'Model weights') or train it with "
                "'python -m signbridge.train_landmark'."
            )
        model = joblib.load(path)  # only load files from our own training run
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"The model cannot give probabilities: {path}")
        if list(model.classes_) != list(range(len(self.labels))):
            raise ValueError(
                f"The SVM has {len(model.classes_)} classes but the label map has "
                f"{len(self.labels)} (indices 0..n-1 expected): {path}"
            )
        return model

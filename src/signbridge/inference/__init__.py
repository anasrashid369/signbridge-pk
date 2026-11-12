"""Shared inference interface for the app, the CLI and the tests."""

from signbridge.inference.errors import (
    InvalidImage,
    ModelNotFound,
    NoHandDetected,
    PredictorUnavailable,
)
from signbridge.inference.image_io import MAX_FILE_BYTES, MAX_SIDE, load_rgb_array
from signbridge.inference.types import Prediction, Predictor, load_labels

__all__ = [
    "InvalidImage",
    "MAX_FILE_BYTES",
    "MAX_SIDE",
    "ModelNotFound",
    "NoHandDetected",
    "Prediction",
    "Predictor",
    "PredictorUnavailable",
    "load_labels",
    "load_rgb_array",
]

"""Glue between the Gradio app and the predictors.

The service keeps one predictor per model alive between clicks, so the
checkpoint is loaded once rather than on every request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from signbridge.app_config import AppSettings
from signbridge.inference import (
    InvalidImage,
    ModelNotFound,
    NoHandDetected,
    Predictor,
    PredictorUnavailable,
    load_rgb_array,
)
from signbridge.predict import build_cnn_predictor, build_landmark_predictor

BUILDERS = {"cnn": build_cnn_predictor, "landmark": build_landmark_predictor}
_PREDICTORS: dict[str, Predictor] = {}


@dataclass(frozen=True)
class PredictionResult:
    """What the app shows after one click.

    Attributes:
        status: Markdown line with the top letter or a readable error.
        rows: One ``[rank, letter, probability]`` row per prediction.
        overlay: RGB image with the hand skeleton, or None when unavailable.
    """

    status: str
    rows: list[list[object]]
    overlay: np.ndarray | None = None


def get_predictor(name: str, settings: AppSettings) -> Predictor:
    """Return the cached predictor for ``name``, building it on first use."""
    if name not in _PREDICTORS:
        _PREDICTORS[name] = BUILDERS[name](settings)
    return _PREDICTORS[name]


def clear_cache() -> None:
    """Forget the cached predictors, for example after replacing a checkpoint."""
    _PREDICTORS.clear()


def predict_image(
    image: np.ndarray | None,
    model_name: str,
    top_k: float,
    settings: AppSettings,
) -> PredictionResult:
    """Predict one image and turn every failure into a readable message.

    Args:
        image: RGB array from Gradio, or None when nothing was uploaded.
        model_name: ``"cnn"`` or ``"landmark"``.
        top_k: Number of rows to show; Gradio's slider gives a float.
        settings: App settings with the paths and limits.

    Returns:
        A status line, the table rows and an optional landmark overlay.
    """
    if image is None:
        return PredictionResult("Upload an image or take a webcam photo first.", [])
    try:
        array = load_rgb_array(
            image, max_bytes=settings.max_file_bytes, max_side=settings.max_side
        )
    except InvalidImage as error:
        return PredictionResult(f"**Invalid image:** {error}", [])
    try:
        predictor = get_predictor(model_name, settings)
        predictions = predictor.predict(array, top_k=int(top_k))
    except (ModelNotFound, NoHandDetected, PredictorUnavailable, ValueError) as error:
        return PredictionResult(f"**{model_name} is not available:** {error}", [])

    top = predictions[0]
    status = f"**{model_name}** predicts **{top.label}** ({top.probability:.1%})"
    rows = [
        [rank, prediction.label, round(prediction.probability, 4)]
        for rank, prediction in enumerate(predictions, start=1)
    ]
    overlay = _overlay(predictor, array) if model_name == "landmark" else None
    return PredictionResult(status, rows, overlay)


def _overlay(predictor: Predictor, image: np.ndarray) -> np.ndarray | None:
    """Draw the hand skeleton when the landmark module offers it."""
    landmarks = getattr(predictor, "landmarks", None)
    if landmarks is None:
        return None
    try:
        from signbridge.inference.overlay import draw_landmarks
    except ImportError:
        return None
    points = landmarks(image)
    if points is None:
        return None
    return draw_landmarks(image, points)

"""Predict the letter in one image from the command line.

Examples:
    python -m signbridge.predict --image path/to/hand.jpg
    python -m signbridge.predict --image path/to/hand.jpg --model both --top-k 5

The image is read with the same helper as the app, so the size and mode limits
are identical. The CNN model loads from ``configs/app.yaml``; the landmark
model is imported only when it is requested.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from signbridge.app_config import (
    APP_CONFIG_PATH,
    MAX_CLASSES,
    MODEL_CHOICES,
    AppSettings,
    load_app_settings,
)
from signbridge.inference import (
    InvalidImage,
    ModelNotFound,
    NoHandDetected,
    Prediction,
    Predictor,
    PredictorUnavailable,
    load_rgb_array,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m signbridge.predict",
        description="Predict the ASL letter in one image.",
    )
    parser.add_argument("--image", type=Path, required=True, help="path to the image file")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="predictor to run; defaults to model.default in configs/app.yaml",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="predictions to show per model; defaults to model.top_k in configs/app.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APP_CONFIG_PATH,
        help="app settings file; defaults to configs/app.yaml",
    )
    return parser


def format_predictions(name: str, image_path: Path, predictions: list[Prediction]) -> str:
    """Return a fixed-width table, one row per prediction."""
    lines = [
        f"{name}: {image_path}",
        f"{'rank':>4}  {'letter':<6}  {'probability':>11}",
    ]
    for rank, prediction in enumerate(predictions, start=1):
        lines.append(f"{rank:>4}  {prediction.label:<6}  {prediction.probability:>11.4f}")
    return "\n".join(lines)


def build_cnn_predictor(settings: AppSettings) -> Predictor:
    """Build the CNN predictor from the app settings.

    The import is inside the function so the CLI starts without torch until a
    model is actually requested.
    """
    from signbridge.inference.cnn_predictor import CNNPredictor

    return CNNPredictor(
        checkpoint_path=settings.cnn_checkpoint,
        label_map_path=settings.label_map,
        image_size=settings.image_size,
        dropout=settings.dropout,
    )


def build_landmark_predictor(settings: AppSettings) -> Predictor:
    """Build the landmark predictor, which lives in a teammate's module.

    Only the arguments that the class declares are passed, so the module can
    choose which of the three settings it needs.

    Raises:
        PredictorUnavailable: If the module or the class cannot be imported.
    """
    try:
        from signbridge.inference.landmark_predictor import LandmarkPredictor
    except ImportError as error:
        raise PredictorUnavailable(
            "the landmark predictor is not available in this checkout "
            f"(signbridge.inference.landmark_predictor): {error}"
        ) from error

    candidates = {
        "model_dir": settings.landmark_model_dir,
        "label_map_path": settings.label_map,
        "hand_model": settings.hand_model,
    }
    accepted = inspect.signature(LandmarkPredictor).parameters
    arguments = {key: value for key, value in candidates.items() if key in accepted}
    try:
        return LandmarkPredictor(**arguments)
    except (OSError, ValueError, RuntimeError) as error:
        raise PredictorUnavailable(f"the landmark predictor could not start: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code (0 ok, 1 error)."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_app_settings(args.config)
    except (FileNotFoundError, ValueError) as error:
        return _fail(error)

    model = args.model or settings.default_model
    top_k = settings.top_k if args.top_k is None else args.top_k
    if not 1 <= top_k <= MAX_CLASSES:
        return _fail(ValueError(f"--top-k must be between 1 and {MAX_CLASSES}, got {top_k}"))
    try:
        image = load_rgb_array(
            args.image, max_bytes=settings.max_file_bytes, max_side=settings.max_side
        )
    except InvalidImage as error:
        return _fail(error)

    builders = {"cnn": build_cnn_predictor, "landmark": build_landmark_predictor}
    names = ["cnn", "landmark"] if model == "both" else [model]
    status = 0
    printed = False
    for name in names:
        try:
            predictions = builders[name](settings).predict(image, top_k=top_k)
        except NoHandDetected as error:
            print(
                f"{name}: no hand detected: {error} "
                "The landmark model needs a visible hand; use --model cnn (or both).",
                file=sys.stderr,
                flush=True,
            )
            status = 1
            continue
        except (InvalidImage, ModelNotFound, PredictorUnavailable) as error:
            print(f"{name}: error: {error}", file=sys.stderr, flush=True)
            status = 1
            continue
        if printed:
            print()
        print(format_predictions(name, args.image, predictions), flush=True)
        printed = True
    return status


def _fail(error: Exception) -> int:
    """Print one error line to stderr and return the failure exit code."""
    print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

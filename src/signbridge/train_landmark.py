"""Train the landmark models (SVM, Random Forest, MLP) and write prediction files.

Run from the repository root:
    python -m signbridge.train_landmark --config configs/landmark.yaml

By default the files go to the scratch folder (output.runs_dir), which Git ignores.
Add --final to write the validation files to output.predictions_dir (committed).
Add --evaluate-test, together with --final, to also write the test files. Do that once,
after the hyperparameters are frozen.

The features come from `python -m data_scripts.features.extract --split-file ...`.
Only images where a hand was found are in that file, so predictions cover those images.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from data_scripts.features.extract import SPLIT_FEATURES_FILE
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging
from signbridge.models.landmark_models import (
    LandmarkMLP,
    build_forest,
    build_svm,
    fit_mlp,
    full_proba,
    predict_proba_mlp,
    write_history_csv,
)
from signbridge.training.loop import set_seed
from signbridge.training.predictions import save_predictions

logger = logging.getLogger(__name__)

LANDMARK_CONFIG_PATH = REPO_ROOT / "configs" / "landmark.yaml"
MODEL_NAMES = ("svm", "forest", "mlp")
MODEL_FILES = {
    "svm": "landmark_svm.pkl",
    "forest": "landmark_forest.pkl",
    "mlp": "landmark_mlp.pt",
}
CACHE_KEYS = ("X", "y", "paths", "split")


@dataclass
class FittedModel:
    """A trained landmark model with what is needed to predict, save and plot it."""

    name: str
    estimator: object
    n_classes: int
    scaler: StandardScaler | None = None  # only the MLP needs one (the SVM has its own)
    history: list[dict] | None = None  # per-epoch numbers, only for the MLP

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n, n_classes)."""
        if self.name == "mlp":
            return predict_proba_mlp(self.estimator, self.scaler.transform(X))
        return full_proba(self.estimator, X, self.n_classes)

    def save(self, path: Path) -> None:
        """Save the model. The MLP file holds plain lists, so torch can load it safely."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.name == "mlp":
            torch.save(
                {
                    "state_dict": self.estimator.state_dict(),
                    "scaler_mean": self.scaler.mean_.tolist(),
                    "scaler_scale": self.scaler.scale_.tolist(),
                },
                path,
            )
        else:
            joblib.dump(self.estimator, path)


def read_label_map(path: Path) -> dict[str, int]:
    """Read label_map.json (class name to index) and check the indices are 0..n-1."""
    with path.open(encoding="utf-8") as handle:
        label_map = {str(name): int(index) for name, index in json.load(handle).items()}
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError(f"Label map indices must be 0..{len(label_map) - 1}: {path}")
    return label_map


def load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load the split landmark cache (X, y, paths, split).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a key is missing, for example the cache has no split column.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Landmark cache not found: {path}. Run: "
            "python -m data_scripts.features.extract --split-file data/processed/split_v1.csv"
        )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in CACHE_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"Landmark cache is missing keys {missing}: {path}")
        return {key: data[key] for key in CACHE_KEYS}


def select_split(
    cache: dict[str, np.ndarray], name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, paths) of one split. Raises ValueError if the split is empty."""
    mask = cache["split"] == name
    if not mask.any():
        raise ValueError(f"No images with a detected hand in the {name} split")
    return cache["X"][mask], cache["y"][mask], [str(path) for path in cache["paths"][mask]]


def fit_model(
    name: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    input_dim: int,
    n_classes: int,
) -> FittedModel:
    """Train one model with the settings in params (the model's section of landmark.yaml).

    The validation split is used by the MLP for early stopping only, never the test split.
    """
    if name == "svm":
        estimator = build_svm(C=params["C"], gamma=params["gamma"], seed=seed)
        return FittedModel(name, estimator.fit(X_train, y_train), n_classes)
    if name == "forest":
        estimator = build_forest(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            seed=seed,
        )
        return FittedModel(name, estimator.fit(X_train, y_train), n_classes)
    if name == "mlp":
        scaler = StandardScaler().fit(X_train)
        network = LandmarkMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_size=int(params["hidden_size"]),
            n_hidden_layers=int(params["n_hidden_layers"]),
            dropout=float(params["dropout"]),
        )
        history = fit_mlp(
            network,
            scaler.transform(X_train),
            y_train,
            scaler.transform(X_val),
            y_val,
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            patience=int(params["patience"]),
            seed=seed,
        )
        return FittedModel(name, network, n_classes, scaler=scaler, history=history)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def score(model: FittedModel, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (accuracy, macro-F1) of a model on a dataset."""
    predicted = model.predict_proba(X).argmax(axis=1)
    return float((predicted == y).mean()), float(f1_score(y, predicted, average="macro"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the landmark models.")
    parser.add_argument("--config", type=Path, default=LANDMARK_CONFIG_PATH)
    parser.add_argument(
        "--data-config", type=Path, default=DEFAULT_CONFIG_PATH, help="Shared data config."
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Landmark cache. Default: paths.processed_dir in data.yaml.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--final",
        action="store_true",
        help="Write to output.predictions_dir, not the scratch folder.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also write test predictions. Needs --final. Use once, after freezing settings.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Train the chosen models, save them and write the prediction files."""
    if args.evaluate_test and not args.final:
        raise ValueError(
            "--evaluate-test needs --final: test files are written once, to the final folder"
        )
    if args.evaluate_test:
        logger.warning(
            "--evaluate-test is enabled. The test split should only be evaluated "
            "once, after hyperparameters have been frozen."
        )
    data_config = load_config(args.data_config)
    config = load_config(args.config)
    seed = int(data_config["seed"])
    set_seed(seed)
    logger.info("Seed: %d", seed)

    label_map = read_label_map(resolve_path(data_config, "label_map"))
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    if len(class_names) != int(config["n_classes"]):
        raise ValueError("n_classes in the landmark config does not match the label map")
    n_classes, input_dim = len(class_names), int(config["input_dim"])

    processed_dir = resolve_path(data_config, "processed_dir")
    features_file = args.features or processed_dir / SPLIT_FEATURES_FILE
    cache = load_cache(features_file)
    X_train, y_train, _ = select_split(cache, "train")
    X_val, y_val, val_paths = select_split(cache, "val")
    logger.info("Images with a hand: %d train, %d val", len(y_train), len(y_val))
    # The test split is only selected when --evaluate-test is given.
    test_data = select_split(cache, "test") if args.evaluate_test else None

    output = config["output"]
    folder = output["predictions_dir"] if args.final else output["runs_dir"]
    prediction_dir = REPO_ROOT / folder
    model_dir = REPO_ROOT / output["model_dir"]
    if args.final:
        history_file = REPO_ROOT / output["history_csv"]
    else:
        history_file = prediction_dir / "landmark_mlp_history.csv"

    summary: dict[str, dict[str, float]] = {}
    for name in args.models:
        model = fit_model(
            name, config[name], X_train, y_train, X_val, y_val, seed, input_dim, n_classes
        )
        accuracy, macro_f1 = score(model, X_val, y_val)
        summary[name] = {"val_accuracy": accuracy, "val_macro_f1": macro_f1}
        logger.info("%s: validation accuracy %.4f, macro-F1 %.4f", name, accuracy, macro_f1)

        model.save(model_dir / MODEL_FILES[name])
        if model.history:
            write_history_csv(model.history, history_file)
        save_predictions(
            prediction_dir / f"landmark_{name}_val.npz",
            y_val,
            model.predict_proba(X_val),
            val_paths,
            class_names,
        )
        if test_data is not None:
            X_test, y_test, test_paths = test_data
            save_predictions(
                prediction_dir / f"landmark_{name}_test.npz",
                y_test,
                model.predict_proba(X_test),
                test_paths,
                class_names,
            )
    logger.info("Predictions written to %s", prediction_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or file error."""
    args = parse_args(argv)
    setup_logging()
    try:
        run(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

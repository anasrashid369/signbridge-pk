"""Tune the landmark models on the validation split only and log every trial.

Run from the repository root:
    python -m signbridge.tune_landmark --config configs/landmark.yaml

The SVM and the forest use the full grid in configs/landmark.yaml. The MLP grid is too big,
so a seeded random sample of tuning.mlp_max_trials combinations is used. Every trial goes
to tuning.trials_csv. A trial that is already in the file is skipped, so an interrupted run
can be started again. The test split is never selected or scored here.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

import numpy as np

from data_scripts.features.extract import SPLIT_FEATURES_FILE
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging
from signbridge.train_landmark import (
    LANDMARK_CONFIG_PATH,
    MODEL_NAMES,
    fit_model,
    load_cache,
    read_label_map,
    score,
    select_split,
)
from signbridge.training.landmark_trials import append_trial, has_trial
from signbridge.training.loop import set_seed
from signbridge.training.trials import read_trials

logger = logging.getLogger(__name__)

# Tuning keys per model. Each key is also a column of the trials CSV.
GRID_KEYS = {
    "svm": ["svm_C", "svm_gamma"],
    "forest": ["forest_n_estimators", "forest_max_depth"],
    "mlp": [
        "mlp_learning_rate",
        "mlp_batch_size",
        "mlp_dropout",
        "mlp_hidden_size",
        "mlp_weight_decay",
    ],
}


def build_trials(tuning: dict, model: str, seed: int) -> list[dict[str, object]]:
    """Return the parameter sets to try for one model, keyed by trial column name.

    The SVM and forest use every combination. The MLP uses a seeded random sample of
    tuning.mlp_max_trials combinations when the full grid is larger than that.
    """
    keys = GRID_KEYS[model]
    combos = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(tuning[key] for key in keys))
    ]
    if model == "mlp":
        limit = int(tuning["mlp_max_trials"])
        if len(combos) > limit:
            chosen = np.random.default_rng(seed).choice(len(combos), size=limit, replace=False)
            combos = [combos[index] for index in sorted(chosen)]
    return combos


def model_settings(config: dict, model: str, params: dict[str, object]) -> dict:
    """Return the model's section of landmark.yaml with the trial values put on top."""
    prefix = f"{model}_"
    trial_values = {key[len(prefix):]: value for key, value in params.items()}
    return {**config[model], **trial_values}


def best_per_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return the row with the highest validation macro-F1 for each model."""
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["model"]
        if name not in best or float(row["val_macro_f1"]) > float(best[name]["val_macro_f1"]):
            best[name] = row
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tune the landmark models on validation.")
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
        "--limit",
        type=int,
        default=0,
        help="Try at most this many trials per model (0 = all). For quick tests.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, str]]:
    """Run the trials that are not in the CSV yet and return all rows of the CSV."""
    data_config = load_config(args.data_config)
    config = load_config(args.config)
    seed = int(data_config["seed"])
    set_seed(seed)
    logger.info("Seed: %d", seed)

    label_map = read_label_map(resolve_path(data_config, "label_map"))
    n_classes, input_dim = len(label_map), int(config["input_dim"])
    processed_dir = resolve_path(data_config, "processed_dir")
    cache = load_cache(args.features or processed_dir / SPLIT_FEATURES_FILE)
    # Only train and val are selected. The test split is never used while tuning.
    X_train, y_train, _ = select_split(cache, "train")
    X_val, y_val, _ = select_split(cache, "val")
    logger.info("Images with a hand: %d train, %d val", len(y_train), len(y_val))

    tuning = config["tuning"]
    trials_csv = REPO_ROOT / tuning["trials_csv"]
    for name in args.models:
        trials = build_trials(tuning, name, seed)
        if args.limit:
            trials = trials[: args.limit]
        for number, params in enumerate(trials, start=1):
            run_name = f"{name}-{number:02d}"
            if has_trial(trials_csv, name, params, seed):
                logger.info("%s already in %s, skipped", run_name, trials_csv.name)
                continue
            settings = model_settings(config, name, params)
            started = time.perf_counter()
            model = fit_model(
                name, settings, X_train, y_train, X_val, y_val, seed, input_dim, n_classes
            )
            seconds = time.perf_counter() - started
            accuracy, macro_f1 = score(model, X_val, y_val)
            append_trial(
                trials_csv,
                name,
                params,
                {"val_macro_f1": macro_f1, "val_accuracy": accuracy, "seconds": seconds},
                run_name,
                seed,
            )
            logger.info(
                "%s (%d of %d): validation macro-F1 %.4f, accuracy %.4f, %.1f s",
                run_name,
                number,
                len(trials),
                macro_f1,
                accuracy,
                seconds,
            )

    rows = read_trials(trials_csv)
    for name, row in best_per_model(rows).items():
        logger.info("Best %s: %s with macro-F1 %s", name, row["run_name"], row["val_macro_f1"])
    return rows


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

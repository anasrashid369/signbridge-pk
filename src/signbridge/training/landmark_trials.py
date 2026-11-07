"""Append and read landmark hyperparameter trials (validation split only).

One row per trial. Parameter columns that do not apply to a model stay empty.
A max_depth of None is written as the text "None" so it differs from an empty cell.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from signbridge.training.trials import read_trials

# Same names as the tuning grid in configs/landmark.yaml.
PARAM_COLUMNS = [
    "svm_C",
    "svm_gamma",
    "forest_n_estimators",
    "forest_max_depth",
    "mlp_learning_rate",
    "mlp_batch_size",
    "mlp_dropout",
    "mlp_hidden_size",
    "mlp_weight_decay",
]
COLUMNS = [
    "timestamp",
    "run_name",
    "model",
    *PARAM_COLUMNS,
    "val_macro_f1",
    "val_accuracy",
    "seconds",
    "seed",
]


def _check(params: dict[str, object]) -> None:
    """Raise ValueError if a parameter name is not a known column."""
    unknown = sorted(set(params) - set(PARAM_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown trial parameters: {unknown}")


def _param_text(params: dict[str, object]) -> list[str]:
    """Return the parameter values as text, in column order ("" when not set)."""
    return [str(params[column]) if column in params else "" for column in PARAM_COLUMNS]


def trial_key(model: str, params: dict[str, object], seed: int) -> tuple:
    """Return the identity of a trial, used to skip repeated trials."""
    _check(params)
    return (model, int(seed), *_param_text(params))


def has_trial(path: Path, model: str, params: dict[str, object], seed: int) -> bool:
    """Return True if this model, parameter set and seed are already in the CSV."""
    wanted = trial_key(model, params, seed)
    for row in read_trials(path):
        try:
            recorded = (row["model"], int(row["seed"]), *(row[c] for c in PARAM_COLUMNS))
        except (KeyError, TypeError, ValueError):
            continue
        if recorded == wanted:
            return True
    return False


def append_trial(
    path: Path,
    model: str,
    params: dict[str, object],
    metrics: dict[str, float],
    run_name: str,
    seed: int,
) -> dict[str, object]:
    """Append one trial row and return the row that was written.

    Args:
        path: The trials CSV, for example reports/tuning/landmark_trials.csv.
        model: "svm", "forest" or "mlp".
        params: Parameter values keyed by column name (see PARAM_COLUMNS).
        metrics: Must contain val_macro_f1, val_accuracy and seconds.
        run_name: A label for the trial.
        seed: The seed used for the trial.
    """
    _check(params)
    row: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "model": model,
        **dict(zip(PARAM_COLUMNS, _param_text(params), strict=True)),
        "val_macro_f1": round(float(metrics["val_macro_f1"]), 4),
        "val_accuracy": round(float(metrics["val_accuracy"]), 4),
        "seconds": round(float(metrics["seconds"]), 2),
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row

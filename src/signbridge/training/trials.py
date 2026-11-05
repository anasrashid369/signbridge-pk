"""Append and read hyperparameter trial results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "timestamp",
    "run_name",
    "lr",
    "batch_size",
    "dropout",
    "weight_decay",
    "unfreeze_blocks",
    "epochs",
    "best_val_macro_f1",
    "val_accuracy",
    "epochs_run",
    "seconds",
    "seed",
    "trainable_parameters",
]
PARAM_COLUMNS = ["lr", "batch_size", "dropout", "weight_decay", "unfreeze_blocks"]


def read_trials(path: Path) -> list[dict[str, str]]:
    """Return all trial rows, or an empty list if the file does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def trial_key(params: dict[str, object], seed: int) -> tuple:
    """Return the identity of a parameter set, used to skip repeated trials."""
    return (int(seed),) + tuple(float(params[column]) for column in PARAM_COLUMNS)


def has_trial(path: Path, params: dict[str, object], seed: int) -> bool:
    """Return True if this parameter set and seed are already in the CSV."""
    wanted = trial_key(params, seed)
    for row in read_trials(path):
        recorded = {column: row[column] for column in PARAM_COLUMNS}
        try:
            if trial_key(recorded, int(row["seed"])) == wanted:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def append_trial(
    path: Path, params: dict[str, object], metrics: dict[str, object], run_name: str, seed: int
) -> dict[str, object]:
    """Append one trial row and return the row that was written."""
    row: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        **{column: params[column] for column in PARAM_COLUMNS},
        "epochs": metrics["epochs"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "val_accuracy": metrics["best_val_accuracy"],
        "epochs_run": metrics["epochs_run"],
        "seconds": round(float(metrics["seconds"]), 2),
        "seed": int(seed),
        "trainable_parameters": metrics.get("trainable_parameters", ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row

"""Tests for the trials CSV helpers."""

from pathlib import Path

from signbridge.training.trials import COLUMNS, append_trial, has_trial, read_trials

PARAMS = {
    "lr": 0.001,
    "batch_size": 32,
    "dropout": 0.2,
    "weight_decay": 0.0001,
    "unfreeze_blocks": 2,
}
METRICS = {
    "epochs": 3,
    "best_val_macro_f1": 0.72,
    "best_val_accuracy": 0.75,
    "epochs_run": 3,
    "seconds": 12.34,
    "trainable_parameters": 1234,
}


def test_append_and_read_trial(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_trial(path, PARAMS, METRICS, "trial_000", seed=42)
    rows = read_trials(path)
    assert len(rows) == 1
    assert list(rows[0]) == COLUMNS
    assert rows[0]["run_name"] == "trial_000"
    assert float(rows[0]["best_val_macro_f1"]) == 0.72


def test_has_trial_matches_parameters_and_seed(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_trial(path, PARAMS, METRICS, "trial_000", seed=42)
    assert has_trial(path, PARAMS, seed=42)
    assert not has_trial(path, PARAMS, seed=7)
    changed = {**PARAMS, "lr": 0.0003}
    assert not has_trial(path, changed, seed=42)


def test_has_trial_on_missing_file_is_false(tmp_path: Path) -> None:
    assert not has_trial(tmp_path / "missing.csv", PARAMS, seed=42)


def test_two_trials_are_appended(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_trial(path, PARAMS, METRICS, "trial_000", seed=42)
    append_trial(path, {**PARAMS, "lr": 0.0003}, METRICS, "trial_001", seed=42)
    assert len(read_trials(path)) == 2

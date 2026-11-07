"""Tests for the landmark trial log."""

from pathlib import Path

import pytest

from signbridge.training.landmark_trials import COLUMNS, append_trial, has_trial
from signbridge.training.trials import read_trials

METRICS = {"val_macro_f1": 0.912345, "val_accuracy": 0.93, "seconds": 1.234}


def test_append_trial_writes_header_and_row(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_trial(path, "svm", {"svm_C": 10.0, "svm_gamma": "scale"}, METRICS, "t1", seed=42)
    rows = read_trials(path)
    assert list(rows[0]) == COLUMNS
    assert rows[0]["model"] == "svm"
    assert rows[0]["svm_C"] == "10.0"
    assert rows[0]["svm_gamma"] == "scale"
    assert rows[0]["forest_max_depth"] == ""  # not used by the SVM
    assert rows[0]["val_macro_f1"] == "0.9123"
    assert rows[0]["seed"] == "42"


def test_has_trial_finds_repeats_only(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    params = {"svm_C": 10.0, "svm_gamma": "scale"}
    assert not has_trial(path, "svm", params, 42)  # no file yet
    append_trial(path, "svm", params, METRICS, "t1", seed=42)
    assert has_trial(path, "svm", params, 42)
    assert not has_trial(path, "svm", params, 7)  # other seed
    assert not has_trial(path, "svm", {"svm_C": 1.0, "svm_gamma": "scale"}, 42)



def test_none_depth_is_kept_apart_from_a_number(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    params = {"forest_n_estimators": 300, "forest_max_depth": None}
    append_trial(path, "forest", params, METRICS, "t1", seed=1)
    assert read_trials(path)[0]["forest_max_depth"] == "None"
    assert has_trial(path, "forest", params, 1)
    assert not has_trial(path, "forest", params | {"forest_max_depth": 10}, 1)


def test_unknown_parameter_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        append_trial(tmp_path / "t.csv", "svm", {"svm_kernel": "rbf"}, METRICS, "t1", seed=1)

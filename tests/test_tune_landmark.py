"""Tests for the landmark tuning command."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import json  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from signbridge.training.trials import read_trials  # noqa: E402
from signbridge.tune_landmark import (  # noqa: E402
    best_per_model,
    build_trials,
    model_settings,
    parse_args,
    run,
)

N_CLASSES = 3
CLASS_NAMES = ["A", "B", "C"]
MODEL_SETTINGS = {
    "svm": {"C": 1.0, "gamma": "scale"},
    "forest": {"n_estimators": 10, "max_depth": None},
    "mlp": {
        "hidden_size": 8,
        "n_hidden_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 16,
        "epochs": 3,
        "patience": 3,
    },
}
SMALL_GRID = {
    "svm_C": [1.0, 10.0],
    "svm_gamma": ["scale"],
    "forest_n_estimators": [5, 10],
    "forest_max_depth": [None],
    "mlp_learning_rate": [0.01],
    "mlp_batch_size": [16],
    "mlp_dropout": [0.1],
    "mlp_hidden_size": [8],
    "mlp_weight_decay": [0.0],
    "mlp_max_trials": 5,
}  # 2 SVM + 2 forest + 1 MLP = 5 trials


def make_cache_without_test(path: Path) -> None:
    """A small separable cache that has train and val images only (no test split)."""
    rng = np.random.default_rng(0)
    X, y, paths, split = [], [], [], []
    for cls in range(N_CLASSES):
        for name, count in (("train", 30), ("val", 10)):
            for number in range(count):
                X.append(rng.normal(loc=3 * cls, scale=1.0, size=63))
                y.append(cls)
                paths.append(f"{CLASS_NAMES[cls]}/{name}{number}.jpg")
                split.append(name)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        paths=np.asarray(paths),
        split=np.asarray(split),
    )


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Path]:
    """Temporary configs and cache. Every path is absolute, so nothing touches the repo."""
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"A": 0, "B": 1, "C": 2}), encoding="utf-8")
    make_cache_without_test(tmp_path / "landmark_features_split.npz")
    trials_csv = tmp_path / "trials.csv"
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {"seed": 1, "paths": {"label_map": str(label_map), "processed_dir": str(tmp_path)}}
        ),
        encoding="utf-8",
    )
    landmark_config = tmp_path / "landmark.yaml"
    landmark_config.write_text(
        yaml.safe_dump(
            {
                "input_dim": 63,
                "n_classes": N_CLASSES,
                **MODEL_SETTINGS,
                "tuning": {"trials_csv": str(trials_csv), **SMALL_GRID},
            }
        ),
        encoding="utf-8",
    )
    return {"data": data_config, "landmark": landmark_config, "trials": trials_csv}


def args_for(project: dict[str, Path], *extra: str):
    return parse_args(
        ["--config", str(project["landmark"]), "--data-config", str(project["data"]), *extra]
    )


def test_build_trials_uses_the_full_grid_for_svm() -> None:
    tuning = {"svm_C": [1, 10], "svm_gamma": ["scale", 0.1]}
    trials = build_trials(tuning, "svm", seed=1)
    assert len(trials) == 4
    assert {"svm_C": 10, "svm_gamma": 0.1} in trials


def test_build_trials_samples_the_mlp_grid_reproducibly() -> None:
    tuning = {
        "mlp_learning_rate": [0.0003, 0.001, 0.003],
        "mlp_batch_size": [32, 64, 128],
        "mlp_dropout": [0.1, 0.3, 0.5],
        "mlp_hidden_size": [64, 128, 256],
        "mlp_weight_decay": [0.0, 0.0001, 0.001],
        "mlp_max_trials": 7,
    }
    first = build_trials(tuning, "mlp", seed=3)
    assert len(first) == 7
    assert first == build_trials(tuning, "mlp", seed=3)
    assert len({tuple(trial.items()) for trial in first}) == 7  # no combination twice


def test_model_settings_puts_trial_values_on_top() -> None:
    config = {"mlp": {"hidden_size": 8, "epochs": 3, "learning_rate": 0.01}}
    settings = model_settings(config, "mlp", {"mlp_learning_rate": 0.003})
    assert settings == {"hidden_size": 8, "epochs": 3, "learning_rate": 0.003}


def test_best_per_model_picks_highest_macro_f1() -> None:
    rows = [
        {"model": "svm", "run_name": "svm-01", "val_macro_f1": "0.80"},
        {"model": "svm", "run_name": "svm-02", "val_macro_f1": "0.91"},
        {"model": "forest", "run_name": "forest-01", "val_macro_f1": "0.70"},
    ]
    best = best_per_model(rows)
    assert best["svm"]["run_name"] == "svm-02"
    assert best["forest"]["run_name"] == "forest-01"


def test_run_logs_every_trial_and_never_needs_the_test_split(project: dict[str, Path]) -> None:
    rows = run(args_for(project))  # the cache has no test images, so this proves test is unused
    assert len(rows) == 5
    assert sorted(row["model"] for row in rows) == ["forest", "forest", "mlp", "svm", "svm"]
    assert all(0.0 <= float(row["val_macro_f1"]) <= 1.0 for row in rows)
    assert len(read_trials(project["trials"])) == 5


def test_second_run_skips_repeated_trials(project: dict[str, Path]) -> None:
    run(args_for(project))
    rows = run(args_for(project))
    assert len(rows) == 5


def test_limit_caps_trials_per_model(project: dict[str, Path]) -> None:
    rows = run(args_for(project, "--limit", "1"))
    assert len(rows) == 3  # one SVM, one forest, one MLP

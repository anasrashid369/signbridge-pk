"""Tests for the tuning driver, with training replaced by a stub."""

import itertools
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from signbridge import tune_cnn
from signbridge.training.trials import PARAM_COLUMNS, read_trials
from tests.cnn_helpers import make_split, write_cnn_config


def fake_train(config, options):  # noqa: ANN001
    """Return a fixed training summary without touching torch."""
    return {
        "best_val_macro_f1": 0.5,
        "best_val_accuracy": 0.6,
        "epochs_run": 1,
        "seconds": 1.0,
        "trainable_parameters": 100,
    }


def test_tune_logs_then_skips_recorded_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    config = write_cnn_config(tmp_path, split_file, image_root, label_map, num_classes=2)
    monkeypatch.setattr(tune_cnn.train_cnn, "train", fake_train)
    assert tune_cnn.main(["--config", str(config)]) == 0
    rows = read_trials(tmp_path / "outputs" / "trials.csv")
    assert len(rows) == 1
    assert rows[0]["run_name"] == "trial_000"

    calls = []
    monkeypatch.setattr(tune_cnn.train_cnn, "train", lambda *args: calls.append(args))
    assert tune_cnn.main(["--config", str(config)]) == 0
    assert calls == []
    assert len(read_trials(tmp_path / "outputs" / "trials.csv")) == 1


def test_best_trial_uses_macro_f1_then_parameter_count() -> None:
    rows = [
        {"best_val_macro_f1": "0.5", "trainable_parameters": "10"},
        {"best_val_macro_f1": "0.7", "trainable_parameters": "100"},
        {"best_val_macro_f1": "0.7", "trainable_parameters": "50"},
    ]
    assert tune_cnn.best_trial(rows) is rows[2]
    assert tune_cnn.best_trial([]) is None


def _grid(seed: int, lr: list[float] | None = None) -> dict:
    """Return a 32-combination config with the given seed."""
    return {
        "seed": seed,
        "tuning": {
            "grid": {
                "lr": lr or [0.0003, 0.001],
                "batch_size": [32, 64],
                "dropout": [0.2, 0.4],
                "weight_decay": [0.0001, 0.001],
                "unfreeze_blocks": [0, 2],
            }
        },
    }


def test_parameter_sets_are_a_seeded_sample_not_the_first_twelve() -> None:
    config = _grid(42)
    first = tune_cnn.parameter_sets(config)
    second = tune_cnn.parameter_sets(config)
    assert len(first) == 32
    assert first == second
    keys = [tuple(params[column] for column in PARAM_COLUMNS) for params in first]
    assert len(set(keys)) == 32

    grid = config["tuning"]["grid"]
    stable = [
        dict(zip(PARAM_COLUMNS, combination, strict=True))
        for combination in itertools.product(*(grid[column] for column in PARAM_COLUMNS))
    ]
    assert first[:12] != stable[:12]
    assert tune_cnn.parameter_sets(_grid(7))[:12] != first[:12]


def test_parameter_sets_remove_duplicate_combinations() -> None:
    config = _grid(42, lr=[0.001, 0.001])
    sets = tune_cnn.parameter_sets(config)
    assert len(sets) == 16
    assert len({tuple(params[column] for column in PARAM_COLUMNS) for params in sets}) == 16


def test_tune_passes_max_per_class_to_each_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    config = write_cnn_config(
        tmp_path, split_file, image_root, label_map, num_classes=2, max_trials=1
    )
    captured = []

    def capture(config, options):  # noqa: ANN001
        captured.append(options)
        return fake_train(config, options)

    monkeypatch.setattr(tune_cnn.train_cnn, "train", capture)
    code = tune_cnn.main(
        ["--config", str(config), "--max-per-class", "1", "--max-batches", "1"]
    )
    assert code == 0
    assert len(captured) == 1
    assert captured[0].max_per_class == 1
    assert captured[0].max_batches == 1

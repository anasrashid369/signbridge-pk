"""Tests for tools/make_fake_predictions.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signbridge.config import REPO_ROOT
from signbridge.training.predictions import load_predictions
from tools.make_fake_predictions import class_names, main, make_fake_predictions


def test_writes_the_contract_arrays(tmp_path: Path) -> None:
    output = tmp_path / "fake_cnn_test.npz"
    assert main(["--output", str(output), "--rows", "40", "--seed", "7"]) == 0
    data = load_predictions(output)
    assert data["y_true"].shape == (40,)
    assert data["y_proba"].shape == (40, 26)
    assert data["paths"].shape == (40,)
    assert list(data["class_names"]) == class_names(26)
    assert np.allclose(data["y_proba"].sum(axis=1), 1.0, atol=1e-5)
    assert data["y_true"].min() >= 0
    assert data["y_true"].max() < 26
    assert all(path.startswith("fake/") for path in data["paths"])


def test_same_seed_gives_the_same_arrays() -> None:
    first = make_fake_predictions(30, 26, seed=42, accuracy=0.8)
    second = make_fake_predictions(30, 26, seed=42, accuracy=0.8)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_another_seed_changes_the_arrays() -> None:
    first = make_fake_predictions(30, 26, seed=42, accuracy=0.8)
    second = make_fake_predictions(30, 26, seed=43, accuracy=0.8)
    assert not np.array_equal(first[1], second[1])


def test_accuracy_controls_the_top_class() -> None:
    y_true, y_proba, _ = make_fake_predictions(500, 26, seed=1, accuracy=1.0)
    assert np.array_equal(y_proba.argmax(axis=1), y_true)


def test_refuses_to_write_inside_the_repository() -> None:
    inside = REPO_ROOT / "fake_predictions_must_not_be_written.npz"
    assert main(["--output", str(inside)]) == 1
    assert not inside.exists()


@pytest.mark.parametrize("args", [["--rows", "0"], ["--accuracy", "2"], ["--classes", "0"]])
def test_bad_arguments_return_one(tmp_path: Path, args: list[str]) -> None:
    assert main(["--output", str(tmp_path / "fake.npz"), *args]) == 1


def test_class_names_are_letters_then_numbered() -> None:
    assert class_names(2) == ["A", "B"]
    assert class_names(28)[26] == "class_26"


def test_help_exits_cleanly() -> None:
    from tools.make_fake_predictions import parse_args

    with pytest.raises(SystemExit) as info:
        parse_args(["--help"])
    assert info.value.code == 0

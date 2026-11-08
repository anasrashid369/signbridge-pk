"""Tests for src/signbridge/evaluation/common_set.py."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.evaluation.common_set import restrict_to_common

NAMES = np.array(["A", "B", "C"])


def make_file(paths: list[str], truth: list[int], offset: float = 0.0) -> dict[str, np.ndarray]:
    """Build a small prediction dictionary; offset changes the probabilities per file."""
    proba = np.full((len(paths), 3), 0.1 + offset)
    proba[np.arange(len(paths)), truth] = 0.8 - 2 * offset
    return {
        "y_true": np.array(truth),
        "y_proba": proba,
        "paths": np.array(paths),
        "class_names": NAMES,
    }


def test_only_shared_images_are_kept_in_the_order_of_the_first_file() -> None:
    first = make_file(["A/1.jpg", "B/2.jpg", "C/3.jpg", "A/4.jpg"], [0, 1, 2, 0])
    second = make_file(["C/3.jpg", "A/1.jpg", "B/9.jpg"], [2, 0, 1], offset=0.05)
    kept_first, kept_second = restrict_to_common(first, second)
    assert kept_first["paths"].tolist() == ["A/1.jpg", "C/3.jpg"]
    assert kept_second["paths"].tolist() == ["A/1.jpg", "C/3.jpg"]
    assert kept_first["y_true"].tolist() == kept_second["y_true"].tolist() == [0, 2]
    assert kept_second["y_proba"][0, 0] == pytest.approx(0.7)  # rows follow their paths


def test_no_common_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="no image in common"):
        restrict_to_common(make_file(["A/1.jpg"], [0]), make_file(["B/2.jpg"], [1]))


def test_different_class_names_are_rejected() -> None:
    other = make_file(["A/1.jpg"], [0])
    other["class_names"] = np.array(["A", "C", "B"])
    with pytest.raises(ValueError, match="class names"):
        restrict_to_common(make_file(["A/1.jpg"], [0]), other)


def test_repeated_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        restrict_to_common(make_file(["A/1.jpg", "A/1.jpg"], [0, 0]), make_file(["A/1.jpg"], [0]))


def test_disagreeing_true_classes_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        restrict_to_common(make_file(["A/1.jpg"], [0]), make_file(["A/1.jpg"], [1]))

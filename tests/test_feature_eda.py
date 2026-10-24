"""Tests for the feature EDA helpers."""

from pathlib import Path

import numpy as np
import pytest

from data_scripts.eda.feature_eda import (
    correlation_matrix,
    detection_rates,
    feature_name,
    load_detections,
    plot_before_after,
    plot_correlation,
    plot_detection_rate,
    save_detection_table,
)
from data_scripts.features.extract import save_detections


def test_detection_rates_counts() -> None:
    records = [("A", True), ("A", False), ("A", True), ("B", False)]
    assert detection_rates(records) == {"A": (2, 3), "B": (0, 1)}


def test_load_detections_reads_booleans(tmp_path: Path) -> None:
    destination = save_detections(tmp_path, [("A/1.jpg", "A", True), ("B/1.jpg", "B", False)])
    assert load_detections(destination) == [("A", True), ("B", False)]


def test_correlation_matrix_drops_constant_columns() -> None:
    features = np.random.default_rng(0).random((10, 63))
    features[:, :3] = 0.0
    matrix, kept = correlation_matrix(features)
    assert matrix.shape == (60, 60)
    assert list(kept[:2]) == [3, 4]


def test_correlation_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        correlation_matrix(np.random.default_rng(0).random((1, 63)))


def test_feature_name() -> None:
    assert feature_name(0) == "0x"
    assert feature_name(25) == "8y"


def test_save_detection_table(tmp_path: Path) -> None:
    rates = {"A": (2, 3), "B": (1, 1)}
    destination = save_detection_table(rates, ["A", "B", "C"], tmp_path / "table.csv")
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "label,images,hand_found,detection_rate_percent"
    assert lines[1] == "A,3,2,66.7"
    assert lines[2] == "B,1,1,100.0"
    assert len(lines) == 3


def test_charts_are_saved(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    raw = rng.random((6, 63))
    features = rng.random((6, 63))
    rates = {"A": (2, 3), "B": (1, 1)}
    plot_detection_rate(rates, ["A", "B"], tmp_path / "d.png", "ASL baseline")
    plot_correlation(features, tmp_path / "c.png", "ASL baseline", 8)
    plot_before_after(raw, features, tmp_path / "b.png", "ASL baseline", 8)
    for name in ("d.png", "c.png", "b.png"):
        assert (tmp_path / name).stat().st_size > 0

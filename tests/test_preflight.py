"""Tests for signbridge.preflight on a synthetic split and tiny model."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from signbridge.preflight import run_preflight, split_statistics  # noqa: E402
from tests.cnn_helpers import make_split, write_cnn_config  # noqa: E402


def test_split_statistics_counts_rows(tmp_path: Path) -> None:
    split_file, _, label_map = make_split(tmp_path, classes=("A", "B"), per_split=2)
    columns = {"path": "path", "label": "label", "split": "split"}
    statistics = split_statistics(split_file, columns)
    assert statistics["rows"] == 12
    assert statistics["classes"] == ["A", "B"]
    assert statistics["counts"] == {"train": 4, "val": 4, "test": 4}
    assert statistics["per_class"]["A"] == {"train": 2, "val": 2, "test": 2}
    assert label_map == {"A": 0, "B": 1}


def test_preflight_measures_batches_without_writing_outputs(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path, classes=("A", "B"), per_split=2)
    config = write_cnn_config(
        tmp_path, split_file, image_root, label_map, num_classes=2, epochs=1, batch_size=2
    )
    result = run_preflight(config, batches=1)
    assert result["statistics"]["rows"] == 12
    assert result["train_images"] == 4
    assert result["val_images"] == 4
    assert result["seconds_per_batch"] > 0
    assert result["epoch_seconds"] > 0
    assert result["tuning_seconds"] > 0
    assert result["final_seconds"] > 0
    assert not (tmp_path / "outputs").exists()

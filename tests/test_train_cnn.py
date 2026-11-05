"""Tests for the train_cnn command."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from signbridge import train_cnn
from signbridge.config import load_config
from signbridge.training.predictions import load_predictions
from tests.cnn_helpers import make_split, write_cnn_config


def test_training_writes_history_checkpoint_and_predictions(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    config = write_cnn_config(tmp_path, split_file, image_root, label_map, num_classes=2)
    code = train_cnn.main(
        [
            "--config",
            str(config),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-batches",
            "2",
            "--run-name",
            "smoke",
        ]
    )
    assert code == 0
    with (tmp_path / "outputs" / "smoke.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "lr",
        "seconds",
    ]
    assert len(rows) == 1
    assert (tmp_path / "outputs" / "smoke.pt").is_file()
    predictions = load_predictions(tmp_path / "outputs" / "predictions" / "smoke_val.npz")
    assert predictions["y_proba"].shape[1] == 2
    assert len(predictions["paths"]) == predictions["y_proba"].shape[0]


def test_test_split_is_untouched_without_the_flag(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    config = write_cnn_config(tmp_path, split_file, image_root, label_map, num_classes=2)
    for row in split_file.read_text(encoding="utf-8").splitlines()[1:]:
        if ",test" in row:
            path = image_root / row.split(",")[0]
            path.unlink()
    code = train_cnn.main(
        ["--config", str(config), "--epochs", "1", "--max-batches", "1", "--run-name", "guard"]
    )
    assert code == 0
    assert not (tmp_path / "outputs" / "predictions" / "guard_test.npz").exists()


def test_evaluate_test_reads_the_test_split(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    config = write_cnn_config(tmp_path, split_file, image_root, label_map, num_classes=2)
    for row in split_file.read_text(encoding="utf-8").splitlines()[1:]:
        if ",test" in row:
            path = image_root / row.split(",")[0]
            path.unlink()
    code = train_cnn.main(
        [
            "--config",
            str(config),
            "--epochs",
            "1",
            "--max-batches",
            "1",
            "--run-name",
            "guard",
            "--evaluate-test",
        ]
    )
    assert code == 1
    assert not (tmp_path / "outputs" / "predictions" / "guard_test.npz").exists()


def test_max_per_class_caps_the_train_split_only(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path, per_split=3)
    config = load_config(
        write_cnn_config(tmp_path, split_file, image_root, label_map, num_classes=2)
    )
    options = SimpleNamespace(
        split_file=None,
        image_root=None,
        epochs=1,
        batch_size=2,
        max_batches=None,
        max_per_class=1,
        run_name="cap",
        evaluate_test=False,
    )
    train_set, val_set, _, _, _ = train_cnn._loaders(config, options)
    assert len(train_set) == 2
    assert len(val_set) == 6

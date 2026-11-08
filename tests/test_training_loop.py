"""Tests for the training loop and prediction files."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from signbridge.training.loop import evaluate_epoch, set_seed, train_one_epoch
from signbridge.training.predictions import load_predictions, save_predictions


class TinyNet(nn.Module):
    """A one-layer network that is fast enough for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(1))


def make_loader() -> DataLoader:
    """Return a small deterministic loader."""
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(8, 3, 4, 4, generator=generator)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def test_training_changes_the_loss() -> None:
    set_seed(3)
    model = TinyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = evaluate_epoch(model, make_loader(), criterion, torch.device("cpu")).loss
    train_one_epoch(model, make_loader(), criterion, optimizer, torch.device("cpu"))
    after = evaluate_epoch(model, make_loader(), criterion, torch.device("cpu")).loss
    assert after != before


def test_frozen_batchnorm_stays_in_eval_during_training() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 2, 1),
        nn.BatchNorm2d(2),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 2),
    )
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    for parameter in model[1].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    train_one_epoch(
        model,
        make_loader(),
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        freeze_batchnorm=True,
    )
    assert model[1].training is False


def test_evaluate_returns_probabilities_and_macro_f1() -> None:
    set_seed(3)
    model = TinyNet()
    metrics = evaluate_epoch(model, make_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert metrics.y_proba.shape == (8, 2)
    assert np.allclose(metrics.y_proba.sum(axis=1), 1.0, atol=1e-5)
    assert metrics.y_true.shape == (8,)
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_prediction_file_round_trip(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 1], dtype=np.int64)
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
    destination = save_predictions(
        tmp_path / "cnn_val.npz", y_true, y_proba, ["A/1.jpg", "B/1.jpg", "B/2.jpg"], ["A", "B"]
    )
    loaded = load_predictions(destination)
    assert set(loaded) == {"y_true", "y_proba", "paths", "class_names"}
    assert np.array_equal(loaded["y_true"], y_true)
    assert np.allclose(loaded["y_proba"], y_proba)
    assert list(loaded["class_names"]) == ["A", "B"]


def test_prediction_length_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        save_predictions(
            tmp_path / "bad.npz",
            np.array([0, 1]),
            np.zeros((2, 2), dtype=np.float32),
            ["A/1.jpg"],
            ["A", "B"],
        )

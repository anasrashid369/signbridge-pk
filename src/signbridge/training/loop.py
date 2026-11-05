"""One training epoch and one evaluation pass."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    """Loss, accuracy, macro-F1 and predictions for one evaluation pass."""

    loss: float
    accuracy: float
    macro_f1: float
    y_true: np.ndarray
    y_proba: np.ndarray


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Return CUDA when it is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _limited(loader: DataLoader, max_batches: int | None) -> Iterator:
    """Yield batches, stopping after max_batches when a limit is given."""
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Train for one epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in _limited(loader, max_batches):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Evaluate one split and return loss, accuracy, macro-F1 and probabilities."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, labels in _limited(loader, max_batches):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        truths.append(labels.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(truths) if truths else np.empty(0, dtype=np.int64)
    y_proba = (
        np.concatenate(probabilities)
        if probabilities
        else np.empty((0, 0), dtype=np.float32)
    )
    macro_f1 = float(f1_score(y_true, y_proba.argmax(axis=1), average="macro", zero_division=0))
    return EpochMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        macro_f1=macro_f1,
        y_true=y_true,
        y_proba=y_proba,
    )

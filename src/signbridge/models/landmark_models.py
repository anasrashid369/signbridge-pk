
"""Landmark-based classifiers: SVM, Random Forest and a PyTorch MLP.

Input: 63 normalised landmark values per hand (see data_scripts/features/landmarks.py).
Output: class probabilities over the 26 letters.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from signbridge.training.loop import set_seed

# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def build_svm(
    C: float = 10.0,
    gamma="scale",
    seed: int = 42,
    calibration_folds: int = 5
) -> Pipeline:
    """RBF SVM with feature scaling and calibrated class probabilities.

    Uses CalibratedClassifierCV for sigmoid probability calibration instead
    of SVC's deprecated probability parameter.

    The SVC is fitted using cross-validated decision values, and calibration
    produces valid class probabilities.
    """

    svc = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        random_state=seed
    )

    calibrated = CalibratedClassifierCV(
        estimator=svc,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False
    )

    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", calibrated)
    ])


def build_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    seed: int = 42
) -> RandomForestClassifier:
    """Random Forest. Tree models do not need feature scaling."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )


def full_proba(
    model,
    X: np.ndarray,
    n_classes: int = 26
) -> np.ndarray:
    """Return predict_proba as an (n, n_classes) array.

    Scikit-learn only returns columns for classes seen in training, so
    classes missing from training are filled with zeros.
    """

    proba = model.predict_proba(X)

    out = np.zeros(
        (len(X), n_classes),
        dtype=np.float64
    )

    out[:, np.asarray(model.classes_, dtype=int)] = proba

    return out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LandmarkMLP(nn.Module):
    """63 -> [Linear, ReLU, Dropout] x n_hidden_layers -> Linear -> 26 logits."""

    def __init__(
        self,
        input_dim: int = 63,
        n_classes: int = 26,
        hidden_size: int = 128,
        n_hidden_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = input_dim

        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

            in_features = hidden_size

        layers.append(
            nn.Linear(in_features, n_classes)
        )  # logits; softmax is inside the loss

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _evaluate(model, X, y, criterion, device):
    """Return loss, accuracy and macro-F1 on a dataset with dropout off."""

    model.eval()

    with torch.no_grad():
        logits = model(X.to(device))
        loss = criterion(logits, y.to(device)).item()
        pred = logits.argmax(dim=1).cpu().numpy()

    y_np = y.numpy()

    return (
        loss,
        float((pred == y_np).mean()),
        float(f1_score(y_np, pred, average="macro"))
    )


def fit_mlp(
    model: LandmarkMLP,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    device: str = "cpu"
) -> list[dict]:
    """Train with CrossEntropyLoss and AdamW; early stopping on validation macro-F1.

    Restores the best weights at the end. Returns one dict per epoch
    (train/val loss and accuracy, val macro-F1).
    """

    set_seed(seed)

    model.to(device)

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)

    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history: list[dict] = []

    best_f1, best_state, wait = -1.0, None, 0

    for epoch in range(1, epochs + 1):

        model.train()

        loss_sum, correct = 0.0, 0

        for xb, yb in loader:

            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item() * len(xb)

            correct += (
                logits.argmax(dim=1) == yb
            ).sum().item()

        val_loss, val_acc, val_f1 = _evaluate(
            model, Xv, yv, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(Xt),
            "train_acc": correct / len(Xt),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        if val_f1 > best_f1:

            best_f1, wait = val_f1, 0

            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

        else:

            wait += 1

            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_proba_mlp(
    model: LandmarkMLP,
    X,
    device: str = "cpu"
) -> np.ndarray:
    """Return softmax probabilities with shape (n, n_classes)."""

    model.to(device).eval()

    with torch.no_grad():

        logits = model(
            torch.as_tensor(
                X,
                dtype=torch.float32
            ).to(device)
        )

        return torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()


def write_history_csv(
    history: list[dict],
    path: str | Path
) -> None:
    """Save per-epoch loss and accuracy for the learning curves."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

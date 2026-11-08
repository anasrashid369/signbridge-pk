import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from signbridge.models.landmark_models import (  # noqa: E402
    LandmarkMLP,
    build_forest,
    build_svm,
    fit_mlp,
    full_proba,
    predict_proba_mlp,
    write_history_csv,
)


def toy_data(n_classes=3, per_class=30, dim=63, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal(loc=3 * c, scale=1.0, size=(per_class, dim)) for c in range(n_classes)
    ])
    y = np.repeat(np.arange(n_classes), per_class)
    return X, y


@pytest.mark.parametrize("builder", [build_svm, build_forest])
def test_sklearn_models_give_valid_probabilities(builder):
    X, y = toy_data()
    model = builder().fit(X, y)
    proba = full_proba(model, X, n_classes=26)
    assert proba.shape == (len(X), 26)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba[:, 3:].sum() == 0.0  # classes unseen in training get zero probability


def test_svm_learns_toy_problem():
    X, y = toy_data()
    assert build_svm().fit(X, y).score(X, y) > 0.95


def test_mlp_output_shape():
    model = LandmarkMLP(input_dim=63, n_classes=26)
    assert model(torch.zeros(5, 63)).shape == (5, 26)


def test_mlp_trains_and_records_history(tmp_path):
    X, y = toy_data(per_class=40)
    model = LandmarkMLP(input_dim=63, n_classes=3, hidden_size=16, n_hidden_layers=1,
                        dropout=0.1)
    history = fit_mlp(model, X, y, X, y, epochs=15, patience=15, batch_size=32)
    assert len(history) == 15
    assert set(history[0]) == {
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1",
    }
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    proba = predict_proba_mlp(model, X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    out = tmp_path / "hist.csv"
    write_history_csv(history, out)
    assert out.read_text().startswith("epoch,train_loss")


def test_mlp_training_is_reproducible():
    X, y = toy_data(per_class=40)
    losses = []
    for _ in range(2):
        torch.manual_seed(0)
        m = LandmarkMLP(input_dim=63, n_classes=3, hidden_size=16, n_hidden_layers=1)
        h = fit_mlp(m, X, y, X, y, epochs=3, patience=3, seed=7)
        losses.append(h[-1]["train_loss"])
    assert losses[0] == pytest.approx(losses[1])


def test_svm_uses_calibration_not_the_deprecated_probability_flag():
    calibrated = build_svm().steps[-1][1]

    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.estimator.probability is not True
    assert calibrated.ensemble is False

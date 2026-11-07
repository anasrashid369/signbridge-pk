import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from signbridge.models.landmark_models import (  # noqa: E402
    build_forest,
    build_svm,
    full_proba,
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

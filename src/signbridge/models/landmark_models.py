"""Landmark-based classifiers: SVM, Random Forest and (added next) a PyTorch MLP.

Input: 63 normalised landmark values per hand (see data_scripts/features/landmarks.py).
Output: class probabilities over the 26 letters.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(C: float = 10.0, gamma="scale", seed: int = 42) -> Pipeline:
    """RBF SVM with feature scaling.

    probability=True turns on predict_proba, which AUC-ROC needs.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=seed)),
    ])


def build_forest(n_estimators: int = 300, max_depth: int | None = None,
                 seed: int = 42) -> RandomForestClassifier:
    """Random Forest. Tree models do not need feature scaling."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )


def full_proba(model, X: np.ndarray, n_classes: int = 26) -> np.ndarray:
    """predict_proba as an (n, n_classes) array.

    scikit-learn only returns columns for classes seen in training, so classes
    missing from the training data are filled with zeros.
    """
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out

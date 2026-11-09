"""Tests for copying the best landmark family to the final prediction names."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from signbridge.train_landmark import publish_best_family  # noqa: E402


def _write_family(folder: Path, name: str) -> None:
    (folder / f"landmark_{name}_val.npz").write_bytes(f"{name}-val".encode())
    (folder / f"landmark_{name}_test.npz").write_bytes(f"{name}-test".encode())


def _summary() -> dict[str, dict[str, float]]:
    return {
        "svm": {"val_accuracy": 0.90, "val_macro_f1": 0.89},
        "forest": {"val_accuracy": 0.95, "val_macro_f1": 0.94},
        "mlp": {"val_accuracy": 0.93, "val_macro_f1": 0.92},
    }


def test_best_family_files_are_copied_under_the_final_names(tmp_path: Path) -> None:
    for name in ("svm", "forest", "mlp"):
        _write_family(tmp_path, name)

    best = publish_best_family(_summary(), tmp_path, include_test=True)

    assert best == "forest"
    assert (tmp_path / "landmark_val.npz").read_bytes() == b"forest-val"
    assert (tmp_path / "landmark_test.npz").read_bytes() == b"forest-test"


def test_test_file_is_not_copied_without_include_test(tmp_path: Path) -> None:
    for name in ("svm", "forest", "mlp"):
        _write_family(tmp_path, name)

    publish_best_family(_summary(), tmp_path, include_test=False)

    assert (tmp_path / "landmark_val.npz").exists()
    assert not (tmp_path / "landmark_test.npz").exists()


def test_per_family_files_are_kept(tmp_path: Path) -> None:
    for name in ("svm", "forest", "mlp"):
        _write_family(tmp_path, name)

    publish_best_family(_summary(), tmp_path, include_test=True)

    for name in ("svm", "forest", "mlp"):
        assert (tmp_path / f"landmark_{name}_val.npz").exists()
        assert (tmp_path / f"landmark_{name}_test.npz").exists()
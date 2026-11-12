"""Tests for configs/app.yaml and its loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from signbridge.app_config import APP_CONFIG_PATH, load_app_settings
from signbridge.config import REPO_ROOT


def read_config() -> dict:
    """Return the real app config as a plain dict."""
    return yaml.safe_load(APP_CONFIG_PATH.read_text(encoding="utf-8"))


def write_config(tmp_path: Path, config: dict) -> Path:
    """Write a modified copy of the app config and return its path."""
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_real_config_loads() -> None:
    settings = load_app_settings()
    assert settings.default_model == "cnn"
    assert settings.image_size == 224
    assert settings.dropout == pytest.approx(0.2)
    assert settings.top_k == 3
    assert settings.max_file_bytes == 20 * 1024 * 1024
    assert settings.max_side == 4000
    assert settings.host == "127.0.0.1"
    assert settings.port == 7860
    assert settings.share is False


def test_paths_are_absolute_and_inside_the_repo() -> None:
    settings = load_app_settings()
    for path in (
        settings.cnn_checkpoint,
        settings.label_map,
        settings.landmark_model_dir,
        settings.hand_model,
    ):
        assert path.is_absolute()
        assert path.is_relative_to(REPO_ROOT)


def test_paths_match_the_documented_locations() -> None:
    settings = load_app_settings()
    assert settings.cnn_checkpoint == (REPO_ROOT / "models" / "cnn_best.pt").resolve()
    assert settings.label_map == (REPO_ROOT / "data" / "processed" / "label_map.json").resolve()
    assert settings.landmark_model_dir == (REPO_ROOT / "models" / "landmark").resolve()


def test_honesty_messages_are_present() -> None:
    settings = load_app_settings()
    assert set(settings.messages) == {"asl_only", "optimistic_scores", "hand_needed"}
    assert all(message.strip() for message in settings.messages.values())


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_app_settings(tmp_path / "missing.yaml")


def test_missing_section(tmp_path: Path) -> None:
    config = read_config()
    del config["limits"]
    with pytest.raises(ValueError, match="limits"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("value", ["svm", "CNN", ""])
def test_rejects_unknown_default_model(tmp_path: Path, value: str) -> None:
    config = read_config()
    config["model"]["default"] = value
    with pytest.raises(ValueError, match="model.default"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("value", [0, 27, -1, 3.5, True])
def test_rejects_bad_top_k(tmp_path: Path, value: object) -> None:
    config = read_config()
    config["model"]["top_k"] = value
    with pytest.raises(ValueError, match="model.top_k"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("value", [0, -5, 1.5])
def test_rejects_bad_image_size(tmp_path: Path, value: object) -> None:
    config = read_config()
    config["model"]["image_size"] = value
    with pytest.raises(ValueError, match="model.image_size"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("value", [1.0, -0.1, "0.2"])
def test_rejects_bad_dropout(tmp_path: Path, value: object) -> None:
    config = read_config()
    config["model"]["dropout"] = value
    with pytest.raises(ValueError, match="model.dropout"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("key", ["max_file_mb", "max_side"])
def test_rejects_bad_limits(tmp_path: Path, key: str) -> None:
    config = read_config()
    config["limits"][key] = 0
    with pytest.raises(ValueError, match=f"limits.{key}"):
        load_app_settings(write_config(tmp_path, config))


@pytest.mark.parametrize("value", [0, 70000])
def test_rejects_bad_port(tmp_path: Path, value: int) -> None:
    config = read_config()
    config["server"]["port"] = value
    with pytest.raises(ValueError, match="server.port"):
        load_app_settings(write_config(tmp_path, config))


def test_rejects_non_boolean_share(tmp_path: Path) -> None:
    config = read_config()
    config["server"]["share"] = "no"
    with pytest.raises(ValueError, match="server.share"):
        load_app_settings(write_config(tmp_path, config))


def test_rejects_blank_message(tmp_path: Path) -> None:
    config = read_config()
    config["messages"]["asl_only"] = "  "
    with pytest.raises(ValueError, match="messages.asl_only"):
        load_app_settings(write_config(tmp_path, config))

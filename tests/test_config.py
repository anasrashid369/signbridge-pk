"""Tests for the shared config loader."""

from pathlib import Path

import pytest

from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path

REQUIRED_SECTIONS = {"seed", "paths", "dataset", "image", "quality"}


def test_repo_config_has_required_sections() -> None:
    """The committed configs/data.yaml contains every top-level section scripts rely on."""
    config = load_config(DEFAULT_CONFIG_PATH)
    assert REQUIRED_SECTIONS <= config.keys()


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    """A YAML list at the top level is rejected."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_path_joins_root(tmp_path: Path) -> None:
    """Relative paths are resolved against the given root."""
    config = {"paths": {"raw_dir": "data/raw"}}
    assert resolve_path(config, "raw_dir", root=tmp_path) == (tmp_path / "data" / "raw").resolve()


def test_resolve_path_missing_key_raises() -> None:
    """An unknown path key raises KeyError naming the key."""
    with pytest.raises(KeyError, match="paths.unknown"):
        resolve_path({"paths": {}}, "unknown")

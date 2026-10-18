"""Tests for argument handling in data_scripts/download_data.py."""

import logging
from pathlib import Path

import pytest

from data_scripts import download_data


def write_config(tmp_path: Path) -> Path:
    """Write a minimal config whose paths point inside tmp_path."""
    path = tmp_path / "data.yaml"
    path.write_text(
        "dataset:\n  kaggle_slug: owner/default-set\n"
        "paths:\n  asl_raw_dir: raw/asl\n  psl_raw_dir: raw/psl\n",
        encoding="utf-8",
    )
    return path


def test_parse_args_defaults() -> None:
    """Without flags, the slug and destination come from the config later."""
    args = download_data.parse_args([])
    assert args.dataset is None
    assert args.dest is None
    assert args.dry_run is False


def test_parse_args_overrides(tmp_path: Path) -> None:
    """Flags are parsed into the expected types."""
    args = download_data.parse_args(["--dataset", "a/b", "--dest", str(tmp_path), "--dry-run"])
    assert args.dataset == "a/b"
    assert args.dest == tmp_path
    assert args.dry_run is True


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """--help prints usage and exits with code 0."""
    with pytest.raises(SystemExit) as exit_info:
        download_data.parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "--dataset" in capsys.readouterr().out


@pytest.mark.parametrize("slug", ["no-slash", "owner/", "/name", "a/b/c", ""])
def test_validate_slug_rejects_bad_slugs(slug: str) -> None:
    """Slugs that are not owner/name are rejected."""
    with pytest.raises(ValueError):
        download_data.validate_slug(slug)


def test_dry_run_uses_dataset_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A dry run logs the overriding slug and never downloads."""
    def fail_download(slug: str, dest: Path) -> None:
        raise AssertionError("download must not run in a dry run")

    monkeypatch.setattr(download_data, "download_dataset", fail_download)
    config = write_config(tmp_path)
    with caplog.at_level(logging.INFO):
        code = download_data.main(["--config", str(config), "--dataset", "x/y", "--dry-run"])
    assert code == 0
    assert "x/y" in caplog.text
    assert "owner/default-set" not in caplog.text


def test_missing_config_returns_error(tmp_path: Path) -> None:
    """A missing config file gives exit code 1 instead of a traceback."""
    assert download_data.main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]) == 1

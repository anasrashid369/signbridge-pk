"""Smoke test that the Gradio app builds. Skipped when Gradio is not installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

from signbridge.app_config import load_app_settings  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def load_app_module() -> ModuleType:
    """Import app/app.py by path, because app/ is not a package."""
    spec = importlib.util.spec_from_file_location("signbridge_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_app_returns_blocks() -> None:
    module = load_app_module()
    demo = module.build_app(load_app_settings())
    assert isinstance(demo, gr.Blocks)


def test_honesty_markdown_lists_every_message() -> None:
    module = load_app_module()
    settings = load_app_settings()
    text = module.honesty_markdown(settings)
    assert "About these numbers" in text
    for message in settings.messages.values():
        assert message in text

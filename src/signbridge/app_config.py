"""Read ``configs/app.yaml`` for the Gradio app and the predict CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signbridge.config import REPO_ROOT, load_config, resolve_path

APP_CONFIG_PATH = REPO_ROOT / "configs" / "app.yaml"
MODEL_CHOICES = ("cnn", "landmark", "both")
MAX_CLASSES = 26  # A-Z; a larger top_k could not be filled


@dataclass(frozen=True)
class AppSettings:
    """Settings shared by the Gradio app and the command-line predictor."""

    cnn_checkpoint: Path
    label_map: Path
    landmark_model_dir: Path
    hand_model: Path
    default_model: str
    image_size: int
    dropout: float
    top_k: int
    max_file_bytes: int
    max_side: int
    host: str
    port: int
    share: bool
    messages: dict[str, str]


def load_app_settings(path: str | Path = APP_CONFIG_PATH) -> AppSettings:
    """Read and validate the app settings.

    Args:
        path: Path to the YAML file. Defaults to ``configs/app.yaml``.

    Returns:
        The validated settings, with the four model paths made absolute.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a section is missing or a value is out of range.
    """
    config = load_config(Path(path))
    model = _mapping(config, "model")
    limits = _mapping(config, "limits")
    server = _mapping(config, "server")
    messages = _messages(config)

    top_k = _positive_int(model, "top_k", "model", maximum=MAX_CLASSES)
    host = server.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"server.host must be a non-empty string, got {host!r}")
    port = _positive_int(server, "port", "server", maximum=65535)
    share = server.get("share")
    if not isinstance(share, bool):
        raise ValueError(f"server.share must be true or false, got {share!r}")

    return AppSettings(
        cnn_checkpoint=resolve_path(config, "cnn_checkpoint"),
        label_map=resolve_path(config, "label_map"),
        landmark_model_dir=resolve_path(config, "landmark_model_dir"),
        hand_model=resolve_path(config, "hand_model"),
        default_model=_choice(model, "default", MODEL_CHOICES, "model"),
        image_size=_positive_int(model, "image_size", "model"),
        dropout=_fraction(model, "dropout", "model"),
        top_k=top_k,
        max_file_bytes=_positive_int(limits, "max_file_mb", "limits") * 1024 * 1024,
        max_side=_positive_int(limits, "max_side", "limits"),
        host=host,
        port=port,
        share=share,
        messages=messages,
    )


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``config[key]`` as a mapping or raise a clear error."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a '{key}' mapping")
    return value


def _messages(config: dict[str, Any]) -> dict[str, str]:
    """Return the honesty messages, all non-empty strings."""
    block = _mapping(config, "messages")
    messages = {}
    for key, value in block.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"messages.{key} must be a non-empty string, got {value!r}")
        messages[str(key)] = value
    if not messages:
        raise ValueError("Config must contain at least one message")
    return messages


def _value(block: dict[str, Any], key: str, section: str) -> Any:
    """Return ``block[key]`` or raise an error naming the missing key."""
    if key not in block:
        raise ValueError(f"Config is missing {section}.{key}")
    return block[key]


def _positive_int(
    block: dict[str, Any], key: str, section: str, maximum: int | None = None
) -> int:
    """Return a positive integer, rejecting booleans and out-of-range values."""
    value = _value(block, key, section)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{section}.{key} must be at most {maximum}, got {value}")
    return value


def _fraction(block: dict[str, Any], key: str, section: str) -> float:
    """Return a probability between 0 (inclusive) and 1 (exclusive)."""
    value = _value(block, key, section)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be a number, got {value!r}")
    if not 0 <= value < 1:
        raise ValueError(f"{section}.{key} must be at least 0 and below 1, got {value!r}")
    return float(value)


def _choice(block: dict[str, Any], key: str, choices: tuple[str, ...], section: str) -> str:
    """Return a value that must be one of ``choices``."""
    value = _value(block, key, section)
    if value not in choices:
        raise ValueError(f"{section}.{key} must be one of {choices}, got {value!r}")
    return str(value)

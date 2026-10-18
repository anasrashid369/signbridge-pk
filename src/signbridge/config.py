"""Load the shared YAML configuration used by the data scripts."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read a YAML config file and return its top-level mapping.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a mapping at the top level.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping at the top level: {path}")
    return config


def resolve_path(config: dict[str, Any], key: str, root: Path = REPO_ROOT) -> Path:
    """Return the absolute path stored under ``paths.<key>`` in the config.

    Raises:
        KeyError: If ``paths.<key>`` is missing.
    """
    try:
        relative = config["paths"][key]
    except KeyError as error:
        raise KeyError(f"Missing config key: paths.{key}") from error
    return (root / relative).resolve()

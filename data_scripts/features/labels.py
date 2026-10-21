"""Label encoding: map each class name to a number and save it as label_map.json.

Run from the repository root:
    python data_scripts/features/labels.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import string
from pathlib import Path

from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

LETTERS = tuple(string.ascii_uppercase)  # A to Z
SPECIAL_CLASSES = ("DELETE", "NOTHING", "SPACE")  # extra classes in the ASL baseline


def build_label_map(include_special: bool) -> dict[str, int]:
    """Return a mapping from class name to class index.

    Letters come first (A=0 ... Z=25). Special classes, if included, follow in
    alphabetical order, so the letter indices never change between runs.
    """
    names = LETTERS + SPECIAL_CLASSES if include_special else LETTERS
    return {name: index for index, name in enumerate(names)}


def encode_label(label: str, label_map: dict[str, int]) -> int:
    """Return the class index for a label.

    Raises:
        KeyError: If the label is not in the map.
    """
    try:
        return label_map[label]
    except KeyError as error:
        raise KeyError(f"Unknown label: {label!r}") from error


def save_label_map(label_map: dict[str, int], path: Path) -> None:
    """Write the label map to a JSON file, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)
        handle.write("\n")


def load_label_map(path: Path) -> dict[str, int]:
    """Read a label map from a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create data/processed/label_map.json.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--include-special",
        action="store_true",
        help="Also include DELETE, NOTHING and SPACE (29 classes instead of 26).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config error."""
    args = parse_args(argv)
    setup_logging()
    try:
        config = load_config(args.config)
        destination = resolve_path(config, "label_map")
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    label_map = build_label_map(args.include_special)
    save_label_map(label_map, destination)
    logger.info("Saved %d classes to %s", len(label_map), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

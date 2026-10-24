"""Build data/processed/manifest.csv from the team's PSL photos with a person-wise split.

File names follow the collection protocol:
    {person_id}_{label}_{lighting}_{background}_{index}.jpg

Run from the repository root:
    python data_scripts/build_manifest.py --help
"""

import argparse
import csv
import logging
import random
import re
from pathlib import Path
from typing import Any

from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLUMNS = ["path", "label", "person_id", "split", "lighting", "background"]
PERSON_ID_PATTERN = re.compile(r"p\d{2,}")
LABEL_PATTERN = re.compile(r"[A-Za-z0-9]+")
MIN_PERSONS = 3  # train, validation and test each need at least one person


def parse_filename(
    name: str, lighting_values: set[str], background_values: set[str]
) -> dict[str, str]:
    """Read person_id, label, lighting and background from a protocol file name.

    Raises:
        ValueError: If the name does not follow the protocol.
    """
    parts = Path(name).stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"Expected 5 parts separated by '_', got {len(parts)}: {name}")
    person_id, label, lighting, background, index = parts
    if not PERSON_ID_PATTERN.fullmatch(person_id):
        raise ValueError(f"Bad person_id {person_id!r} in {name}")
    if not LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"Bad label {label!r} in {name}")
    if lighting not in lighting_values:
        raise ValueError(f"Unknown lighting {lighting!r} in {name}")
    if background not in background_values:
        raise ValueError(f"Unknown background {background!r} in {name}")
    if not index.isdigit():
        raise ValueError(f"Index must be a number, got {index!r} in {name}")
    return {
        "label": label,
        "person_id": person_id,
        "lighting": lighting,
        "background": background,
    }


def assign_splits(
    person_ids: list[str], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    """Assign whole people to train, val or test. The same seed gives the same split.

    Raises:
        ValueError: If there are fewer than MIN_PERSONS people or a fraction is invalid.
    """
    people = sorted(set(person_ids))
    if len(people) < MIN_PERSONS:
        raise ValueError(
            f"Need at least {MIN_PERSONS} people for a person-wise split, got {len(people)}"
        )
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1")
    random.Random(seed).shuffle(people)
    n_test = max(1, round(len(people) * test_fraction))
    n_val = max(1, round(len(people) * val_fraction))
    splits = {}
    for index, person in enumerate(people):
        if index < n_test:
            splits[person] = "test"
        elif index < n_test + n_val:
            splits[person] = "val"
        else:
            splits[person] = "train"
    return splits


def manifest_path(path: Path, root: Path) -> str:
    """Return the path relative to root, or the absolute path if the file is outside root."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_images(
    psl_dir: Path, lighting_values: set[str], background_values: set[str], root: Path
) -> list[dict[str, str]]:
    """List the PSL images as manifest rows without a split. Bad names are logged and skipped."""
    if not psl_dir.is_dir():
        raise FileNotFoundError(f"PSL folder not found: {psl_dir}")
    rows = []
    for path in sorted(psl_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            row = parse_filename(path.name, lighting_values, background_values)
        except ValueError as error:
            logger.warning("Skipping %s", error)
            continue
        row["path"] = manifest_path(path, root)
        rows.append(row)
    if not rows:
        raise ValueError(f"No correctly named images found in {psl_dir}")
    return rows


def check_no_leakage(rows: list[dict[str, str]]) -> None:
    """Raise ValueError if any person_id appears in more than one split."""
    splits_by_person: dict[str, set[str]] = {}
    for row in rows:
        splits_by_person.setdefault(row["person_id"], set()).add(row["split"])
    leaked = sorted(person for person, splits in splits_by_person.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"These people appear in more than one split: {leaked}")


def build_rows(
    config: dict[str, Any], psl_dir: Path, root: Path = REPO_ROOT
) -> list[dict[str, str]]:
    """Scan the PSL folder, assign person-wise splits and return the manifest rows."""
    settings = config["manifest"]
    rows = scan_images(
        psl_dir, set(settings["lighting_values"]), set(settings["background_values"]), root
    )
    splits = assign_splits(
        [row["person_id"] for row in rows],
        settings["val_fraction"],
        settings["test_fraction"],
        config["seed"],
    )
    for row in rows:
        row["split"] = splits[row["person_id"]]
    check_no_leakage(rows)
    return rows


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    """Write the manifest CSV with the protocol columns."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def log_summary(rows: list[dict[str, str]]) -> None:
    """Log image and person counts per split."""
    for split in ("train", "val", "test"):
        in_split = [row for row in rows if row["split"] == split]
        people = {row["person_id"] for row in in_split}
        logger.info("%s: %d images from %d people", split, len(in_split), len(people))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the PSL manifest with a person-wise train/val/test split."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--psl-dir", type=Path, help="PSL image folder. Default: paths.psl_raw_dir in the config."
    )
    parser.add_argument(
        "--output", type=Path, help="Manifest file to write. Default: paths.manifest in the config."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Resolve settings, build the manifest and write it."""
    config = load_config(args.config)
    psl_dir = args.psl_dir or resolve_path(config, "psl_raw_dir")
    output = args.output or resolve_path(config, "manifest")
    rows = build_rows(config, psl_dir)
    write_manifest(rows, output)
    log_summary(rows)
    logger.info("Wrote %d rows to %s", len(rows), output)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config, folder or naming error."""
    args = parse_args(argv)
    setup_logging()
    try:
        run(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

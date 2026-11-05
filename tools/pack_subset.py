"""Create a zip with only the images named in a split file."""

from __future__ import annotations

import argparse
import csv
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zip only the images listed in a split file, keeping <class>/<file> paths."
    )
    parser.add_argument("--split-file", type=Path, required=True, help="Split CSV to read.")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Folder the split paths are relative to.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination zip, preferably outside the repo.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=DEFAULT_SPLITS,
        help="Split names to include (default: train val test).",
    )
    return parser.parse_args(argv)


def collect_paths(split_file: Path, splits: set[str]) -> list[str]:
    """Return the relative image paths whose split is selected."""
    with split_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split file is missing columns: {sorted(missing)}")
        return [row["path"] for row in reader if row["split"] in splits]


def write_subset(split_file: Path, image_root: Path, output: Path, splits: set[str]) -> int:
    """Write the selected images to a zip and return the number of files added."""
    relative_paths = collect_paths(split_file, splits)
    if not relative_paths:
        raise ValueError(f"No rows for splits {sorted(splits)} in {split_file}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in relative_paths:
            source = image_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Image not found: {source}")
            archive.write(source, arcname=Path(relative).as_posix())
    return len(relative_paths)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a file or CSV error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        count = write_subset(args.split_file, args.image_root, args.output, set(args.splits))
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Wrote %d images to %s", count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

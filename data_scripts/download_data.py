"""Download the ASL alphabet baseline dataset from Kaggle.

Run from the repository root:
    python data_scripts/download_data.py --help
"""

import argparse
import logging
from pathlib import Path

from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the ASL alphabet baseline dataset from Kaggle."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--dataset", help="Kaggle slug owner/name. Default: dataset.kaggle_slug in the config."
    )
    parser.add_argument(
        "--dest", type=Path, help="Download folder. Default: paths.asl_raw_dir in the config."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be downloaded, then stop."
    )
    return parser.parse_args(argv)


def validate_slug(slug: str) -> str:
    """Return the slug if it has the form owner/name, otherwise raise ValueError."""
    owner, separator, name = slug.partition("/")
    if not owner or not separator or not name or "/" in name:
        raise ValueError(f"Dataset slug must look like 'owner/name', got: {slug!r}")
    return slug


def download_dataset(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset into dest. Requires Kaggle credentials."""
    # Imported here because importing kaggle starts authentication and prints instructions.
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)


def run(args: argparse.Namespace) -> None:
    """Resolve settings from the config and arguments, then download."""
    config = load_config(args.config)
    slug = validate_slug(args.dataset or config["dataset"]["kaggle_slug"])
    dest = args.dest or resolve_path(config, "asl_raw_dir")
    logger.info("Dataset: %s (ASL baseline data, not PSL)", slug)
    logger.info("Destination: %s", dest)
    if args.dry_run:
        logger.info("Dry run: nothing downloaded.")
    else:
        download_dataset(slug, dest)
        logger.info("Download complete.")
    psl_dir = resolve_path(config, "psl_raw_dir")
    logger.info("Place the team's PSL images under %s. Never commit them.", psl_dir)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or argument error."""
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

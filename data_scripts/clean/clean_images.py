"""Clean an image folder: measure quality, flag problems and write the cleaning report.

Cleaning steps, in order (each step counts what is still kept after it):
    1. Unreadable files and images without a class label are removed.
    2. Exact duplicates (identical file hash) are removed.
    3. Near-duplicates (almost identical image hash, same class) are removed.
    4. Quality outliers (too small, blurry, too dark, too bright) are removed.

Nothing is deleted from disk. The scripts only flag images and count them.

Run from the repository root:
    python -m data_scripts.clean.clean_images --source asl --limit-per-class 500
    python -m data_scripts.clean.clean_images --help
"""

import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_scripts.build_manifest import IMAGE_SUFFIXES, manifest_path, parse_filename
from data_scripts.clean.duplicates import find_exact_duplicates, find_near_duplicates
from data_scripts.clean.quality import READ_ERRORS, measure_image, quality_flags
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SOURCES = ("asl", "psl")
SOURCE_NAMES = {"asl": "ASL baseline", "psl": "PSL"}
QUALITY_FLAGS = ("too_small", "blurry", "too_dark", "too_bright")
REPORT_COLUMNS = [
    "path",
    "source",
    "label",
    "person_id",
    "width",
    "height",
    "sha256",
    "blur_score",
    "brightness",
    "corrupt",
    "no_label",
    "exact_duplicate",
    "near_duplicate",
    "duplicate_of",
    *QUALITY_FLAGS,
    "keep",
]
# Flags that remove an image at each step. Counts after step k include the flags of steps 1 to k.
STEP_FLAGS = [
    ("corrupt", "no_label"),
    ("exact_duplicate",),
    ("near_duplicate",),
    QUALITY_FLAGS,
]
STEP_COLUMNS = [
    "after_unreadable_or_unlabelled",
    "after_exact_duplicates",
    "after_near_duplicates",
    "after_quality_outliers",
]
SUMMARY_COLUMNS = ["source", "label", "found", *STEP_COLUMNS]
LOG_EVERY = 1000  # log progress after this many images


def resolve_image_path(path_text: str, root: Path = REPO_ROOT) -> Path:
    """Turn a path stored in the report back into a file path."""
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def new_record(path_text: str, source: str) -> dict[str, Any]:
    """Create a report row for one image with every flag set to 0."""
    record: dict[str, Any] = dict.fromkeys(REPORT_COLUMNS, "")
    record.update(path=path_text, source=source, label="", person_id="")
    for flag in ("corrupt", "no_label", "exact_duplicate", "near_duplicate", *QUALITY_FLAGS):
        record[flag] = 0
    return record


def find_records(
    source: str, directory: Path, config: dict[str, Any], root: Path = REPO_ROOT
) -> list[dict[str, Any]]:
    """List the images of one source as report rows, labelled from folder or file names.

    ASL images are labelled by their folder name. PSL images are labelled from the file name
    pattern in the collection protocol; files that do not follow it get no label.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If the source is unknown or the folder holds no images.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Image folder not found: {directory}")
    base = directory
    train_folder = directory / config["dataset"]["asl_train_folder"]
    if source == "asl" and train_folder.is_dir():
        base = train_folder
    lighting = set(config["manifest"]["lighting_values"])
    background = set(config["manifest"]["background_values"])
    records = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        record = new_record(manifest_path(path, root), source)
        if source == "asl":
            record["label"] = path.parent.name if path.parent != base else ""
        else:
            try:
                fields = parse_filename(path.name, lighting, background)
            except ValueError as error:
                logger.warning("No label: %s", error)
            else:
                record["label"] = fields["label"]
                record["person_id"] = fields["person_id"]
        record["no_label"] = int(record["label"] == "")
        records.append(record)
    if not records:
        raise ValueError(f"No images found in {base}")
    return records


def sample_per_class(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Keep a reproducible random sample of at most limit images per class (0 keeps all)."""
    if limit <= 0:
        return records
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    chosen = []
    for label in sorted(by_label):
        group = by_label[label]
        chosen.extend(random.Random(seed).sample(group, min(limit, len(group))))
    return sorted(chosen, key=lambda record: record["path"])


def measure_records(records: list[dict[str, Any]], root: Path = REPO_ROOT) -> dict[str, int]:
    """Measure every image, fill in the report rows and flag unreadable files.

    Returns:
        A mapping of image path to its 64-bit difference hash (kept out of the report).
    """
    hashes: dict[str, int] = {}
    for count, record in enumerate(records, start=1):
        try:
            stats = measure_image(resolve_image_path(record["path"], root))
        except READ_ERRORS as error:
            record["corrupt"] = 1
            logger.warning("Unreadable image %s: %s", record["path"], error)
        else:
            record.update(
                width=stats.width,
                height=stats.height,
                sha256=stats.sha256,
                blur_score=round(stats.blur_score, 3),
                brightness=round(stats.brightness, 3),
            )
            hashes[record["path"]] = stats.dhash
        if count % LOG_EVERY == 0:
            logger.info("Measured %d of %d images", count, len(records))
    return hashes


def flag_duplicates(
    records: list[dict[str, Any]], hashes: dict[str, int], max_distance: int
) -> None:
    """Flag exact duplicates across all classes and near-duplicates within each class.

    The first image in path order is kept; later copies are flagged and point to it.
    """
    by_path = {record["path"]: record for record in records}
    usable = [record for record in records if not (record["corrupt"] or record["no_label"])]
    exact = find_exact_duplicates([(record["path"], record["sha256"]) for record in usable])
    for path, kept in exact.items():
        by_path[path].update(exact_duplicate=1, duplicate_of=kept)
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for record in usable:
        if not record["exact_duplicate"]:
            groups[record["label"]].append((record["path"], hashes[record["path"]]))
    for items in groups.values():
        for path, kept in find_near_duplicates(items, max_distance).items():
            by_path[path].update(near_duplicate=1, duplicate_of=kept)


def flag_quality(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> None:
    """Flag readable images that are too small, blurry, too dark or too bright."""
    for record in records:
        if record["corrupt"]:
            continue
        record.update(
            quality_flags(
                int(record["width"]),
                int(record["height"]),
                float(record["blur_score"]),
                float(record["brightness"]),
                thresholds,
            )
        )


def mark_keep(records: list[dict[str, Any]]) -> None:
    """Set keep to 1 for images that no cleaning step flagged, otherwise 0."""
    flags = [flag for step in STEP_FLAGS for flag in step]
    for record in records:
        record["keep"] = int(not any(record[flag] for flag in flags))


def summarize(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Count images per class that remain after each cleaning step."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    rows = []
    for label in sorted(by_label):
        group = by_label[label]
        row: dict[str, Any] = {"source": source, "label": label or "(no label)"}
        row["found"] = len(group)
        removing: list[str] = []
        for column, flags in zip(STEP_COLUMNS, STEP_FLAGS, strict=True):
            removing.extend(flags)
            row[column] = sum(1 for record in group if not any(record[f] for f in removing))
        rows.append(row)
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into rows, or return an empty list if it does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str], source: str) -> None:
    """Write rows to a CSV file, replacing earlier rows of the same source but keeping others."""
    kept = [row for row in read_rows(path) if row["source"] != source]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)


def write_latex_table(summary_path: Path, output: Path) -> None:
    """Write a LaTeX table of the total counts per dataset for the report."""
    totals: dict[str, dict[str, int]] = {}
    for row in read_rows(summary_path):
        entry = totals.setdefault(row["source"], dict.fromkeys(["found", *STEP_COLUMNS], 0))
        for column in entry:
            entry[column] += int(row[column])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Number of images still kept after each cleaning step.}",
        r"\label{tab:cleaning-counts}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Found & Readable, labelled & No exact duplicates & No near-duplicates "
        r"& Quality ok \\",
        r"\midrule",
    ]
    for source in SOURCES:
        if source in totals:
            values = " & ".join(
                f"{totals[source][c]:,}".replace(",", "{,}") for c in totals[source]
            )
            lines.append(f"{SOURCE_NAMES[source]} & {values} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Flag unreadable, duplicate and low-quality images; write the cleaning report."
    )
    parser.add_argument("--source", required=True, choices=SOURCES, help="Dataset to clean.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Image folder. Default: paths.asl_raw_dir or paths.psl_raw_dir in the config.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Random sample of this many images per class, using the config seed. 0 uses all.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Clean one dataset and write the report, the summary and the LaTeX table."""
    config = load_config(args.config)
    thresholds = config["quality"]
    directory = args.dir or resolve_path(config, f"{args.source}_raw_dir")
    records = find_records(args.source, directory, config)
    records = sample_per_class(records, args.limit_per_class, config["seed"])
    logger.info("Cleaning %d images from the %s dataset", len(records), SOURCE_NAMES[args.source])
    hashes = measure_records(records)
    flag_duplicates(records, hashes, thresholds["near_duplicate_distance"])
    flag_quality(records, thresholds)
    mark_keep(records)
    summary = summarize(records, args.source)
    summary_path = resolve_path(config, "cleaning_summary")
    write_rows(resolve_path(config, "cleaning_report"), records, REPORT_COLUMNS, args.source)
    write_rows(summary_path, summary, SUMMARY_COLUMNS, args.source)
    write_latex_table(summary_path, resolve_path(config, "cleaning_table"))
    kept = sum(record["keep"] for record in records)
    logger.info("Kept %d of %d images after all cleaning steps", kept, len(records))


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

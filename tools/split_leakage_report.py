"""Report near-duplicate leakage from the train split into validation and test.

For every validation and test image the tool computes the 64-bit dHash (the same
function the cleaning scripts use) and finds the smallest Hamming distance to a
training image of the same class. A small distance means the two images look
almost identical, so the test score can be optimistic.

Run from the repository root, writing outside the repository:
    python -m tools.split_leakage_report --split-file data/processed/split_v1.csv \
        --image-root PATH/TO/images --output-dir PATH/OUTSIDE/THE/REPO

The default output folder is a scratch folder under the system temp directory,
never the repository. The command writes a text summary and a CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data_scripts.clean.duplicates import hamming_distances
from data_scripts.clean.quality import READ_ERRORS, difference_hash
from signbridge.config import REPO_ROOT

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
THRESHOLDS = (4, 6, 8, 10)
REQUIRED_COLUMNS = ("path", "label", "split")
CSV_COLUMNS = [
    "scope",
    "split",
    "images",
    "min",
    "p5",
    "median",
    "frac_le_4",
    "frac_le_6",
    "frac_le_8",
    "frac_le_10",
]
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "signbridge_split_leakage"


def read_split(path: Path) -> list[dict[str, str]]:
    """Read the split CSV.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Split file is missing columns: {missing}")
        return list(reader)


def image_dhash(path: Path) -> int:
    """Return the 64-bit dHash of one image using the cleaning helper.

    Raises:
        OSError, ValueError, SyntaxError: If the file cannot be read as an image.
    """
    with Image.open(path) as image:
        image.load()
        gray = np.asarray(image.convert("L"))
    return difference_hash(gray)


def load_hash_cache(path: Path) -> dict[str, int]:
    """Load a path to dHash cache written by :func:`save_hash_cache`."""
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {str(key): int(value) for key, value in zip(data["paths"], data["hashes"])}


def save_hash_cache(path: Path, cache: dict[str, int]) -> None:
    """Write the path to dHash cache as a compressed npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    np.savez_compressed(
        path,
        paths=np.asarray(keys, dtype=str),
        hashes=np.asarray([cache[key] for key in keys], dtype=np.uint64),
    )


def hashes_for(
    image_root: Path, paths: list[str], cache: dict[str, int]
) -> dict[str, int]:
    """Hash every path that is not cached yet and return the hashes that exist.

    Unreadable images are logged and skipped.
    """
    missing = [path for path in paths if path not in cache]
    for number, relative in enumerate(missing, start=1):
        try:
            cache[relative] = image_dhash(image_root / relative)
        except READ_ERRORS as error:
            logger.warning("Unreadable image %s: %s", relative, error)
        if number % 500 == 0:
            logger.info("  hashed %d of %d", number, len(missing))
    return {path: cache[path] for path in paths if path in cache}


def summarise(distances: list[int]) -> dict[str, float | int]:
    """Return count, minimum, 5th percentile, median and threshold fractions."""
    values = np.asarray(distances, dtype=np.int64)
    record: dict[str, float | int] = {
        "images": int(values.size),
        "min": int(values.min()),
        "p5": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
    }
    for threshold in THRESHOLDS:
        record[f"frac_le_{threshold}"] = float(np.mean(values <= threshold))
    return record


def build_report(
    rows: list[dict[str, str]], hashes: dict[str, int]
) -> list[dict[str, float | int | str]]:
    """Compute the overall and per-class leakage rows for validation and test."""
    by_label: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )
    for row in rows:
        if row["path"] in hashes:
            by_label[row["label"]][row["split"]].append(hashes[row["path"]])
    overall: dict[str, list[int]] = {"val": [], "test": []}
    records: list[dict[str, float | int | str]] = []
    for label in sorted(by_label):
        train = np.asarray(by_label[label]["train"], dtype=np.uint64)
        for split in ("val", "test"):
            eval_hashes = by_label[label][split]
            if train.size == 0 or not eval_hashes:
                continue
            distances = [
                int(hamming_distances(train, int(value)).min()) for value in eval_hashes
            ]
            overall[split].extend(distances)
            records.append({"scope": label, "split": split, **summarise(distances)})
    summary = [
        {"scope": "overall", "split": split, **summarise(overall[split])}
        for split in ("val", "test")
        if overall[split]
    ]
    return summary + records


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    """Write the report rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def format_report(
    records: list[dict[str, float | int | str]],
    split_file: Path,
    image_root: Path,
    hashed: int,
    total: int,
) -> str:
    """Return the text summary."""
    lines = [
        f"Split file: {split_file}",
        f"Image root: {image_root}",
        f"Images hashed: {hashed} of {total}",
        "",
        f"{'scope':8s} {'split':5s} {'n':>5s} {'min':>4s} {'p5':>5s} {'median':>6s} "
        f"{'<=4':>6s} {'<=6':>6s} {'<=8':>6s} {'<=10':>6s}",
    ]
    for record in records:
        lines.append(
            f"{str(record['scope']):8s} {str(record['split']):5s} "
            f"{int(record['images']):5d} {int(record['min']):4d} "
            f"{float(record['p5']):5.1f} {float(record['median']):6.1f} "
            f"{float(record['frac_le_4']):6.1%} {float(record['frac_le_6']):6.1%} "
            f"{float(record['frac_le_8']):6.1%} {float(record['frac_le_10']):6.1%}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report near-duplicate leakage from train into val and test."
    )
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for the text and CSV reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--hash-cache",
        type=Path,
        help="Optional npz cache of image dHashes, reused across splits.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a bad path or split file."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = REPO_ROOT.resolve()
    try:
        output_dir = args.output_dir.resolve()
        if output_dir.is_relative_to(root):
            raise ValueError(f"Refusing to write inside the repository ({root}).")
        if args.hash_cache and args.hash_cache.resolve().is_relative_to(root):
            raise ValueError(f"Refusing to write the hash cache inside the repository ({root}).")
        rows = read_split(args.split_file)
        cache = load_hash_cache(args.hash_cache) if args.hash_cache else {}
        paths = sorted({row["path"] for row in rows})
        hashes = hashes_for(args.image_root, paths, cache)
        if args.hash_cache:
            save_hash_cache(args.hash_cache, cache)
        records = build_report(rows, hashes)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    stem = args.split_file.stem
    csv_path = output_dir / f"{stem}_leakage.csv"
    text_path = output_dir / f"{stem}_leakage.txt"
    write_csv(csv_path, records)
    text = format_report(records, args.split_file, args.image_root, len(hashes), len(paths))
    text_path.write_text(text, encoding="utf-8")
    logger.info("%s", text)
    logger.info("Wrote %s and %s", csv_path, text_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

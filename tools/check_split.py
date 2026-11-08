"""Validate the shared split file before any training run.

Run from the repository root:
    python -m tools.check_split --split-file data/processed/split_v1.csv \
        --image-root PATH/TO/asl_images

The command prints the split counts, the per-class counts and every problem it
finds. It exits with 0 only when the file matches the contract in WORKFLOW.md:
columns ``path``, ``label``, ``group_id`` and ``split``; split values ``train``,
``val`` and ``test``; labels from ``label_map.json``; existing images; unique
paths; and no near-duplicate group in more than one split.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from signbridge.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("path", "label", "group_id", "split")
VALID_SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass
class SplitReport:
    """Counts and problems found while validating one split file."""

    counts: dict[str, int]
    ratios: dict[str, float]
    per_class: dict[str, dict[str, int]]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no problem was found."""
        return not self.problems


def _ratio_problems(
    counts: dict[str, int], total: int, tolerance: float, prefix: str
) -> list[str]:
    """Return a message for every split whose share is too far from its target."""
    problems: list[str] = []
    for split, target in TARGET_RATIOS.items():
        share = counts.get(split, 0) / total
        if abs(share - target) > tolerance:
            problems.append(
                f"{prefix}{split} share is {share:.1%}, target is {target:.0%} "
                f"(tolerance {tolerance:.0%})"
            )
    return problems


def validate_rows(
    rows: list[dict[str, str]],
    label_map: dict[str, int],
    image_root: Path,
    tolerance: float = 0.05,
    class_tolerance: float = 0.10,
) -> SplitReport:
    """Validate split rows and return the counts and every problem found.

    Args:
        rows: CSV rows with the contract columns and an optional ``_line`` key.
        label_map: Class name to index map from ``label_map.json``.
        image_root: Folder that the ``path`` column is relative to.
        tolerance: Allowed absolute difference from 70/15/15 overall.
        class_tolerance: Allowed absolute difference from 70/15/15 per class.
    """
    counts: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = {}
    group_splits: dict[str, set[str]] = {}
    seen_paths: dict[str, int] = {}
    problems: list[str] = []
    check_images = image_root.is_dir()
    if not check_images:
        problems.append(f"Image root not found: {image_root}")

    for row in rows:
        line = row.get("_line", "?")
        path = row.get("path", "").strip()
        label = row.get("label", "").strip()
        group = row.get("group_id", "").strip()
        split = row.get("split", "").strip()
        location = f"line {line}"
        if not path:
            problems.append(f"{location}: empty path")
        else:
            normalized = path.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            if PurePosixPath(normalized).is_absolute() or ".." in parts:
                problems.append(f"{location}: path must stay under the image root: {path!r}")
            if path in seen_paths:
                problems.append(
                    f"{location}: duplicate path {path!r}, first seen on line {seen_paths[path]}"
                )
            else:
                seen_paths[path] = int(line) if str(line).isdigit() else 0
            if check_images and not (image_root / path).is_file():
                problems.append(f"{location}: image not found: {image_root / path}")
        if split not in VALID_SPLITS:
            problems.append(f"{location}: split must be one of {VALID_SPLITS}, got {split!r}")
        else:
            counts[split] += 1
        if label not in label_map:
            problems.append(f"{location}: label {label!r} is not in label_map.json")
        if not group:
            problems.append(f"{location}: empty group_id")
        else:
            group_splits.setdefault(group, set()).add(split)
        per_class.setdefault(label, Counter())[split] += 1

    for group, splits in sorted(group_splits.items()):
        real_splits = sorted(split for split in splits if split in VALID_SPLITS)
        if len(real_splits) > 1:
            problems.append(
                f"group {group!r} appears in more than one split: {', '.join(real_splits)}"
            )

    total = sum(counts.values())
    ratios = {split: counts.get(split, 0) / total for split in VALID_SPLITS} if total else {}
    if total:
        problems.extend(_ratio_problems(counts, total, tolerance, "overall: "))
    for label, label_counts in sorted(per_class.items()):
        label_total = sum(label_counts.values())
        if label_total:
            problems.extend(
                _ratio_problems(label_counts, label_total, class_tolerance, f"class {label}: ")
            )

    return SplitReport(
        counts={split: counts.get(split, 0) for split in VALID_SPLITS},
        ratios=ratios,
        per_class={label: dict(value) for label, value in sorted(per_class.items())},
        problems=problems,
    )


def read_rows(split_file: Path) -> list[dict[str, str]]:
    """Read the split CSV and add a line number to every row.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If a required column is missing or the file has no rows.
    """
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Split file is missing columns: {missing}")
        rows = [{"_line": str(line), **row} for line, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError(f"Split file has no data rows: {split_file}")
    return rows


def format_report(report: SplitReport, split_file: Path, image_root: Path) -> str:
    """Return a printable report of the split counts and problems."""
    total = sum(report.counts.values())
    lines = [
        f"Split file: {split_file}",
        f"Image root: {image_root}",
        f"Rows: {total}",
        "Split counts:",
    ]
    for split in VALID_SPLITS:
        lines.append(
            f"  {split:<5} {report.counts.get(split, 0):>6}  {report.ratios.get(split, 0):>6.1%}"
        )
    lines.append("Per-class counts:")
    for label, counts in report.per_class.items():
        label_total = sum(counts.values())
        lines.append(
            f"  {label:<3} train {counts.get('train', 0):>4}  "
            f"val {counts.get('val', 0):>3}  "
            f"test {counts.get('test', 0):>3}  total {label_total:>4}"
        )
    if report.ok:
        lines.append("OK: the split file matches the contract.")
    else:
        lines.append(f"Problems ({len(report.problems)}):")
        lines.extend(f"  - {problem}" for problem in report.problems)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate the shared split file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cnn.yaml",
        help="Config that supplies default split, image root and label map paths.",
    )
    parser.add_argument("--split-file", type=Path, help="Override data.split_file.")
    parser.add_argument("--image-root", type=Path, help="Override data.image_root.")
    parser.add_argument("--label-map", type=Path, help="Override data.label_map.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute difference from the 70/15/15 shares overall.",
    )
    parser.add_argument(
        "--class-tolerance",
        type=float,
        default=0.10,
        help="Allowed absolute difference from the 70/15/15 shares per class.",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    """Resolve a config path against the repository root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 when the split is valid and 1 otherwise."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    try:
        config = load_config(args.config)["data"]
        split_file = _resolve(args.split_file or config["split_file"])
        image_root = _resolve(args.image_root or config["image_root"])
        label_map_path = _resolve(args.label_map or config["label_map"])
        with label_map_path.open(encoding="utf-8") as handle:
            label_map = json.load(handle)
        rows = read_rows(split_file)
        report = validate_rows(
            rows, label_map, image_root, args.tolerance, args.class_tolerance
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        logger.error("%s", error)
        return 1
    logger.info("%s", format_report(report, split_file, image_root))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

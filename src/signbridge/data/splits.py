"""Group-aware train/val/test split for the ASL images.

The input is a pool file with one row per image: ``path,label,group_id``. Images whose
group_id is the same look almost identical (near-duplicates), so a group is never cut:
all of its images go to the same split. The output is ``data/processed/split_v1.csv`` with
the columns ``path,label,group_id,split``.

Run from the repository root after ``python -m data_scripts.build_split_pool``:
    python -m signbridge.data.splits
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
POOL_COLUMNS = ("path", "label", "group_id")
SPLIT_COLUMNS = ("path", "label", "group_id", "split")
MIN_GROUPS_PER_CLASS = 3  # one group per split is the least that can fill all three splits


def check_fractions(fractions: dict[str, float]) -> None:
    """Check that the split fractions cover train, val and test and add up to 1.

    Raises:
        ValueError: If a split is missing, a fraction is not positive or the sum is not 1.
    """
    if set(fractions) != set(SPLITS):
        raise ValueError(f"fractions must have exactly the keys {SPLITS}, got {sorted(fractions)}")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError(f"every fraction must be above 0, got {fractions}")
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must add up to 1, got {sum(fractions.values()):.4f}")


def assign_groups(
    group_sizes: dict[str, int], fractions: dict[str, float], seed: int | str
) -> dict[str, str]:
    """Assign whole groups to splits so that image counts land close to the fractions.

    Groups are placed largest first. Each group goes to the split that is furthest below its
    target count, which keeps the shares close to the targets even when group sizes differ.
    Groups of equal size are shuffled with the seed, so the result is reproducible.

    Raises:
        ValueError: If the fractions are invalid or there are fewer than three groups.
    """
    check_fractions(fractions)
    if len(group_sizes) < MIN_GROUPS_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_GROUPS_PER_CLASS} groups to fill three splits, "
            f"got {len(group_sizes)}"
        )
    total = sum(group_sizes.values())
    target = {split: total * fractions[split] for split in SPLITS}
    filled = dict.fromkeys(SPLITS, 0)
    order = sorted(group_sizes)
    random.Random(str(seed)).shuffle(order)
    order.sort(key=lambda group: -group_sizes[group])  # stable: ties keep the shuffled order
    assignment: dict[str, str] = {}
    for group in order:
        # Fill the three splits once each first, so none of them stays empty.
        empty = [split for split in SPLITS if filled[split] == 0]
        choices = empty or list(SPLITS)
        best = max(choices, key=lambda split: target[split] - filled[split])
        assignment[group] = best
        filled[best] += group_sizes[group]
    return assignment


def check_group_labels(rows: list[dict[str, str]]) -> None:
    """Fail if one group_id is used by more than one class.

    Raises:
        ValueError: Naming the first group that is shared between classes.
    """
    labels: dict[str, str] = {}
    for row in rows:
        group, label = row["group_id"], row["label"]
        if labels.setdefault(group, label) != label:
            raise ValueError(f"group_id {group!r} is used by classes {labels[group]} and {label}")


def check_no_group_leak(rows: list[dict[str, str]]) -> None:
    """Fail if a near-duplicate group appears in more than one split.

    Raises:
        ValueError: Naming a group that is in two or more splits.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[row["group_id"]].add(row["split"])
    leaked = sorted(group for group, splits in seen.items() if len(splits) > 1)
    if leaked:
        raise ValueError(
            f"{len(leaked)} group(s) appear in more than one split, first: {leaked[0]!r} "
            f"in {sorted(seen[leaked[0]])}"
        )


def build_split_rows(
    pool: list[dict[str, str]], fractions: dict[str, float], seed: int
) -> list[dict[str, str]]:
    """Split every class separately, keeping near-duplicate groups together.

    Args:
        pool: Rows with ``path``, ``label`` and ``group_id``.
        fractions: Share of images per split, for example 0.70, 0.15 and 0.15.
        seed: Seed from the config. The class name is mixed in, so each class differs.

    Returns:
        Rows with ``path``, ``label``, ``group_id`` and ``split``, sorted by label and path.

    Raises:
        ValueError: If the pool is empty, a path repeats, a group is shared between classes,
            a class has too few groups or the final check finds a leak.
    """
    if not pool:
        raise ValueError("The pool is empty.")
    paths = Counter(row["path"] for row in pool)
    repeated = [path for path, count in paths.items() if count > 1]
    if repeated:
        raise ValueError(f"Path appears more than once in the pool: {repeated[0]!r}")
    check_group_labels(pool)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        by_label[row["label"]].append(row)
    rows: list[dict[str, str]] = []
    for label in sorted(by_label):
        sizes = Counter(row["group_id"] for row in by_label[label])
        try:
            assignment = assign_groups(dict(sizes), fractions, f"{seed}:{label}")
        except ValueError as error:
            raise ValueError(f"Class {label}: {error}") from error
        for row in by_label[label]:
            rows.append({**{c: row[c] for c in POOL_COLUMNS}, "split": assignment[row["group_id"]]})
    rows.sort(key=lambda row: (row["label"], row["path"]))
    check_no_group_leak(rows)
    return rows


def read_pool(path: Path) -> list[dict[str, str]]:
    """Read the pool CSV.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a column is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Pool file not found: {path}. Run build_split_pool first.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in POOL_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Pool file is missing columns: {missing}")
        return list(reader)


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the split CSV with the columns path,label,group_id,split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fractions_from_config(config: dict) -> dict[str, float]:
    """Read the split fractions from the ``split`` section of the config."""
    section = config["split"]
    return {split: float(section[f"{split}_fraction"]) for split in SPLITS}


def log_counts(rows: list[dict[str, str]]) -> None:
    """Log the number and share of images per split."""
    counts = Counter(row["split"] for row in rows)
    for split in SPLITS:
        share = 100 * counts[split] / len(rows)
        logger.info("%-5s %6d images (%.1f%%)", split, counts[split], share)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Write the group-aware split file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pool", type=Path, help="Override paths.split_pool.")
    parser.add_argument("--output", type=Path, help="Override paths.split_file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    setup_logging()
    try:
        config = load_config(args.config)
        pool = read_pool(args.pool or resolve_path(config, "split_pool"))
        rows = build_split_rows(pool, fractions_from_config(config), int(config["seed"]))
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1
    output = args.output or resolve_path(config, "split_file")
    write_split(output, rows)
    logger.info("Wrote %d rows to %s", len(rows), output)
    log_counts(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

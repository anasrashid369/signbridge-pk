"""Clean the ASL training images class by class and write the pool for the split.

For each class the script measures every image, then drops unreadable images, quality
failures, exact duplicates and near-duplicates (same rules as the cleaning script). The
images that remain are grouped: images whose hashes differ by at most `split.group_distance`
bits share a group_id, so the split never puts look-alike frames on both sides. Finally it
keeps at most `split.images_per_class` images per class, drawn with the config seed.

Run one class first and note the time, then run all classes:
    python -m data_scripts.build_split_pool --classes A
    python -m data_scripts.build_split_pool

Output: paths.split_pool (not in Git) and paths.split_summary (committed). Re-running a class
replaces that class's rows. Then run `python -m signbridge.data.splits`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from data_scripts.clean.duplicates import (
    find_exact_duplicates,
    find_near_duplicates,
    hamming_distances,
)
from data_scripts.clean.quality import READ_ERRORS, ImageStats, measure_image, quality_flags
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.data.splits import POOL_COLUMNS, read_pool
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUMMARY_COLUMNS = [
    "label",
    "found",
    "unreadable",
    "quality_flagged",
    "exact_duplicates",
    "near_duplicates",
    "after_cleaning",
    "groups",
    "largest_group",
    "kept",
]
LOG_EVERY = 500


def list_class_images(image_root: Path, label: str) -> list[str]:
    """Return the image paths of one class relative to image_root, with forward slashes."""
    folder = image_root / label
    if not folder.is_dir():
        raise FileNotFoundError(f"Class folder not found: {folder}")
    names = sorted(path.name for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return [f"{label}/{name}" for name in names]


def measure_class(image_root: Path, paths: list[str]) -> tuple[dict[str, ImageStats], int]:
    """Measure every image. Returns the stats of the readable ones and the unreadable count."""
    stats: dict[str, ImageStats] = {}
    unreadable = 0
    for number, path in enumerate(paths, start=1):
        try:
            stats[path] = measure_image(image_root / path)
        except READ_ERRORS as error:
            logger.warning("Unreadable image %s: %s", path, error)
            unreadable += 1
        if number % LOG_EVERY == 0:
            logger.info("  measured %d of %d", number, len(paths))
    return stats, unreadable


def group_by_distance(
    items: list[tuple[str, int]], max_distance: int, label: str
) -> dict[str, str]:
    """Give images a shared group id when their hashes are close, directly or through a chain.

    Returns a mapping of path to group id such as ``A_g0007``. Ids are numbered in path order.
    """
    paths = [path for path, _ in items]
    hashes = np.array([value for _, value in items], dtype=np.uint64)
    parent = list(range(len(paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in range(len(paths) - 1):
        distances = hamming_distances(hashes[index + 1 :], int(hashes[index]))
        close = np.nonzero(distances <= max_distance)[0]
        for offset in close:
            other = index + 1 + int(offset)
            parent[find(other)] = find(index)
    names: dict[int, str] = {}
    groups: dict[str, str] = {}
    for index, path in enumerate(paths):
        root = find(index)
        if root not in names:
            names[root] = f"{label}_g{len(names):04d}"
        groups[path] = names[root]
    return groups


def group_by_leader(items: list[tuple[str, int]], max_distance: int, label: str) -> dict[str, str]:
    """Give images a group id with bounded (leader) grouping instead of chaining.

    Images are visited in path order. An image joins the first earlier group leader whose hash
    is within max_distance bits; otherwise it becomes a new leader. Every member is therefore
    close to its own leader, so a long video-like sequence cannot merge into one big group.

    Returns a mapping of path to group id such as ``A_g0007``.
    """
    leader_hashes = np.zeros(len(items), dtype=np.uint64)
    leader_count = 0
    groups: dict[str, str] = {}
    for path, value in items:
        if leader_count:
            distances = hamming_distances(leader_hashes[:leader_count], value)
            closest = int(distances.argmin())
            if distances[closest] <= max_distance:
                groups[path] = f"{label}_g{closest:04d}"
                continue
        leader_hashes[leader_count] = np.uint64(value)
        groups[path] = f"{label}_g{leader_count:04d}"
        leader_count += 1
    return groups


def clean_class(
    image_root: Path, label: str, config: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Clean and group one class. Returns the pool rows before sampling and the summary row."""
    paths = list_class_images(image_root, label)
    stats, unreadable = measure_class(image_root, paths)
    thresholds = config["quality"]
    readable = [path for path in paths if path in stats]
    good = [
        path
        for path in readable
        if not any(
            quality_flags(
                stats[path].width,
                stats[path].height,
                stats[path].blur_score,
                stats[path].brightness,
                thresholds,
            ).values()
        )
    ]
    exact = find_exact_duplicates([(path, stats[path].sha256) for path in good])
    unique = [path for path in good if path not in exact]
    near_distance = config["split"].get(
        "near_duplicate_distance", thresholds["near_duplicate_distance"]
    )
    near = find_near_duplicates([(path, stats[path].dhash) for path in unique], int(near_distance))
    kept = [path for path in unique if path not in near]
    group_function = {"chain": group_by_distance, "leader": group_by_leader}[
        config["split"].get("group_method", "leader")
    ]
    groups = group_function(
        [(path, stats[path].dhash) for path in kept], int(config["split"]["group_distance"]), label
    )
    sizes: dict[str, int] = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    summary = {
        "label": label,
        "found": len(paths),
        "unreadable": unreadable,
        "quality_flagged": len(readable) - len(good),
        "exact_duplicates": len(exact),
        "near_duplicates": len(near),
        "after_cleaning": len(kept),
        "groups": len(sizes),
        "largest_group": max(sizes.values(), default=0),
    }
    rows = [{"path": path, "label": label, "group_id": groups[path]} for path in kept]
    return rows, summary


def sample_rows(
    rows: list[dict[str, str]], limit: int, seed: int, label: str
) -> list[dict[str, str]]:
    """Keep at most limit rows, drawn with a seed that depends on the class (0 keeps all)."""
    if limit <= 0 or len(rows) <= limit:
        return rows
    return sorted(random.Random(f"{seed}:{label}").sample(rows, limit), key=lambda r: r["path"])


def replace_rows(
    path: Path, columns: list[str], new_rows: list[dict[str, Any]], labels: set[str]
) -> None:
    """Write a CSV where the rows of the given classes are replaced by new_rows."""
    old: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            old = [row for row in csv.DictReader(handle) if row["label"] not in labels]
    merged = sorted([*old, *new_rows], key=lambda row: (row["label"], row.get("path", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Clean ASL classes and write the split pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--classes", nargs="+", help="Class names to process. Default: all 26.")
    parser.add_argument("--image-root", type=Path, help="Folder with one sub-folder per class.")
    parser.add_argument("--images-per-class", type=int, help="Override split.images_per_class.")
    return parser.parse_args(argv)


def default_image_root(config: dict[str, Any]) -> Path:
    """Read the image root from configs/cnn.yaml, the same folder the training code uses."""
    cnn = load_config(REPO_ROOT / "configs" / "cnn.yaml")
    return (REPO_ROOT / cnn["data"]["image_root"]).resolve()


def run(args: argparse.Namespace) -> None:
    """Process the chosen classes and update the pool and summary files."""
    config = load_config(args.config)
    image_root = (args.image_root or default_image_root(config)).resolve()
    with resolve_path(config, "label_map").open(encoding="utf-8") as handle:
        classes = args.classes or sorted(json.load(handle))
    limit = (
        args.images_per_class
        if args.images_per_class is not None
        else int(config["split"]["images_per_class"])
    )
    seed = int(config["seed"])
    pool_rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for label in classes:
        started = time.perf_counter()
        rows, summary = clean_class(image_root, label, config)
        kept = sample_rows(rows, limit, seed, label)
        summary["kept"] = len(kept)
        pool_rows.extend(kept)
        summaries.append(summary)
        logger.info(
            "%s: found %d, unreadable %d, quality %d, exact dup %d, near dup %d, "
            "after cleaning %d in %d groups (largest %d), kept %d, %.1f s",
            label,
            summary["found"],
            summary["unreadable"],
            summary["quality_flagged"],
            summary["exact_duplicates"],
            summary["near_duplicates"],
            summary["after_cleaning"],
            summary["groups"],
            summary["largest_group"],
            summary["kept"],
            time.perf_counter() - started,
        )
    done = set(classes)
    replace_rows(resolve_path(config, "split_pool"), list(POOL_COLUMNS), pool_rows, done)
    replace_rows(resolve_path(config, "split_summary"), SUMMARY_COLUMNS, summaries, done)
    logger.info("Pool now holds %d images.", len(read_pool(resolve_path(config, "split_pool"))))


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
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

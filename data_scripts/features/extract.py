"""Extract normalized MediaPipe hand landmarks from images and cache them (feature set B).

Two modes, both run from the repository root:

    python -m data_scripts.features.extract --help
    python -m data_scripts.features.extract --split-file data/processed/split_v1.csv

Without --split-file, a random sample per class is taken from a folder of class folders.
With --split-file, every image listed in the split file is processed, and the cache keeps
the split name of each image. "Detected" means a hand was found and all its landmark
values are finite, so the features are usable.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_scripts.features.labels import load_label_map
from data_scripts.features.landmarks import NUM_COORDS, NUM_LANDMARKS, normalize_landmarks
from signbridge.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Kaggle folder names (upper case) that differ from the class names in label_map.json.
FOLDER_ALIASES = {"DEL": "DELETE"}
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63 numbers per hand
FEATURES_FILE = "landmark_features.npz"  # saved inside paths.processed_dir
DETECTIONS_FILE = "landmark_detections.csv"  # saved inside paths.processed_dir
LOG_EVERY = 100  # log progress after this many images

# Split mode (images listed in split_v1.csv).
SPLIT_NAMES = ("train", "val", "test")
SPLIT_COLUMNS = ("path", "label", "split")  # columns this script needs from the split file
SPLIT_FEATURES_FILE = "landmark_features_split.npz"  # saved inside paths.processed_dir
SPLIT_DETECTIONS_FILE = "landmark_detections_split.csv"  # saved inside paths.processed_dir
DETECTION_TABLE_FILE = "landmark_detection_by_split.csv"  # saved inside paths.figures_dir
# The split paths are relative to the folder named here, so both models read the same one.
CNN_CONFIG_PATH = REPO_ROOT / "configs" / "cnn.yaml"


@dataclass
class ExtractionResult:
    """Everything collected while extracting landmarks from a list of images."""

    vectors: list[np.ndarray] = field(default_factory=list)
    raw_vectors: list[np.ndarray] = field(default_factory=list)
    label_ids: list[int] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    # One row per readable image: (path, label, split, usable hand found).
    records: list[tuple[str, str, str, bool]] = field(default_factory=list)
    unreadable: int = 0
    non_finite: int = 0


def create_landmarker(model_path: Path) -> Any:
    """Create a MediaPipe hand landmarker in image mode. Use it in a ``with`` block."""
    import mediapipe as mp  # imported here so the helpers below work without MediaPipe

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(points: Any) -> np.ndarray:
    """Convert MediaPipe landmarks (objects with x, y, z) to an array of shape (n, 3)."""
    return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)


def extract_raw_landmarks(landmarker: Any, image_path: Path) -> np.ndarray | None:
    """Return the raw MediaPipe landmarks of the hand in an image, shape (21, 3).

    Returns:
        The landmark array, or None if no hand was detected.

    Raises:
        RuntimeError, ValueError, OSError: If the image cannot be read.
            The caller logs these and skips the image.
    """
    import mediapipe as mp

    image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    return landmarks_to_array(result.hand_landmarks[0])


def extract_features(landmarker: Any, image_path: Path) -> np.ndarray | None:
    """Return the 63 normalized landmark values of the hand in an image, or None if no hand."""
    raw = extract_raw_landmarks(landmarker, image_path)
    return None if raw is None else normalize_landmarks(raw)


def is_finite_vector(values: np.ndarray) -> bool:
    """Return True if every value is a finite number (no NaN, no infinity)."""
    return bool(np.all(np.isfinite(values)))


def folder_to_label(folder_name: str) -> str:
    """Return the class name for a folder name: upper case, with known aliases applied."""
    name = folder_name.upper()
    return FOLDER_ALIASES.get(name, name)


def find_images(
    root: Path, label_map: dict[str, int], max_per_class: int, seed: int
) -> tuple[list[tuple[Path, str]], int]:
    """Find images under root and label each one by the name of its folder.

    Args:
        root: Folder searched recursively for .jpg, .jpeg and .png files.
        label_map: Known class names. Images in other folders are skipped.
        max_per_class: Random sample size per class. 0 means no limit.
        seed: Seed for the random sample, so runs are reproducible.

    Returns:
        A list of (path, label) pairs and the number of images skipped because
        their folder name is not in the label map.
    """
    by_label: dict[str, list[Path]] = {}
    skipped = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = folder_to_label(path.parent.name)
        if label not in label_map:
            skipped += 1
            continue
        by_label.setdefault(label, []).append(path)

    rng = np.random.default_rng(seed)
    items: list[tuple[Path, str]] = []
    for label in sorted(by_label):
        paths = by_label[label]
        if max_per_class and len(paths) > max_per_class:
            chosen = rng.choice(len(paths), size=max_per_class, replace=False)
            paths = [paths[index] for index in sorted(chosen)]
        items.extend((path, label) for path in paths)
    return items, skipped


def read_split_file(path: Path, label_map: dict[str, int]) -> list[tuple[str, str, str]]:
    """Read the split file and return (relative image path, label, split) rows in file order.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If a needed column is missing, a label is not in the label map,
            or a split name is not train, val or test.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in SPLIT_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Split file is missing columns: {missing}")
        rows: list[tuple[str, str, str]] = []
        for number, row in enumerate(reader, start=2):
            label, split = row["label"], row["split"]
            if label not in label_map:
                raise ValueError(f"Line {number}: label {label!r} is not in the label map")
            if split not in SPLIT_NAMES:
                raise ValueError(f"Line {number}: split {split!r} is not one of {SPLIT_NAMES}")
            rows.append((row["path"], label, split))
    return rows


def process_images(
    landmarker: Any,
    entries: list[tuple[Path, str, str]],
    root: Path,
    label_map: dict[str, int],
) -> ExtractionResult:
    """Extract normalized landmarks for every (image path, label, split) entry.

    Unreadable images are logged and left out of the records. Images with no hand,
    or with non-finite landmark values, are recorded as not usable.
    """
    result = ExtractionResult()
    for index, (path, label, split) in enumerate(entries, start=1):
        relative = path.relative_to(root).as_posix()
        try:
            raw = extract_raw_landmarks(landmarker, path)
            vector = None if raw is None else normalize_landmarks(raw)
        except (RuntimeError, ValueError, OSError) as error:
            logger.warning("Skipping unreadable image %s: %s", relative, error)
            result.unreadable += 1
            continue

        usable = False
        if raw is None or vector is None:
            logger.warning("No hand found in %s", relative)
        elif not (is_finite_vector(raw) and is_finite_vector(vector)):
            logger.warning("Non-finite landmark values in %s, image not used", relative)
            result.non_finite += 1
        else:
            usable = True
            result.vectors.append(vector)
            result.raw_vectors.append(raw.flatten())
            result.label_ids.append(label_map[label])
            result.paths.append(relative)
            result.splits.append(split)
        result.records.append((relative, label, split, usable))
        if index % LOG_EVERY == 0:
            logger.info("Processed %d of %d images", index, len(entries))
    return result


def save_cache(
    out_dir: Path,
    vectors: list[np.ndarray],
    raw_vectors: list[np.ndarray],
    label_ids: list[int],
    paths: list[str],
    splits: list[str] | None = None,
    filename: str = FEATURES_FILE,
) -> Path:
    """Save normalized features (X), raw landmarks (raw), class indices (y) and paths.

    If splits is given, the split name of each kept image is saved as ``split``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / filename
    arrays = {
        "X": np.asarray(vectors, dtype=np.float32).reshape(-1, NUM_FEATURES),
        "raw": np.asarray(raw_vectors, dtype=np.float32).reshape(-1, NUM_FEATURES),
        "y": np.asarray(label_ids, dtype=np.int64),
        "paths": np.asarray(paths, dtype=str),
    }
    if splits is not None:
        arrays["split"] = np.asarray(splits, dtype=str)
    np.savez_compressed(destination, **arrays)
    return destination


def save_detections(out_dir: Path, records: list[tuple[str, str, bool]]) -> Path:
    """Save one row per image (path, label, detected) for the detection-rate chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / DETECTIONS_FILE
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "detected"])
        writer.writerows(records)
    return destination


def save_split_detections(out_dir: Path, records: list[tuple[str, str, str, bool]]) -> Path:
    """Save one row per image (path, label, split, detected) in split mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / SPLIT_DETECTIONS_FILE
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "split", "detected"])
        writer.writerows(records)
    return destination


def detection_rate_by_split(
    records: list[tuple[str, str, str, bool]],
) -> list[dict[str, object]]:
    """Return one row per split (and one for all images) with the hand detection rate."""
    groups = [(name, [row for row in records if row[2] == name]) for name in SPLIT_NAMES]
    groups.append(("all", list(records)))
    table: list[dict[str, object]] = []
    for name, group in groups:
        if not group:
            continue
        detected = sum(1 for row in group if row[3])
        table.append(
            {
                "split": name,
                "images": len(group),
                "detected": detected,
                "detection_rate": round(detected / len(group), 4),
            }
        )
    return table


def save_detection_table(path: Path, table: list[dict[str, object]]) -> Path:
    """Write the detection-rate-per-split table as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "images", "detected", "detection_rate"]
        )
        writer.writeheader()
        writer.writerows(table)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract normalized hand landmarks from images and cache them."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the data config YAML."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help=(
            "Folder with one sub-folder per class. Default: paths.asl_raw_dir in the config, "
            "or data.image_root in configs/cnn.yaml when --split-file is used."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Where to save the cache. Default: paths.processed_dir."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Random sample size per class (0 = all images). Ignored with --split-file.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Split CSV (path,label,group_id,split). Extract every image it lists.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Extract landmarks from all selected images and save the cache."""
    config = load_config(args.config)
    model_path = resolve_path(config, "hand_model")
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")
    split_mode = args.split_file is not None
    if args.images_dir is not None:
        images_dir = args.images_dir
    elif split_mode:
        images_dir = REPO_ROOT / load_config(CNN_CONFIG_PATH)["data"]["image_root"]
    else:
        images_dir = resolve_path(config, "asl_raw_dir")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    out_dir = args.out_dir or resolve_path(config, "processed_dir")
    seed = int(config["seed"])
    label_map = load_label_map(resolve_path(config, "label_map"))

    logger.info("Images folder: %s | seed: %d", images_dir, seed)
    if split_mode:
        rows = read_split_file(args.split_file, label_map)
        entries = [(images_dir / relative, label, split) for relative, label, split in rows]
        logger.info("Split file %s lists %d images", args.split_file, len(entries))
    else:
        items, skipped = find_images(images_dir, label_map, args.max_per_class, seed)
        logger.info(
            "Found %d images in %d classes (%d skipped: folder name not in label map)",
            len(items),
            len({label for _, label in items}),
            skipped,
        )
        entries = [(path, label, "") for path, label in items]

    with create_landmarker(model_path) as landmarker:
        result = process_images(landmarker, entries, images_dir, label_map)

    logger.info(
        "Usable hand in %d of %d images (%d unreadable, %d with non-finite values)",
        len(result.vectors),
        len(entries),
        result.unreadable,
        result.non_finite,
    )
    if not split_mode:
        features_file = save_cache(
            out_dir, result.vectors, result.raw_vectors, result.label_ids, result.paths
        )
        detections_file = save_detections(
            out_dir, [(path, label, usable) for path, label, _, usable in result.records]
        )
        logger.info("Saved features to %s and detections to %s", features_file, detections_file)
        return

    features_file = save_cache(
        out_dir,
        result.vectors,
        result.raw_vectors,
        result.label_ids,
        result.paths,
        splits=result.splits,
        filename=SPLIT_FEATURES_FILE,
    )
    detections_file = save_split_detections(out_dir, result.records)
    table = detection_rate_by_split(result.records)
    table_file = save_detection_table(
        resolve_path(config, "figures_dir") / DETECTION_TABLE_FILE, table
    )
    for row in table:
        logger.info(
            "%s: hand found in %d of %d images (%.1f%%)",
            row["split"],
            row["detected"],
            row["images"],
            100 * float(row["detection_rate"]),
        )
    logger.info(
        "Saved features to %s, detections to %s and the table to %s",
        features_file,
        detections_file,
        table_file,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or file error."""
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

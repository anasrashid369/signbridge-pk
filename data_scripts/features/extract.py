"""Extract normalized MediaPipe hand landmarks from images and cache them (feature set B).

Run from the repository root:
    python -m data_scripts.features.extract --help
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data_scripts.features.labels import load_label_map
from data_scripts.features.landmarks import NUM_COORDS, NUM_LANDMARKS, normalize_landmarks
from signbridge.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from signbridge.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63 numbers per hand
FEATURES_FILE = "landmark_features.npz"  # saved inside paths.processed_dir
DETECTIONS_FILE = "landmark_detections.csv"  # saved inside paths.processed_dir
LOG_EVERY = 100  # log progress after this many images


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


def extract_features(landmarker: Any, image_path: Path) -> np.ndarray | None:
    """Return the 63 normalized landmark values of the hand in an image.

    Returns:
        A flat array of 63 numbers, or None if no hand was detected.

    Raises:
        RuntimeError, ValueError, OSError: If the image is unreadable or the
            landmarks cannot be normalized. The caller logs these and skips the image.
    """
    import mediapipe as mp

    image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    return normalize_landmarks(landmarks_to_array(result.hand_landmarks[0]))


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
        label = path.parent.name.upper()
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


def save_cache(
    out_dir: Path, vectors: list[np.ndarray], label_ids: list[int], paths: list[str]
) -> Path:
    """Save features (X), class indices (y) and image paths to one .npz file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / FEATURES_FILE
    np.savez_compressed(
        destination,
        X=np.asarray(vectors, dtype=np.float32).reshape(-1, NUM_FEATURES),
        y=np.asarray(label_ids, dtype=np.int64),
        paths=np.asarray(paths, dtype=str),
    )
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
        help="Folder with one sub-folder per class. Default: paths.asl_raw_dir in the config.",
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Where to save the cache. Default: paths.processed_dir."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Random sample size per class (0 = all images). Default: 50.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Extract landmarks from all selected images and save the cache."""
    config = load_config(args.config)
    model_path = resolve_path(config, "hand_model")
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")
    images_dir = args.images_dir or resolve_path(config, "asl_raw_dir")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    out_dir = args.out_dir or resolve_path(config, "processed_dir")
    seed = int(config["seed"])
    label_map = load_label_map(resolve_path(config, "label_map"))

    logger.info("Images folder: %s | seed: %d", images_dir, seed)
    items, skipped = find_images(images_dir, label_map, args.max_per_class, seed)
    logger.info(
        "Found %d images in %d classes (%d skipped: folder name not in label map)",
        len(items),
        len({label for _, label in items}),
        skipped,
    )

    vectors: list[np.ndarray] = []
    label_ids: list[int] = []
    kept_paths: list[str] = []
    records: list[tuple[str, str, bool]] = []
    unreadable = 0
    with create_landmarker(model_path) as landmarker:
        for index, (path, label) in enumerate(items, start=1):
            relative = path.relative_to(images_dir).as_posix()
            try:
                vector = extract_features(landmarker, path)
            except (RuntimeError, ValueError, OSError) as error:
                logger.warning("Skipping unreadable image %s: %s", relative, error)
                unreadable += 1
                continue
            records.append((relative, label, vector is not None))
            if vector is None:
                logger.warning("No hand found in %s", relative)
            else:
                vectors.append(vector)
                label_ids.append(label_map[label])
                kept_paths.append(relative)
            if index % LOG_EVERY == 0:
                logger.info("Processed %d of %d images", index, len(items))

    features_file = save_cache(out_dir, vectors, label_ids, kept_paths)
    detections_file = save_detections(out_dir, records)
    logger.info(
        "Hand found in %d of %d images (%d unreadable images skipped)",
        len(vectors),
        len(items),
        unreadable,
    )
    logger.info("Saved features to %s and detections to %s", features_file, detections_file)


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

"""Shared helpers for the CNN tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLIT_COLUMNS = ["path", "label", "group_id", "split"]


def make_image(path: Path, seed: int, size: int = 32) -> Path:
    """Save a small reproducible RGB image and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


def make_split(
    root: Path, classes: tuple[str, ...] = ("A", "B"), per_split: int = 2, size: int = 32
) -> tuple[Path, Path, dict[str, int]]:
    """Create images and a split CSV, returning (split file, image root, label map)."""
    image_root = root / "images"
    rows = []
    for class_index, label in enumerate(classes):
        for split in ("train", "val", "test"):
            for number in range(per_split):
                relative = f"{label}/{label}{split}{number}.png"
                make_image(image_root / relative, seed=class_index * 100 + number, size=size)
                rows.append(
                    {
                        "path": relative,
                        "label": label,
                        "group_id": f"{label}_{split}_{number}",
                        "split": split,
                    }
                )
    split_file = root / "split.csv"
    with split_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    label_map = {label: index for index, label in enumerate(classes)}
    return split_file, image_root, label_map


def write_cnn_config(
    root: Path,
    split_file: Path,
    image_root: Path,
    label_map: dict[str, int],
    num_classes: int,
    epochs: int = 1,
    batch_size: int = 2,
    max_trials: int = 2,
    grid: dict[str, list[object]] | None = None,
) -> Path:
    """Write a tiny cnn.yaml that points all outputs inside root."""
    tuning_grid = grid or {
        "lr": [0.001],
        "batch_size": [2],
        "dropout": [0.1],
        "weight_decay": [0.0001],
        "unfreeze_blocks": [0],
    }
    config = {
        "seed": 7,
        "data": {
            "split_file": str(split_file),
            "image_root": str(image_root),
            "label_map": str(root / "label_map.json"),
            "image_size": 32,
            "columns": {"path": "path", "label": "label", "split": "split", "group_id": "group_id"},
        },
        "model": {
            "num_classes": num_classes,
            "pretrained": False,
            "dropout": 0.1,
            "unfreeze_blocks": 0,
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.001,
            "backbone_lr_mult": 0.1,
            "weight_decay": 0.0001,
            "label_smoothing": 0.0,
            "scheduler": "cosine",
            "patience": 2,
            "num_workers": 0,
        },
        "outputs": {
            "checkpoint": str(root / "outputs" / "best.pt"),
            "history_csv": str(root / "outputs" / "history.csv"),
            "predictions_dir": str(root / "outputs" / "predictions"),
            "trials_csv": str(root / "outputs" / "trials.csv"),
        },
        "tuning": {
            "epochs": 1,
            "max_trials": max_trials,
            "grid": tuning_grid,
        },
    }
    path = root / "cnn.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    import json

    (root / "label_map.json").write_text(json.dumps(label_map), encoding="utf-8")
    return path

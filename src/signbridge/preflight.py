"""Check the training environment and estimate the run time before training.

Run from the repository root:
    python -m signbridge.preflight --config configs/cnn.yaml

The command prints the PyTorch and torchvision versions, whether CUDA is
available and the GPU name, the split statistics, and the measured seconds per
training batch. It then estimates one epoch, the whole tuning search and the
final run from ``configs/cnn.yaml``. It writes no file inside the repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from signbridge.config import REPO_ROOT, load_config
from signbridge.data.image_dataset import SplitImageDataset
from signbridge.data.transforms import build_eval_transform, build_train_transform
from signbridge.models.cnn import build_mobilenet_v2
from signbridge.train_cnn import make_optimizer, resolve_repo_path
from signbridge.training.loop import select_device, set_seed, train_one_epoch

logger = logging.getLogger(__name__)

CNN_CONFIG = REPO_ROOT / "configs" / "cnn.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check GPU readiness and estimate training time.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CNN_CONFIG,
        help="CNN config YAML (default: configs/cnn.yaml).",
    )
    parser.add_argument("--split-file", type=Path, help="Override data.split_file.")
    parser.add_argument("--image-root", type=Path, help="Override data.image_root.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Estimate a capped training split (validation and test stay complete).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=5,
        help="Training batches to time (default: 5).",
    )
    return parser.parse_args(argv)


def split_statistics(split_file: Path, columns: dict[str, str]) -> dict[str, object]:
    """Return row counts per split and per class from the split file."""
    import csv

    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    counts: Counter[str] = Counter()
    per_class: Counter[tuple[str, str]] = Counter()
    with split_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            name for name in (columns["path"], columns["label"], columns["split"])
            if name not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Split file is missing columns: {sorted(missing)}")
        for row in reader:
            split = row[columns["split"]]
            label = row[columns["label"]]
            counts[split] += 1
            per_class[(label, split)] += 1
    labels = sorted({label for label, _ in per_class})
    return {
        "rows": sum(counts.values()),
        "counts": dict(counts),
        "classes": labels,
        "per_class": {
            label: {split: per_class[(label, split)] for split in ("train", "val", "test")}
            for label in labels
        },
    }


def _load_label_map(path: Path) -> dict[str, int]:
    """Load the committed class-name to index map."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _seconds(value: float) -> str:
    """Format a duration in hours, minutes and seconds."""
    hours, remainder = divmod(int(round(value)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def measure_training(
    config: dict,
    split_file: Path,
    image_root: Path,
    batches: int,
    max_per_class: int | None,
) -> dict[str, object]:
    """Time a few real training batches and estimate the full runs."""
    if batches <= 0:
        raise ValueError(f"batches must be positive, got {batches}")
    data = config["data"]
    label_map = _load_label_map(resolve_repo_path(data["label_map"]))
    size = int(data["image_size"])
    columns = data["columns"]
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device()
    train_set = SplitImageDataset(
        split_file,
        image_root,
        "train",
        label_map,
        build_train_transform(size),
        columns,
        max_per_class=max_per_class,
        seed=seed,
    )
    val_set = SplitImageDataset(
        split_file, image_root, "val", label_map, build_eval_transform(size), columns
    )
    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model = build_mobilenet_v2(
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
        pretrained=bool(config["model"]["pretrained"]),
        unfreeze_blocks=int(config["model"]["unfreeze_blocks"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"]["label_smoothing"]))
    optimizer = make_optimizer(
        model,
        float(config["train"]["lr"]),
        float(config["train"]["backbone_lr_mult"]),
        float(config["train"]["weight_decay"]),
    )
    start = time.perf_counter()
    train_one_epoch(
        model, train_loader, criterion, optimizer, device, batches, freeze_batchnorm=True
    )
    elapsed = time.perf_counter() - start
    measured = min(batches, len(train_loader))
    seconds_per_batch = elapsed / measured
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    epoch_seconds = (train_batches + val_batches) * seconds_per_batch
    tuning_seconds = (
        int(config["tuning"]["max_trials"])
        * int(config["tuning"]["epochs"])
        * epoch_seconds
    )
    final_seconds = int(config["train"]["epochs"]) * epoch_seconds
    return {
        "device": str(device),
        "seconds_per_batch": seconds_per_batch,
        "train_images": len(train_set),
        "val_images": len(val_set),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "epoch_seconds": epoch_seconds,
        "tuning_seconds": tuning_seconds,
        "final_seconds": final_seconds,
    }


def run_preflight(
    config_path: Path,
    split_file: Path | None = None,
    image_root: Path | None = None,
    batches: int = 5,
    max_per_class: int | None = None,
) -> dict[str, object]:
    """Print the environment report and return the measured estimates."""
    config = load_config(config_path)
    data = config["data"]
    split_path = resolve_repo_path(split_file or data["split_file"])
    image_path = resolve_repo_path(image_root or data["image_root"])
    statistics = split_statistics(split_path, data["columns"])
    logger.info("torch: %s", torch.__version__)
    logger.info("torchvision: %s", torchvision.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("GPU: none, training will use the CPU")
    logger.info(
        "Split rows: %d (%s)",
        statistics["rows"],
        ", ".join(f"{name} {count}" for name, count in statistics["counts"].items()),
    )
    logger.info("Classes: %d", len(statistics["classes"]))
    estimates = measure_training(
        config, split_path, image_path, batches, max_per_class
    )
    logger.info(
        "Timed %d training batches: %.3f s per batch",
        min(batches, int(estimates["train_batches"])),
        estimates["seconds_per_batch"],
    )
    logger.info(
        "Train images: %d, validation images: %d",
        estimates["train_images"],
        estimates["val_images"],
    )
    logger.info(
        "Estimated epoch (train + validation): %s",
        _seconds(float(estimates["epoch_seconds"])),
    )
    logger.info(
        "Estimated tuning (%d trials x %d epochs): %s",
        int(config["tuning"]["max_trials"]),
        int(config["tuning"]["epochs"]),
        _seconds(float(estimates["tuning_seconds"])),
    )
    logger.info(
        "Estimated final run (%d epochs): %s",
        int(config["train"]["epochs"]),
        _seconds(float(estimates["final_seconds"])),
    )
    logger.info(
        "Estimated total: %s",
        _seconds(float(estimates["tuning_seconds"]) + float(estimates["final_seconds"])),
    )
    return {"statistics": statistics, **estimates}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        run_preflight(
            args.config,
            split_file=args.split_file,
            image_root=args.image_root,
            batches=args.batches,
            max_per_class=args.max_per_class,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

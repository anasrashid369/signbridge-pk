"""Train the MobileNetV2 CNN on the team split.

Run from the repository root:
    python -m signbridge.train_cnn --config configs/cnn.yaml --help

The test split is never read unless ``--evaluate-test`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from signbridge.config import REPO_ROOT, load_config
from signbridge.data.image_dataset import SplitImageDataset
from signbridge.data.transforms import build_eval_transform, build_train_transform
from signbridge.models.cnn import build_mobilenet_v2, count_trainable_parameters
from signbridge.training.loop import (
    evaluate_epoch,
    select_device,
    set_seed,
    train_one_epoch,
)
from signbridge.training.predictions import save_predictions

logger = logging.getLogger(__name__)

CNN_CONFIG = REPO_ROOT / "configs" / "cnn.yaml"
HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "lr",
    "seconds",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the MobileNetV2 letter classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CNN_CONFIG,
        help="CNN config YAML (default: configs/cnn.yaml).",
    )
    parser.add_argument("--split-file", type=Path, help="Override data.split_file.")
    parser.add_argument("--image-root", type=Path, help="Override data.image_root.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size.")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Use at most this many batches per epoch (smoke tests only).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help=(
            "Cap the TRAIN split at this many images per class, using a seeded "
            "sample. Validation and test are unchanged."
        ),
    )
    parser.add_argument(
        "--run-name", default="cnn", help="Name used for history, checkpoint and prediction files."
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test split once, after hyperparameters are frozen.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a config path against the repository root, unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_label_map(path: Path) -> dict[str, int]:
    """Load the committed class-name to index map."""
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_optimizer(
    model: nn.Module, learning_rate: float, backbone_multiplier: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Return AdamW with separate learning rates for the head and the backbone."""
    head = list(model.classifier.parameters())
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": learning_rate * backbone_multiplier})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer, name: str, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Return the scheduler named in the config."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(epochs // 3, 1), gamma=0.1
        )
    raise ValueError(f"Unknown scheduler {name!r}; use 'cosine' or 'step'")


def output_paths(config: dict, run_name: str) -> tuple[Path, Path]:
    """Return checkpoint and history paths, adding the run name for non-default runs."""
    checkpoint = resolve_repo_path(config["outputs"]["checkpoint"])
    history = resolve_repo_path(config["outputs"]["history_csv"])
    if run_name != "cnn":
        checkpoint = checkpoint.with_name(f"{run_name}{checkpoint.suffix}")
        history = history.with_name(f"{run_name}{history.suffix}")
    return checkpoint, history


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the per-epoch history CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _loaders(
    config: dict, options: SimpleNamespace
) -> tuple[SplitImageDataset, SplitImageDataset, DataLoader, DataLoader, dict[str, int]]:
    """Build the train and validation datasets and loaders."""
    data = config["data"]
    split_file = resolve_repo_path(options.split_file or data["split_file"])
    image_root = resolve_repo_path(options.image_root or data["image_root"])
    label_map = load_label_map(resolve_repo_path(data["label_map"]))
    size = int(data["image_size"])
    columns = data["columns"]
    train_set = SplitImageDataset(
        split_file,
        image_root,
        "train",
        label_map,
        build_train_transform(size),
        columns,
        max_per_class=getattr(options, "max_per_class", None),
        seed=int(config["seed"]),
    )
    val_set = SplitImageDataset(
        split_file, image_root, "val", label_map, build_eval_transform(size), columns
    )
    workers = int(config["train"]["num_workers"])
    batch_size = int(getattr(options, "batch_size", None) or config["train"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_set, val_set, train_loader, val_loader, label_map


def _test_predictions(
    config: dict, options: SimpleNamespace, model: nn.Module, device: torch.device, run_name: str
) -> None:
    """Evaluate the test split once and save its predictions."""
    data = config["data"]
    split_file = resolve_repo_path(options.split_file or data["split_file"])
    image_root = resolve_repo_path(options.image_root or data["image_root"])
    label_map = load_label_map(resolve_repo_path(data["label_map"]))
    test_set = SplitImageDataset(
        split_file,
        image_root,
        "test",
        label_map,
        build_eval_transform(int(data["image_size"])),
        data["columns"],
    )
    loader = DataLoader(
        test_set,
        batch_size=int(getattr(options, "batch_size", None) or config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )
    metrics = evaluate_epoch(
        model, loader, nn.CrossEntropyLoss(), device, getattr(options, "max_batches", None)
    )
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    destination = resolve_repo_path(config["outputs"]["predictions_dir"]) / f"{run_name}_test.npz"
    save_predictions(
        destination,
        metrics.y_true,
        metrics.y_proba,
        test_set.relative_paths[: len(metrics.y_true)],
        class_names,
    )
    logger.warning(
        "Test set evaluated once: macro-F1 %.4f, accuracy %.4f. Use it only after "
        "hyperparameters are frozen.",
        metrics.macro_f1,
        metrics.accuracy,
    )


def train(config: dict, options: SimpleNamespace) -> dict[str, object]:
    """Train, select the best checkpoint by validation macro-F1 and save predictions.

    Args:
        config: The loaded cnn.yaml mapping.
        options: Run options: split_file, image_root, epochs, batch_size,
            max_batches, run_name and evaluate_test.

    Returns:
        A summary with the best validation scores, epochs run, seconds and
        trainable parameter count.
    """
    seed = int(config["seed"])
    set_seed(seed)
    logger.info("Seed: %d", seed)
    device = select_device()
    logger.info("Device: %s", device)
    train_set, val_set, train_loader, val_loader, label_map = _loaders(config, options)
    model = build_mobilenet_v2(
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
        pretrained=bool(config["model"]["pretrained"]),
        unfreeze_blocks=int(config["model"]["unfreeze_blocks"]),
    ).to(device)
    trainable = count_trainable_parameters(model)
    logger.info("Trainable parameters: %d", trainable)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"]["label_smoothing"]))
    optimizer = make_optimizer(
        model,
        float(config["train"]["lr"]),
        float(config["train"]["backbone_lr_mult"]),
        float(config["train"]["weight_decay"]),
    )
    epochs = int(options.epochs or config["train"]["epochs"])
    scheduler = make_scheduler(optimizer, str(config["train"]["scheduler"]), epochs)
    checkpoint, history_path = output_paths(config, options.run_name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_accuracy = 0.0
    epochs_without_improvement = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            options.max_batches,
            freeze_batchnorm=True,
        )
        val = evaluate_epoch(model, val_loader, criterion, device, options.max_batches)
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val.loss, 6),
            "val_acc": round(val.accuracy, 6),
            "val_macro_f1": round(val.macro_f1, 6),
            "lr": round(learning_rate, 8),
            "seconds": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d train_loss %.4f train_acc %.4f val_loss %.4f val_acc %.4f "
            "val_macro_f1 %.4f (%.1fs)",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val.loss,
            val.accuracy,
            val.macro_f1,
            row["seconds"],
        )
        if val.macro_f1 > best_f1:
            best_f1, best_accuracy = val.macro_f1, val.accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_acc": best_accuracy,
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["train"]["patience"]):
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    epochs_without_improvement,
                )
                break
        scheduler.step()
    write_history(history_path, history)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    val = evaluate_epoch(model, val_loader, criterion, device, options.max_batches)
    predictions_dir = resolve_repo_path(config["outputs"]["predictions_dir"])
    destination = predictions_dir / f"{options.run_name}_val.npz"
    evaluated = len(val.y_true)
    save_predictions(
        destination, val.y_true, val.y_proba, val_set.relative_paths[:evaluated], class_names
    )
    if options.evaluate_test:
        _test_predictions(config, options, model, device, options.run_name)
    seconds = time.perf_counter() - start
    return {
        "best_val_macro_f1": best_f1,
        "best_val_accuracy": best_accuracy,
        "epochs_run": len(history),
        "seconds": seconds,
        "trainable_parameters": trainable,
        "checkpoint": str(checkpoint),
        "history_csv": str(history_path),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Load the config, apply command-line overrides and train."""
    config = load_config(args.config)
    options = SimpleNamespace(
        split_file=args.split_file,
        image_root=args.image_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_per_class=args.max_per_class,
        run_name=args.run_name,
        evaluate_test=args.evaluate_test,
    )
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    return train(config, options)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Best validation macro-F1: %.4f", result["best_val_macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the CNN hyperparameter search from configs/cnn.yaml.

Trial selection: the grid defines a cartesian product of parameter values.
A ``random.Random(seed)`` built from the config ``seed`` shuffles that full
list once; the search then walks the shuffled list from the start and runs at
most ``tuning.max_trials`` trials. Positions whose parameter set and seed are
already recorded in the trials CSV are skipped, so a resumed search fills the
remaining budget in the same order instead of starting a new search. The
result is a seeded random sample of the grid, not the first N grid
combinations.

Ranking: highest validation macro-F1 wins. Ties go to the trial with fewer
trainable parameters. The test split is never read here.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import random
from pathlib import Path
from types import SimpleNamespace

from signbridge import train_cnn
from signbridge.config import REPO_ROOT, load_config
from signbridge.training.trials import PARAM_COLUMNS, append_trial, has_trial, read_trials

logger = logging.getLogger(__name__)

CNN_CONFIG = REPO_ROOT / "configs" / "cnn.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the CNN tuning grid.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CNN_CONFIG,
        help="CNN config YAML (default: configs/cnn.yaml).",
    )
    parser.add_argument("--split-file", type=Path, help="Override data.split_file.")
    parser.add_argument("--image-root", type=Path, help="Override data.image_root.")
    parser.add_argument(
        "--epochs", type=int, help="Short epochs per trial (default: tuning.epochs)."
    )
    parser.add_argument("--max-trials", type=int, help="Override tuning.max_trials.")
    parser.add_argument("--max-batches", type=int, help="Use at most this many batches per epoch.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        help=(
            "Cap each trial's TRAIN split at this many images per class, using "
            "a seeded sample. Validation and test are unchanged."
        ),
    )
    return parser.parse_args(argv)


def parameter_sets(config: dict) -> list[dict[str, object]]:
    """Return every grid combination in a seeded random order.

    The grid in ``configs/cnn.yaml`` has five parameters with two values each,
    so the cartesian product has 32 entries. ``random.Random(config["seed"])``
    shuffles the list, and duplicate combinations (if a config lists the same
    value twice) are removed while keeping the first shuffled occurrence.
    ``tune`` walks this order from the start and skips positions already
    recorded in the trials CSV.
    """
    grid = config["tuning"]["grid"]
    values = [grid[column] for column in PARAM_COLUMNS]
    combinations = list(itertools.product(*values))
    random.Random(int(config["seed"])).shuffle(combinations)
    ordered: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for combination in combinations:
        if combination in seen:
            continue
        seen.add(combination)
        ordered.append(dict(zip(PARAM_COLUMNS, combination, strict=True)))
    return ordered


def best_trial(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the best trial: highest validation macro-F1, then fewest parameters."""
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["best_val_macro_f1"]),
            -int(float(row["trainable_parameters"] or 0)),
        ),
    )


def tune(config: dict, options: argparse.Namespace) -> list[dict[str, str]]:
    """Run unseen parameter sets and return all recorded trial rows."""
    trials_path = train_cnn.resolve_repo_path(config["outputs"]["trials_csv"])
    seed = int(config["seed"])
    epochs = int(options.epochs or config["tuning"]["epochs"])
    limit = int(options.max_trials or config["tuning"]["max_trials"])
    for index, params in enumerate(parameter_sets(config)):
        if index >= limit:
            break
        if has_trial(trials_path, params, seed):
            logger.info("Skipping recorded trial %d: %s", index, params)
            continue
        run_name = f"trial_{index:03d}"
        trial_config = copy.deepcopy(config)
        trial_config["model"]["dropout"] = params["dropout"]
        trial_config["model"]["unfreeze_blocks"] = params["unfreeze_blocks"]
        trial_config["train"]["lr"] = params["lr"]
        trial_config["train"]["batch_size"] = params["batch_size"]
        trial_config["train"]["weight_decay"] = params["weight_decay"]
        trial_config["outputs"]["predictions_dir"] = "reports/predictions/runs"
        trial_config["outputs"]["history_csv"] = "reports/predictions/runs/cnn_history.csv"
        run_options = SimpleNamespace(
            split_file=options.split_file,
            image_root=options.image_root,
            epochs=epochs,
            batch_size=params["batch_size"],
            max_batches=options.max_batches,
            max_per_class=getattr(options, "max_per_class", None),
            run_name=run_name,
            evaluate_test=False,
        )
        logger.info("Trial %d: %s", index, params)
        result = train_cnn.train(trial_config, run_options)
        append_trial(
            trials_path,
            params,
            {**result, "epochs": epochs},
            run_name,
            seed,
        )
    rows = read_trials(trials_path)
    winner = best_trial(rows)
    if winner is not None:
        logger.info(
            "Best trial %s: macro-F1 %s with %s trainable parameters",
            winner["run_name"],
            winner["best_val_macro_f1"],
            winner["trainable_parameters"],
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or data error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config)
        tune(config, args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

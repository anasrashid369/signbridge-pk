"""Turn a tuning trials CSV into a plain LaTeX tabular."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from signbridge.config import REPO_ROOT

logger = logging.getLogger(__name__)

COLUMNS = [
    "run_name",
    "lr",
    "batch_size",
    "dropout",
    "weight_decay",
    "unfreeze_blocks",
    "best_val_macro_f1",
    "val_accuracy",
    "epochs_run",
    "seconds",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert a tuning trials CSV to a LaTeX tabular.")
    parser.add_argument("--csv", type=Path, required=True, help="Trials CSV to read.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .tex file (default: reports/tuning/<csv stem>_table.tex).",
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the trials CSV and check that every required column exists."""
    if not path.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Trials CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Trials CSV has no rows: {path}")
    return rows


def write_table(rows: list[dict[str, str]], output: Path) -> None:
    """Write a plain tabular with one row per trial."""
    header = " & ".join(COLUMNS).replace("_", r"\_")
    lines = [
        r"\begin{tabular}{" + "l" + "r" * (len(COLUMNS) - 1) + "}",
        r"\hline",
        header + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [str(row[column]).replace("_", r"\_") for column in COLUMNS]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a CSV error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    output = args.output or REPO_ROOT / "reports" / "tuning" / f"{args.csv.stem}_table.tex"
    try:
        write_table(read_rows(args.csv), output)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

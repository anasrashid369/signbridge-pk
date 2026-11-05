"""Count non-merge commits per author and module, and write a LaTeX table."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
from collections import Counter
from pathlib import Path

from signbridge.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["module", "owner", "author", "commits"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the contribution table from git history.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/modules.yaml")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--revision", default="main", help="Revision to count from (default: main)."
    )
    parser.add_argument(
        "--csv", type=Path, default=REPO_ROOT / "reports/contribution_table.csv"
    )
    parser.add_argument(
        "--tex", type=Path, default=REPO_ROOT / "reports/contribution_table.tex"
    )
    return parser.parse_args(argv)


def author_counts(repo: Path, revision: str, paths: list[str]) -> Counter[str]:
    """Return the number of non-merge commits per author for the given paths."""
    command = [
        "git",
        "-C",
        str(repo),
        "log",
        "--use-mailmap",
        "--no-merges",
        "--format=%aN",
        revision,
        "--",
        *paths,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return Counter(line for line in result.stdout.splitlines() if line.strip())


def build_rows(config: dict, repo: Path, revision: str) -> list[dict[str, object]]:
    """Return one row per module and author with at least one commit."""
    rows: list[dict[str, object]] = []
    for name, module in config["modules"].items():
        counts = author_counts(repo, revision, module["paths"])
        for author, commits in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {"module": name, "owner": module["owner"], "author": author, "commits": commits}
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the contribution rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _escape(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a plain LaTeX tabular, without extra packages."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["module"]), []).append(row)
    lines = [
        r"\begin{tabular}{lll}",
        r"\hline",
        r"Module & Owner & Authors (commits) \\",
        r"\hline",
    ]
    for module, module_rows in grouped.items():
        owner = module_rows[0]["owner"]
        authors = ", ".join(
            f"{_escape(str(row['author']))} ({row['commits']})" for row in module_rows
        )
        lines.append(f"{_escape(module)} & {_escape(str(owner))} & {authors} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 on a config or git error."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config)
        rows = build_rows(config, args.repo, args.revision)
        write_csv(args.csv, rows)
        write_tex(args.tex, rows)
    except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Wrote %d rows to %s and %s", len(rows), args.csv, args.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

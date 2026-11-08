"""Count non-merge commits per author and module, and write a LaTeX table.

A commit can touch files from several modules, so every changed file is first
assigned to exactly one module: the most specific matching path pattern wins.
An exact path therefore beats a ``**`` glob, and a file that matches no module
is not counted.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from signbridge.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["module", "owner", "author", "commits"]
WILDCARDS = ("*", "?")


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


def _normalise(value: str) -> str:
    """Return a POSIX path without a leading ``./`` or ``/``."""
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.lstrip("/")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Turn a POSIX glob into an anchored regular expression.

    ``**`` matches any number of path segments and ``*`` matches inside one
    segment, so ``src/signbridge/training/**`` covers nested files while
    ``reports/predictions/landmark_*`` stays in one folder.
    """
    pattern = _normalise(pattern)
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif character == "*":
            parts.append("[^/]*")
            index += 1
        elif character == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(character))
            index += 1
    return re.compile("^" + "".join(parts) + "$")


def _specificity(pattern: str) -> tuple[int, int]:
    """Rank a pattern: an exact path beats a glob, then the longer literal wins."""
    literal = len(pattern) - sum(pattern.count(mark) for mark in WILDCARDS)
    return (2, literal) if not any(mark in pattern for mark in WILDCARDS) else (1, literal)


def module_for_path(path: str, modules: dict[str, dict[str, object]]) -> str | None:
    """Return the most specific module that owns ``path``, or None.

    Raises:
        ValueError: If two modules match with the same specificity, because
            that ownership is ambiguous.
    """
    relative = PurePosixPath(_normalise(path)).as_posix()
    matches: list[tuple[tuple[int, int], str]] = []
    for name, module in modules.items():
        for pattern in module["paths"]:
            if _glob_regex(str(pattern)).match(relative):
                matches.append((_specificity(str(pattern)), name))
                break
    if not matches:
        return None
    matches.sort(reverse=True)
    best_score, best_name = matches[0]
    if len(matches) > 1 and matches[1][0] == best_score and matches[1][1] != best_name:
        raise ValueError(
            f"Path {relative!r} matches {best_name!r} and {matches[1][1]!r} equally well; "
            "make one pattern more specific"
        )
    return best_name


def commit_files(repo: Path, revision: str) -> list[tuple[str, list[str]]]:
    """Return (author, changed files) for every non-merge commit on ``revision``."""
    command = [
        "git",
        "-C",
        str(repo),
        "log",
        "--use-mailmap",
        "--no-merges",
        "--name-only",
        "--format=%x1e%aN",
        revision,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    commits: list[tuple[str, list[str]]] = []
    for record in result.stdout.split("\x1e"):
        lines = [line.strip() for line in record.splitlines() if line.strip()]
        if not lines:
            continue
        commits.append((lines[0], lines[1:]))
    return commits


def build_rows(
    config: dict, repo: Path, revision: str
) -> tuple[list[dict[str, object]], list[str]]:
    """Return table rows and the changed files that belong to no module."""
    modules = config["modules"]
    counts: dict[str, Counter[str]] = {name: Counter() for name in modules}
    unmatched: set[str] = set()
    for author, files in commit_files(repo, revision):
        touched: set[str] = set()
        for path in files:
            if PurePosixPath(path).name == ".gitkeep":
                continue
            module = module_for_path(path, modules)
            if module is None:
                unmatched.add(path)
            else:
                touched.add(module)
        for module in touched:
            counts[module][author] += 1
    rows: list[dict[str, object]] = []
    for name, module in modules.items():
        for author, commits in sorted(counts[name].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {"module": name, "owner": module["owner"], "author": author, "commits": commits}
            )
    return rows, sorted(unmatched)


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
        rows, unmatched = build_rows(config, args.repo, args.revision)
        write_csv(args.csv, rows)
        write_tex(args.tex, rows)
    except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        logger.error("%s", error)
        return 1
    if unmatched:
        logger.warning("%d changed files belong to no module: %s", len(unmatched), unmatched)
    logger.info("Wrote %d rows to %s and %s", len(rows), args.csv, args.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for tools/trials_to_tex.py."""

import csv
from pathlib import Path

from tools.trials_to_tex import COLUMNS, main, read_rows


def write_csv(path: Path) -> Path:
    """Write one trial row with every expected column."""
    row = {column: "0.5" for column in COLUMNS}
    row.update(run_name="trial_000", batch_size="32", unfreeze_blocks="2")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_writes_tabular(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "trials.csv")
    output = tmp_path / "table.tex"
    assert main(["--csv", str(source), "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in text
    assert "trial\\_000" in text
    assert "best\\_val\\_macro\\_f1" in text


def test_missing_column_raises(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("run_name,lr\nx,0.1\n", encoding="utf-8")
    try:
        read_rows(source)
    except ValueError as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("ValueError was not raised")

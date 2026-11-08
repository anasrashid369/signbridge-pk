"""Tests for src/signbridge/data/splits.py."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from signbridge.data import splits
from signbridge.data.splits import (
    assign_groups,
    build_split_rows,
    check_no_group_leak,
    read_pool,
    write_split,
)

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_pool(
    labels: tuple[str, ...] = ("A", "B"), groups: int = 40, per_group: int = 5
) -> list[dict[str, str]]:
    """Create pool rows: every group holds per_group images."""
    rows = []
    for label in labels:
        for group in range(groups):
            for image in range(per_group):
                rows.append(
                    {
                        "path": f"{label}/{label}_{group}_{image}.jpg",
                        "label": label,
                        "group_id": f"{label}_g{group:04d}",
                    }
                )
    return rows


def test_no_group_appears_in_two_splits() -> None:
    rows = build_split_rows(make_pool(), FRACTIONS, seed=42)
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(found) == 1 for found in splits_by_group.values())


def test_check_no_group_leak_fails_when_a_group_is_in_two_splits() -> None:
    rows = [
        {"group_id": "A_g0001", "split": "train"},
        {"group_id": "A_g0001", "split": "test"},
    ]
    with pytest.raises(ValueError, match="A_g0001"):
        check_no_group_leak(rows)


def test_shares_are_close_to_70_15_15_in_every_class() -> None:
    rows = build_split_rows(make_pool(), FRACTIONS, seed=42)
    for label in ("A", "B"):
        counts = Counter(row["split"] for row in rows if row["label"] == label)
        total = sum(counts.values())
        assert total == 200
        assert abs(counts["train"] / total - 0.70) <= 0.03
        assert abs(counts["val"] / total - 0.15) <= 0.03
        assert abs(counts["test"] / total - 0.15) <= 0.03


def test_uneven_group_sizes_still_fill_all_three_splits() -> None:
    sizes = {"g1": 30, "g2": 20, "g3": 10, "g4": 5, "g5": 5, "g6": 1, "g7": 1}
    assignment = assign_groups(sizes, FRACTIONS, seed=1)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_same_seed_gives_same_split_and_other_seed_differs() -> None:
    pool = make_pool()
    first = build_split_rows(pool, FRACTIONS, seed=42)
    assert first == build_split_rows(pool, FRACTIONS, seed=42)
    assert first != build_split_rows(pool, FRACTIONS, seed=43)


def test_output_is_sorted_and_has_the_contract_columns() -> None:
    rows = build_split_rows(make_pool(), FRACTIONS, seed=42)
    assert list(rows[0]) == ["path", "label", "group_id", "split"]
    assert rows == sorted(rows, key=lambda row: (row["label"], row["path"]))


@pytest.mark.parametrize(
    "fractions",
    [
        {"train": 0.6, "val": 0.2},
        {"train": 0.6, "val": 0.3, "test": 0.3},
        {"train": 1.0, "val": 0.0, "test": 0.0},
    ],
)
def test_bad_fractions_are_rejected(fractions: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        assign_groups({"a": 1, "b": 1, "c": 1}, fractions, seed=1)


def test_class_with_too_few_groups_is_rejected() -> None:
    pool = make_pool(labels=("A",), groups=2)
    with pytest.raises(ValueError, match="Class A"):
        build_split_rows(pool, FRACTIONS, seed=1)


def test_group_shared_between_classes_is_rejected() -> None:
    pool = make_pool(labels=("A", "B"), groups=5)
    pool[-1]["group_id"] = pool[0]["group_id"]
    with pytest.raises(ValueError, match="used by classes"):
        build_split_rows(pool, FRACTIONS, seed=1)


def test_repeated_path_and_empty_pool_are_rejected() -> None:
    pool = make_pool(groups=5)
    pool.append(dict(pool[0]))
    with pytest.raises(ValueError, match="more than once"):
        build_split_rows(pool, FRACTIONS, seed=1)
    with pytest.raises(ValueError, match="empty"):
        build_split_rows([], FRACTIONS, seed=1)


def test_main_reads_pool_and_writes_split(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.csv"
    pool_path.write_text(
        "path,label,group_id\n"
        + "".join(f"{r['path']},{r['label']},{r['group_id']}\n" for r in make_pool()),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "split_v1.csv"
    code = splits.main(["--pool", str(pool_path), "--output", str(output)])
    assert code == 0
    assert output.read_text(encoding="utf-8").splitlines()[0] == "path,label,group_id,split"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 401


def test_main_returns_error_for_missing_pool(tmp_path: Path) -> None:
    assert splits.main(["--pool", str(tmp_path / "none.csv"), "--output", str(tmp_path / "o")]) == 1


def test_read_and_write_round_trip(tmp_path: Path) -> None:
    rows = build_split_rows(make_pool(groups=5), FRACTIONS, seed=1)
    path = tmp_path / "split.csv"
    write_split(path, rows)
    assert read_pool(path) == rows


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as info:
        splits.parse_args(["--help"])
    assert info.value.code == 0

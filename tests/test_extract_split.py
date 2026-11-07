"""Tests for the split-file mode of the landmark extraction."""

from pathlib import Path

import numpy as np
import pytest

from data_scripts.features import extract as extract_module
from data_scripts.features.extract import (
    FEATURES_FILE,
    detection_rate_by_split,
    is_finite_vector,
    process_images,
    read_split_file,
    save_cache,
    save_detection_table,
    save_split_detections,
)

LABEL_MAP = {"A": 0, "B": 1}


def write_split(path: Path, rows: list[str], header: str = "path,label,group_id,split") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_read_split_file_returns_rows_in_order(tmp_path: Path) -> None:
    split_file = write_split(
        tmp_path / "split.csv", ["A/1.jpg,A,g1,train", "B/1.jpg,B,g2,val", "A/2.jpg,A,g3,test"]
    )
    assert read_split_file(split_file, LABEL_MAP) == [
        ("A/1.jpg", "A", "train"),
        ("B/1.jpg", "B", "val"),
        ("A/2.jpg", "A", "test"),
    ]


def test_read_split_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_split_file(tmp_path / "nope.csv", LABEL_MAP)


def test_read_split_file_missing_column(tmp_path: Path) -> None:
    split_file = write_split(tmp_path / "split.csv", ["A/1.jpg,A,g1"], header="path,label,group_id")
    with pytest.raises(ValueError, match="missing columns"):
        read_split_file(split_file, LABEL_MAP)


def test_read_split_file_unknown_label(tmp_path: Path) -> None:
    split_file = write_split(tmp_path / "split.csv", ["Z/1.jpg,Z,g1,train"])
    with pytest.raises(ValueError, match="label"):
        read_split_file(split_file, LABEL_MAP)


def test_read_split_file_unknown_split_name(tmp_path: Path) -> None:
    split_file = write_split(tmp_path / "split.csv", ["A/1.jpg,A,g1,valid"])
    with pytest.raises(ValueError, match="split"):
        read_split_file(split_file, LABEL_MAP)


def test_is_finite_vector() -> None:
    assert is_finite_vector(np.zeros(63))
    assert not is_finite_vector(np.array([0.0, np.nan]))
    assert not is_finite_vector(np.array([0.0, np.inf]))


def test_detection_rate_by_split() -> None:
    records = [
        ("A/1.jpg", "A", "train", True),
        ("A/2.jpg", "A", "train", False),
        ("B/1.jpg", "B", "val", True),
    ]
    table = {row["split"]: row for row in detection_rate_by_split(records)}
    assert list(table) == ["train", "val", "all"]  # test has no images, so no row
    assert table["train"]["images"] == 2
    assert table["train"]["detected"] == 1
    assert table["train"]["detection_rate"] == 0.5
    assert table["val"]["detection_rate"] == 1.0
    assert table["all"]["images"] == 3
    assert table["all"]["detection_rate"] == pytest.approx(0.6667)


def test_save_detection_table_writes_header_and_rows(tmp_path: Path) -> None:
    table = detection_rate_by_split([("A/1.jpg", "A", "train", True)])
    lines = save_detection_table(tmp_path / "t" / "table.csv", table).read_text().splitlines()
    assert lines[0] == "split,images,detected,detection_rate"
    assert lines[1] == "train,1,1,1.0"


def test_save_split_detections_has_split_column(tmp_path: Path) -> None:
    destination = save_split_detections(tmp_path, [("A/1.jpg", "A", "train", True)])
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,label,split,detected"
    assert lines[1] == "A/1.jpg,A,train,True"


def test_save_cache_keeps_split_column(tmp_path: Path) -> None:
    destination = save_cache(
        tmp_path,
        [np.zeros(63), np.ones(63)],
        [np.zeros(63), np.ones(63)],
        [0, 1],
        ["A/1.jpg", "B/1.jpg"],
        splits=["train", "test"],
        filename="other.npz",
    )
    assert destination.name == "other.npz"
    data = np.load(destination)
    assert list(data["split"]) == ["train", "test"]


def test_save_cache_without_splits_keeps_old_format(tmp_path: Path) -> None:
    destination = save_cache(tmp_path, [np.zeros(63)], [np.zeros(63)], [0], ["A/1.jpg"])
    assert destination.name == FEATURES_FILE
    assert "split" not in np.load(destination).files


def test_process_images_handles_every_case(tmp_path: Path, monkeypatch) -> None:
    def fake_extract(landmarker, image_path: Path):
        if image_path.name == "bad.jpg":
            raise OSError("cannot read")
        if image_path.name == "none.jpg":
            return None
        raw = np.linspace(0.0, 1.0, 63).reshape(21, 3)
        if image_path.name == "nan.jpg":
            raw[0, 0] = np.nan
        return raw

    monkeypatch.setattr(extract_module, "extract_raw_landmarks", fake_extract)
    monkeypatch.setattr(extract_module, "normalize_landmarks", lambda raw: raw.flatten())

    entries = [
        (tmp_path / "A" / "1.jpg", "A", "train"),
        (tmp_path / "A" / "none.jpg", "A", "val"),
        (tmp_path / "B" / "nan.jpg", "B", "test"),
        (tmp_path / "B" / "bad.jpg", "B", "test"),
    ]
    result = process_images(None, entries, tmp_path, LABEL_MAP)

    assert result.paths == ["A/1.jpg"]
    assert result.splits == ["train"]
    assert result.label_ids == [0]
    assert len(result.vectors) == 1
    assert result.records == [
        ("A/1.jpg", "A", "train", True),
        ("A/none.jpg", "A", "val", False),
        ("B/nan.jpg", "B", "test", False),
    ]
    assert result.unreadable == 1
    assert result.non_finite == 1

"""Tests for the CSV-backed image dataset."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from signbridge.data.image_dataset import SplitImageDataset
from signbridge.data.transforms import build_eval_transform
from tests.cnn_helpers import make_split


def test_filters_one_split_and_maps_labels(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    dataset = SplitImageDataset(
        split_file, image_root, "val", label_map, build_eval_transform(32)
    )
    assert len(dataset) == 4
    tensor, label = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype.is_floating_point
    assert label in label_map.values()
    assert dataset.relative_paths[0].startswith(("A/", "B/"))


def test_unknown_label_raises(tmp_path: Path) -> None:
    split_file, image_root, _ = make_split(tmp_path)
    with pytest.raises(ValueError, match="Unknown label"):
        SplitImageDataset(split_file, image_root, "train", {"A": 0}, build_eval_transform(32))


def test_missing_split_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Split file"):
        SplitImageDataset(
            tmp_path / "missing.csv", tmp_path, "train", {"A": 0}, build_eval_transform(32)
        )


def test_unreadable_image_raises(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    (image_root / "A" / "Atrain0.png").write_text("not an image", encoding="utf-8")
    dataset = SplitImageDataset(
        split_file, image_root, "train", label_map, build_eval_transform(32)
    )
    with pytest.raises(RuntimeError, match="Could not read image"):
        _ = dataset[0]


def test_unknown_split_raises(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    with pytest.raises(ValueError, match="split must be"):
        SplitImageDataset(split_file, image_root, "holdout", label_map, build_eval_transform(32))


def test_max_per_class_caps_train_split_with_a_seeded_sample(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path, per_split=4)
    first = SplitImageDataset(
        split_file,
        image_root,
        "train",
        label_map,
        build_eval_transform(32),
        max_per_class=2,
        seed=7,
    )
    second = SplitImageDataset(
        split_file,
        image_root,
        "train",
        label_map,
        build_eval_transform(32),
        max_per_class=2,
        seed=7,
    )
    assert len(first) == 4
    assert first.relative_paths == second.relative_paths
    counts = {label: 0 for label in label_map}
    for path in first.relative_paths:
        counts[path.split("/")[0]] += 1
    assert counts == {"A": 2, "B": 2}


def test_max_per_class_is_rejected_outside_the_train_split(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    with pytest.raises(ValueError, match="only allowed for the train split"):
        SplitImageDataset(
            split_file,
            image_root,
            "val",
            label_map,
            build_eval_transform(32),
            max_per_class=1,
        )


def test_max_per_class_must_be_positive(tmp_path: Path) -> None:
    split_file, image_root, label_map = make_split(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        SplitImageDataset(
            split_file,
            image_root,
            "train",
            label_map,
            build_eval_transform(32),
            max_per_class=0,
        )

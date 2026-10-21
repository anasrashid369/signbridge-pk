"""Tests for label encoding."""

from pathlib import Path

import pytest

from data_scripts.features.labels import (
    build_label_map,
    encode_label,
    load_label_map,
    save_label_map,
)


def test_letters_only_has_26_classes() -> None:
    label_map = build_label_map(include_special=False)
    assert len(label_map) == 26
    assert label_map["A"] == 0
    assert label_map["Z"] == 25


def test_with_special_has_29_classes() -> None:
    label_map = build_label_map(include_special=True)
    assert len(label_map) == 29
    assert {"DELETE", "NOTHING", "SPACE"} <= set(label_map)


def test_letter_indices_do_not_change_with_special() -> None:
    plain = build_label_map(include_special=False)
    extended = build_label_map(include_special=True)
    assert all(extended[letter] == index for letter, index in plain.items())


def test_indices_are_unique_and_consecutive() -> None:
    label_map = build_label_map(include_special=True)
    assert sorted(label_map.values()) == list(range(29))


def test_encode_known_label() -> None:
    assert encode_label("B", build_label_map(include_special=False)) == 1


def test_encode_unknown_label_raises() -> None:
    with pytest.raises(KeyError):
        encode_label("7", build_label_map(include_special=False))


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    label_map = build_label_map(include_special=True)
    destination = tmp_path / "nested" / "label_map.json"
    save_label_map(label_map, destination)
    assert load_label_map(destination) == label_map

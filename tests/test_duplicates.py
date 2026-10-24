"""Tests for data_scripts/clean/duplicates.py."""

import numpy as np
import pytest

from data_scripts.clean.duplicates import (
    find_exact_duplicates,
    find_near_duplicates,
    hamming_distances,
)


def test_exact_duplicates_keep_the_first_copy() -> None:
    items = [("a.jpg", "h1"), ("b.jpg", "h2"), ("c.jpg", "h1"), ("d.jpg", "h1")]
    assert find_exact_duplicates(items) == {"c.jpg": "a.jpg", "d.jpg": "a.jpg"}


def test_exact_duplicates_none_found() -> None:
    assert find_exact_duplicates([("a.jpg", "h1"), ("b.jpg", "h2")]) == {}


def test_exact_duplicates_empty_input() -> None:
    assert find_exact_duplicates([]) == {}


def test_hamming_distances_count_differing_bits() -> None:
    hashes = np.array([0b0000, 0b0111, 2**63], dtype=np.uint64)
    assert hamming_distances(hashes, 0b0001).tolist() == [1, 2, 2]


def test_near_duplicates_use_the_distance_limit() -> None:
    items = [("a", 0b0000), ("b", 0b0011), ("c", 0b1111_1111)]
    assert find_near_duplicates(items, max_distance=2) == {"b": "a"}
    assert find_near_duplicates(items, max_distance=1) == {}


def test_near_duplicates_point_to_the_closest_kept_image() -> None:
    items = [("a", 0b000), ("b", 0b111), ("c", 0b110)]
    assert find_near_duplicates(items, max_distance=1) == {"c": "b"}


def test_near_duplicates_handle_large_hash_values() -> None:
    top_bit = 2**63
    assert find_near_duplicates([("a", top_bit), ("b", top_bit + 1)], 1) == {"b": "a"}


def test_near_duplicates_empty_input() -> None:
    assert find_near_duplicates([], max_distance=4) == {}


@pytest.mark.parametrize("distance", [-1, 65])
def test_near_duplicates_reject_bad_distance(distance: int) -> None:
    with pytest.raises(ValueError):
        find_near_duplicates([("a", 1)], max_distance=distance)

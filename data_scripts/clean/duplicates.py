"""Find exact and near-duplicate images from their hashes."""

import numpy as np

HASH_BITS = 64
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def find_exact_duplicates(items: list[tuple[str, str]]) -> dict[str, str]:
    """Find files with identical content.

    Args:
        items: (path, sha256) pairs in the order the first copy should be kept.

    Returns:
        A mapping of each duplicate path to the path of the copy that is kept.
    """
    first_seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for path, digest in items:
        if digest in first_seen:
            duplicates[path] = first_seen[digest]
        else:
            first_seen[digest] = path
    return duplicates


def hamming_distances(hashes: np.ndarray, value: int) -> np.ndarray:
    """Return the number of differing bits between value and each 64-bit hash in hashes."""
    xor = np.bitwise_xor(hashes, np.uint64(value))
    return _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def find_near_duplicates(items: list[tuple[str, int]], max_distance: int) -> dict[str, str]:
    """Find images whose hashes differ by at most max_distance bits from an earlier kept image.

    Args:
        items: (path, 64-bit hash) pairs in the order the first image should be kept.
        max_distance: Largest number of differing bits that still counts as a near-duplicate.

    Returns:
        A mapping of each near-duplicate path to the path of the image that is kept.

    Raises:
        ValueError: If max_distance is outside 0 to 64.
    """
    if not 0 <= max_distance <= HASH_BITS:
        raise ValueError(f"max_distance must be between 0 and {HASH_BITS}, got {max_distance}")
    kept_hashes = np.zeros(len(items), dtype=np.uint64)
    kept_paths: list[str] = []
    duplicates: dict[str, str] = {}
    for path, value in items:
        if kept_paths:
            distances = hamming_distances(kept_hashes[: len(kept_paths)], value)
            closest = int(distances.argmin())
            if distances[closest] <= max_distance:
                duplicates[path] = kept_paths[closest]
                continue
        kept_hashes[len(kept_paths)] = np.uint64(value)
        kept_paths.append(path)
    return duplicates

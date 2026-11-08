"""Restrict two prediction files to the test images that appear in both.

The landmark model can only predict on images where a hand was found, so its file has fewer rows
than the CNN file. A fair comparison uses the images present in both files.
"""

from __future__ import annotations

import numpy as np

ROW_KEYS = ("y_true", "y_proba", "paths")


def subset(data: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    """Return a prediction dictionary that keeps only the rows at the given indices."""
    chosen = {key: data[key][indices] for key in ROW_KEYS}
    chosen["class_names"] = data["class_names"]
    return chosen


def restrict_to_common(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Keep only the images present in both files, in the row order of the first file.

    Raises:
        ValueError: If the class names differ, a path appears twice in one file, the files share
            no image, or the two files disagree about the true class of a shared image.
    """
    if list(first["class_names"]) != list(second["class_names"]):
        raise ValueError("The two prediction files use different class names or order")
    paths_first = [str(path) for path in first["paths"]]
    paths_second = [str(path) for path in second["paths"]]
    if len(set(paths_first)) != len(paths_first) or len(set(paths_second)) != len(paths_second):
        raise ValueError("A prediction file lists the same image path twice")
    position = {path: index for index, path in enumerate(paths_second)}
    keep_first = [index for index, path in enumerate(paths_first) if path in position]
    if not keep_first:
        raise ValueError("The two prediction files have no image in common")
    keep_second = [position[paths_first[index]] for index in keep_first]
    if not np.array_equal(first["y_true"][keep_first], second["y_true"][keep_second]):
        raise ValueError("The two files disagree about the true class of a shared image")
    return subset(first, keep_first), subset(second, keep_second)

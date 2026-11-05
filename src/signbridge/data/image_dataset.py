"""PyTorch dataset backed by the team split file."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_COLUMNS = {"path": "path", "label": "label", "split": "split"}
VALID_SPLITS = ("train", "val", "test")


class SplitImageDataset(Dataset[tuple[Tensor, int]]):
    """Read one split from a CSV file and return (image tensor, class index).

    The CSV is the shared contract: one row per image with a path relative to
    ``image_root``, a class label, a near-duplicate group id and a split name.
    Only rows whose split column equals ``split`` are used.
    """

    def __init__(
        self,
        split_file: Path,
        image_root: Path,
        split: str,
        label_map: dict[str, int],
        transform: Callable[[Image.Image], Tensor],
        columns: dict[str, str] | None = None,
        max_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        """Load and validate the rows of one split.

        Args:
            split_file: CSV file with the split contract columns.
            image_root: Folder that the path column is relative to.
            split: One of ``train``, ``val`` or ``test``.
            label_map: Mapping from class name to class index.
            transform: Callable that turns a PIL RGB image into a tensor.
            columns: Optional mapping of logical names to CSV column names.
            max_per_class: Optional cap on the number of images per class.
                Only allowed for the ``train`` split, so validation and test
                stay comparable across runs. Selection is seeded by ``seed``.
            seed: Seed for the per-class subsampling.

        Raises:
            FileNotFoundError: If the split file does not exist.
            ValueError: If the split name is unknown, a column is missing, a
                label is not in the label map, the split has no rows, or
                ``max_per_class`` is used on a non-train split.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        if max_per_class is not None and split != "train":
            raise ValueError(
                "max_per_class is only allowed for the train split; "
                f"got split={split!r}"
            )
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        names = {**DEFAULT_COLUMNS, **(columns or {})}
        self.image_root = Path(image_root)
        self.transform = transform
        self._rows = self._read_rows(split_file, split, label_map, names)
        if not self._rows:
            raise ValueError(f"No rows with split {split!r} in {split_file}")
        if max_per_class is not None:
            self._rows = self._cap_per_class(self._rows, max_per_class, seed)

    @staticmethod
    def _read_rows(
        split_file: Path, split: str, label_map: dict[str, int], columns: dict[str, str]
    ) -> list[tuple[str, int]]:
        """Return (relative path, class index) for the requested split."""
        rows: list[tuple[str, int]] = []
        with split_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in columns.values() if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Split file is missing columns: {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                if row[columns["split"]] != split:
                    continue
                label = row[columns["label"]]
                if label not in label_map:
                    raise ValueError(
                        f"Unknown label {label!r} on line {line} of {split_file}; "
                        "add it to the label map or fix the split file"
                    )
                rows.append((row[columns["path"]], label_map[label]))
        return rows

    @staticmethod
    def _cap_per_class(
        rows: list[tuple[str, int]], max_per_class: int, seed: int
    ) -> list[tuple[str, int]]:
        """Return a seeded random sample of at most ``max_per_class`` rows per class.

        Rows are grouped by class index and sorted by relative path before
        sampling, so the result depends only on the split file, the cap and the
        seed. Classes with fewer rows than the cap are kept unchanged.
        """
        if max_per_class <= 0:
            raise ValueError(f"max_per_class must be positive, got {max_per_class}")
        by_class: dict[int, list[tuple[str, int]]] = {}
        for row in rows:
            by_class.setdefault(row[1], []).append(row)
        rng = random.Random(seed)
        kept: list[tuple[str, int]] = []
        for class_index in sorted(by_class):
            class_rows = sorted(by_class[class_index])
            if len(class_rows) <= max_per_class:
                kept.extend(class_rows)
            else:
                kept.extend(rng.sample(class_rows, max_per_class))
        return kept

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load one image, apply the transform and return (tensor, class index).

        Raises:
            FileNotFoundError: If the image is missing.
            RuntimeError: If PIL cannot read the image as RGB.
        """
        relative_path, class_index = self._rows[index]
        path = self.image_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
        except OSError as error:
            raise RuntimeError(f"Could not read image {path}: {error}") from error
        return tensor, class_index

    @property
    def relative_paths(self) -> list[str]:
        """Return the split paths in dataset order, for prediction files."""
        return [path for path, _ in self._rows]

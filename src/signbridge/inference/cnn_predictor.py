"""MobileNetV2 predictor that implements the shared inference interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from signbridge.config import REPO_ROOT
from signbridge.data.transforms import build_eval_transform
from signbridge.inference.errors import ModelNotFound
from signbridge.inference.image_io import load_rgb_array
from signbridge.inference.types import Prediction, load_labels
from signbridge.models.cnn import build_mobilenet_v2

DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "cnn_best.pt"
DEFAULT_LABEL_MAP = REPO_ROOT / "data" / "processed" / "label_map.json"


class CNNPredictor:
    """The Deliverable 2 MobileNetV2, loaded once and reused for every image.

    The checkpoint is the file ``signbridge.train_cnn`` writes: a dict with
    ``model_state_dict`` and the validation scores. It is loaded with
    ``weights_only=True`` because a state dict needs no pickled objects.

    Attributes:
        name: ``"cnn"``, so the app can label the result.
        labels: Class names in model output order.
        device: Torch device the model runs on.
    """

    name = "cnn"

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        label_map_path: str | Path = DEFAULT_LABEL_MAP,
        image_size: int = 224,
        dropout: float = 0.2,
        device: str | torch.device | None = None,
    ) -> None:
        """Load the labels and the checkpoint.

        Args:
            checkpoint_path: File written by ``signbridge.train_cnn``.
            label_map_path: ``label_map.json``; its indices must be 0..25.
            image_size: Side length in pixels used during training.
            dropout: Dropout probability used when the checkpoint was trained.
            device: ``"cpu"``, ``"cuda"`` or a ``torch.device``. Defaults to
                CUDA when it is available, otherwise the CPU.

        Raises:
            ModelNotFound: If the checkpoint file does not exist.
            ValueError: If the checkpoint has no ``model_state_dict`` or its
                output size does not match the label map.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.labels = load_labels(label_map_path)
        self.image_size = image_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = build_eval_transform(image_size)
        self._model = self._load_model(dropout)

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[Prediction]:
        """Return the ``top_k`` most likely letters for one RGB image.

        Args:
            image: uint8 RGB array of shape (height, width, 3). The shared
                helper rejects other dtypes, shapes and oversized images.
            top_k: Number of predictions to return, between 1 and 26.

        Returns:
            Predictions sorted from highest to lowest probability. The
            probabilities are a softmax over all 26 classes, so they sum to 1
            before the list is truncated.

        Raises:
            InvalidImage: If ``image`` is not a uint8 RGB array within limits.
            ValueError: If ``top_k`` is outside 1..26.
        """
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(f"top_k must be between 1 and {len(self.labels)}, got {top_k}")
        array = load_rgb_array(image)
        tensor = self.transform(Image.fromarray(array)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
        scores, indices = torch.topk(probabilities, k=top_k)
        return [
            Prediction(label=self.labels[index], probability=float(score))
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
        ]

    def _load_model(self, dropout: float) -> torch.nn.Module:
        """Build the architecture and copy the trained weights into it."""
        if not self.checkpoint_path.is_file():
            raise ModelNotFound(
                f"CNN checkpoint not found: {self.checkpoint_path}\n"
                "Download models/cnn_best.pt as described in README.md "
                "(section 'Model weights') or train it with "
                "'python -m signbridge.train_cnn'."
            )
        model = build_mobilenet_v2(
            num_classes=len(self.labels), dropout=dropout, pretrained=False, unfreeze_blocks=0
        )
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise ValueError(f"Checkpoint has no 'model_state_dict': {self.checkpoint_path}")
        try:
            model.load_state_dict(state["model_state_dict"])
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint does not match {len(self.labels)} classes: {self.checkpoint_path}"
            ) from error
        model.to(self.device)
        model.eval()
        return model

"""MobileNetV2 classifier for the 26 letters."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(
    num_classes: int = 26,
    dropout: float = 0.2,
    pretrained: bool = True,
    unfreeze_blocks: int = 2,
) -> nn.Module:
    """Build a MobileNetV2 with a new dropout and linear classifier head.

    The convolutional features are frozen first. ``unfreeze_blocks`` then
    re-enables the last N modules of ``model.features``; 0 trains the head only.
    A value at least as large as the number of feature modules trains all of
    them. Frozen BatchNorm layers must stay in evaluation mode during training
    (see :func:`set_frozen_batchnorm_eval`), otherwise their running statistics
    would drift on the small fine-tuning batches.

    Args:
        num_classes: Number of output classes (26 for A-Z).
        dropout: Dropout probability before the final linear layer.
        pretrained: Load ImageNet weights. Tests pass False so nothing is downloaded.
        unfreeze_blocks: Number of final feature modules to train.

    Returns:
        A MobileNetV2 model whose classifier outputs ``num_classes`` logits.

    Raises:
        ValueError: If num_classes is below 1 or unfreeze_blocks is negative.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if unfreeze_blocks < 0:
        raise ValueError(f"unfreeze_blocks must be zero or more, got {unfreeze_blocks}")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    blocks = list(model.features)
    for block in blocks[len(blocks) - min(unfreeze_blocks, len(blocks)) :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Put BatchNorm layers with no trainable parameters back into eval mode.

    Call this after ``model.train()`` at the start of every epoch. It keeps the
    running mean and variance of frozen blocks fixed while still letting the
    unfrozen blocks and the head train.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters that require a gradient."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

"""Tests for the MobileNetV2 wrapper."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
from torch import nn

from signbridge.models.cnn import (
    build_mobilenet_v2,
    count_trainable_parameters,
    set_frozen_batchnorm_eval,
)


def test_output_shape_is_batch_by_classes() -> None:
    model = build_mobilenet_v2(num_classes=26, pretrained=False, unfreeze_blocks=0)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 26)


def test_head_has_dropout_and_linear() -> None:
    model = build_mobilenet_v2(num_classes=26, dropout=0.3, pretrained=False, unfreeze_blocks=0)
    assert isinstance(model.classifier[0], nn.Dropout)
    assert model.classifier[0].p == 0.3
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].in_features == 1280
    assert model.classifier[1].out_features == 26


def test_trainable_parameters_grow_with_unfreeze_blocks() -> None:
    head_only = count_trainable_parameters(
        build_mobilenet_v2(pretrained=False, unfreeze_blocks=0)
    )
    two_blocks = count_trainable_parameters(
        build_mobilenet_v2(pretrained=False, unfreeze_blocks=2)
    )
    all_blocks = count_trainable_parameters(
        build_mobilenet_v2(pretrained=False, unfreeze_blocks=100)
    )
    assert 0 < head_only < two_blocks < all_blocks


def test_frozen_batchnorm_stays_in_eval_mode() -> None:
    model = build_mobilenet_v2(pretrained=False, unfreeze_blocks=2)
    model.train()
    set_frozen_batchnorm_eval(model)
    batchnorms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert batchnorms[0].training is False
    assert batchnorms[-1].training is True


def test_negative_unfreeze_blocks_raises() -> None:
    with pytest.raises(ValueError, match="unfreeze_blocks"):
        build_mobilenet_v2(pretrained=False, unfreeze_blocks=-1)

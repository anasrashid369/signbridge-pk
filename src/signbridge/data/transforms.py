"""Data transforms for the CNN (Deliverable 2)."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int) -> transforms.Compose:
    """Return the training transform for one RGB image.

    The transform resizes to a square, applies a small rotation and mild colour
    jitter, converts to a tensor and normalises with the ImageNet statistics
    used by the pretrained backbone.

    There is deliberately no horizontal flip. A mirrored hand is not the same
    sign: handedness and the direction of the thumb change the meaning of many
    letters, so a flip would create wrong labels.

    Args:
        image_size: Side length in pixels after resizing.

    Returns:
        A composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Return the deterministic transform used for validation and test images.

    Args:
        image_size: Side length in pixels after resizing.

    Returns:
        A composed torchvision transform with no random augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

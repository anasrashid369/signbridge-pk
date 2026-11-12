"""Errors raised by the inference layer."""


class NoHandDetected(Exception):
    """The landmark model found no hand in the image."""


class InvalidImage(Exception):
    """The uploaded file is not a usable image."""


class ModelNotFound(Exception):
    """The model weights file is missing."""

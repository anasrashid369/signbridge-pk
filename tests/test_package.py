"""Smoke test that the signbridge package imports."""

import signbridge


def test_package_exposes_version() -> None:
    """The package exposes a version string."""
    assert isinstance(signbridge.__version__, str)

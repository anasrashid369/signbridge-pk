"""Logging setup shared by the command-line scripts."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for a script run."""
    logging.basicConfig(level=level, format=LOG_FORMAT)

"""Logging utility functions."""

from logging import (
    CRITICAL,
    FileHandler,
    Formatter,
    Logger,
    NullHandler,
    StreamHandler,
    getLogger,
)
from os import getcwd

from more_bikes.util.args import get_logging_args


def disable_logging(name: str) -> None:
    """Disable logging."""
    logger = getLogger(name)
    logger.addHandler(NullHandler())
    logger.propagate = False
    logger.setLevel(CRITICAL)


def create_logger(name: str, path: str = getcwd()) -> Logger:
    """Create a logger."""
    file, level = get_logging_args()

    formatter = Formatter("%(asctime)s %(name)s %(message)s")

    stream_handler = StreamHandler()
    stream_handler.setLevel(level.upper())
    stream_handler.setFormatter(formatter)

    logger = getLogger(name)
    logger.setLevel(level.upper())
    logger.addHandler(stream_handler)

    if file:
        file_handler = FileHandler(f"{path}/{name}.log")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

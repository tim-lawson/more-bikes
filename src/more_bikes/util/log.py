"""Logging utility functions."""

from argparse import ArgumentParser
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


def disable_logging(name: str) -> None:
    """Disable logging."""
    logger = getLogger(name)
    logger.addHandler(NullHandler())
    logger.propagate = False
    logger.setLevel(CRITICAL)


def create_logger(name: str, path: str = getcwd()) -> Logger:
    """Create a logger."""
    file, level = get_args()

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


def get_args() -> tuple[bool, str]:
    """Get command-line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="info",
    )

    args = parser.parse_args()

    assert isinstance(args.file, bool)
    assert isinstance(args.level, str)

    return args.file, args.level

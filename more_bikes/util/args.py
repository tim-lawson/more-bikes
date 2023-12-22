"""Command-line arguments."""

from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
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
    parser.add_argument(
        "-e",
        "--experiment",
        nargs="+",
        type=str,
        default=[],
    )

    return parser


argument_parser = create_parser()


def get_logging_args() -> tuple[bool, str]:
    """Get logging command-line arguments."""
    args = argument_parser.parse_args()

    assert isinstance(args.file, bool)
    assert isinstance(args.level, str)

    return args.file, args.level


def get_task_args() -> list[str]:
    """Get task command-line arguments."""
    args = argument_parser.parse_args()

    assert isinstance(args.experiment, list)
    assert all(isinstance(experiment, str) for experiment in args.experiment)

    return args.experiment

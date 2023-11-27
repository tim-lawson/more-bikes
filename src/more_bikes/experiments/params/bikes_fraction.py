"""Utility to make `bikes_fraction` experiment params."""

from more_bikes.experiments.experiment import Processing
from more_bikes.util.processing import (
    BIKES_FRACTION,
    post_undo_bikes_fraction,
    pre_do_bikes_fraction,
    pre_dropna_row,
)


def proc_bikes_fraction(bikes_fraction: bool) -> Processing:
    """Make `bikes_fraction` processing specification."""
    if bikes_fraction:
        return Processing(
            target=BIKES_FRACTION,
            pre=[
                pre_do_bikes_fraction,
                pre_dropna_row([BIKES_FRACTION]),
            ],
            post=post_undo_bikes_fraction,
        )
    return Processing(
        pre=[
            pre_dropna_row(),
        ],
    )

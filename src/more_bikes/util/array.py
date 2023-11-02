"""NumPy array utilities."""

from typing import Any, TypeVar

from numpy import dtype, generic, isnan, ndarray

DTYPE_co = TypeVar("DTYPE_co", covariant=True, bound=generic)

NDArray = ndarray[Any, dtype[DTYPE_co]]


def dropna_col(x: NDArray) -> NDArray:
    """Drop columns with `NaN` values in any row."""
    return x[:, ~isnan(x).any(axis=0)]


def dropna_row(x: NDArray, column_index: int) -> NDArray:
    """Drop rows with `NaN` values in a given column."""
    return x[~isnan(x[:, column_index])]

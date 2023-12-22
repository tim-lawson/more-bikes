"""Pre- and post-processing utilities."""

from functools import reduce
from typing import Any, Callable, TypeVar

from numpy import dtype, float_, generic, ndarray
from pandas import DataFrame, Series

from more_bikes.data.feature import BIKES, Feature

DTYPE_co = TypeVar("DTYPE_co", covariant=True, bound=generic)

NDArray = ndarray[Any, dtype[DTYPE_co]]


def dropna_col(data: DataFrame) -> DataFrame:
    """Drop columns with `NaN` values."""
    return data.dropna(axis=1)


def split(data: DataFrame, target: Feature | str = BIKES) -> tuple[DataFrame, Series]:
    """Split the feature and target columns."""
    return data.drop(columns=target), data[target]


# Pre-processing.

PreProcessing = Callable[[DataFrame], DataFrame]


def pre_chain(sequence: list[PreProcessing]) -> PreProcessing:
    """Chain a sequence of pre-processing functions (left to right)."""
    return lambda data: reduce(lambda x, f: f(x), sequence, data)


def pre_identity(data: DataFrame) -> DataFrame:
    """Pre-processing: Identity/no-op."""
    return data


def pre_drop_columns(columns: list[str]) -> PreProcessing:
    """Pre-processing: Drop columns."""
    return lambda data: data.drop(columns=columns)


def pre_dropna_row(columns: list[str] | None = None) -> PreProcessing:
    """Preprocessing: Drop rows with `NaN` values in the specified columns."""
    return lambda data: data.dropna(axis=0, subset=columns or [BIKES])


# Post-processing.

PostProcessing = Callable[[DataFrame], Callable[[NDArray[float_]], NDArray[float_]]]


def post_identity(_x_test: DataFrame) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """Post-processing: Identity/no-op."""
    return lambda y_pred: y_pred


# Submission.

Submission = Callable[[DataFrame, NDArray[float_]], DataFrame]


def submission(x_test: DataFrame, y_pred: NDArray[float_]) -> DataFrame:
    """Format the predictions to submit."""
    return DataFrame(
        {
            "Id": x_test["id"].values,
            "bikes": y_pred,
        }
    )

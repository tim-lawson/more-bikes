"""pandas DataFrame utilities."""

from functools import reduce
from typing import Callable

from numpy import clip, float_
from pandas import DataFrame, Series

from more_bikes.data.feature import BIKES, Feature
from more_bikes.util.array import NDArray


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
    """Drop columns."""
    return lambda data: data.drop(columns=columns)


def pre_dropna_row(columns: list[str] | None = None) -> PreProcessing:
    """
    Drop rows with `NaN` values in the specified columns (defaults to target).
    """
    return lambda data: data.dropna(axis=0, subset=columns or [BIKES])


BIKES_FRACTION = "bikes_fraction"


def pre_do_bikes_fraction(data: DataFrame) -> DataFrame:
    """Pre-processing: Add a feature that is `bikes` divided by `docks`."""
    data = data.copy()
    data[BIKES_FRACTION] = data[BIKES] / data["docks"]
    data.drop(BIKES, axis=1, inplace=True)
    return data


# Post-processing.

PostProcessing = Callable[[DataFrame], Callable[[NDArray[float_]], NDArray[float_]]]


def post_identity(_x_test: DataFrame) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """Post-processing: Identity/no-op."""
    return lambda y_pred: y_pred


def post_undo_bikes_fraction(
    x_test: DataFrame,
) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """Post-processing: Multiply `bikes_fraction` by `docks`."""
    return lambda y_pred: clip(y_pred, 0.0, 1.0) * x_test["docks"]


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

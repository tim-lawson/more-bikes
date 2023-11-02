"""pandas DataFrame utilities."""

from functools import reduce
from typing import Callable

from pandas import DataFrame

from more_bikes.data.feature import TARGET
from more_bikes.util.array import NDArray

DataFrameMap = Callable[[DataFrame], DataFrame]


def chain_map(sequence: list[DataFrameMap]) -> DataFrameMap:
    """Chain a sequence of functions."""
    return lambda data: reduce(lambda x, f: f(x), sequence, data)


def identity(data: DataFrame) -> DataFrame:
    """Identity map."""
    return data


def create_drop(columns: list[str]) -> DataFrameMap:
    """Drop columns."""
    return lambda data: data.drop(columns=columns)


def dropna_col(data: DataFrame) -> DataFrame:
    """Drop columns with `NaN` values."""
    return data.dropna(axis=1)


def create_dropna_row(columns: list[str] | None = None) -> DataFrameMap:
    """
    Drop rows with `NaN` values in the specified columns (defaults to target).
    """
    return lambda data: data.dropna(axis=0, subset=columns or [TARGET])


def split(data: DataFrame):
    """Split the feature and target columns."""
    return data.drop(columns=TARGET), data[TARGET]


def submission(x_test: DataFrame, y_pred: NDArray) -> DataFrame:
    """Format the predictions to submit."""
    return DataFrame(
        {
            "Id": x_test["id"].values,
            "bikes": y_pred,
        }
    )

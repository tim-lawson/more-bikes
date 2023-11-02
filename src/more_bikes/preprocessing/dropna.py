"""Transformer to drop rows with `NaN` values in a feature column."""

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from more_bikes.util.array import NDArray, dropna_row


class DropNaTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop rows with `NaN` values in a given column."""

    def __init__(self, column_index: int):
        super().__init__()
        self.column_index = column_index

    def fit(self, _x: NDArray[Any], _y: NDArray[Any] | None = None):
        """No-op."""
        return self

    def transform(self, x: NDArray[Any], _y: NDArray[Any] | None = None):
        """Drop rows with `NaN` values in a given column."""
        return dropna_row(x, self.column_index)

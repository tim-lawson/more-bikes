"""Transformer that divides `bikes` by `docks`."""

from sklearn.base import BaseEstimator, TransformerMixin

from more_bikes.util.array import NDArray


class BikesFractionTransformer(BaseEstimator, TransformerMixin):
    """Divide `bikes` by `docks`."""

    def fit(self, _x: NDArray, _y: NDArray):
        """Fit."""
        return self

    def transform(self, x: NDArray, y: NDArray) -> NDArray:
        """Transform."""
        return y / x["docks"]

    def inverse_transform(self, x: NDArray, y: NDArray) -> NDArray:
        """Inverse transform."""
        return y * x["docks"]

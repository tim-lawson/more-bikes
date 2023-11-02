"""A regressor that predicts the average of the y-value of the training data."""

from typing import Literal

import numpy
from sklearn.base import BaseEstimator, RegressorMixin

Average = Literal["mean"]


class AverageRegressor(BaseEstimator, RegressorMixin):
    """Predicts the average of the y-value of the training data."""

    def __init__(self):
        self._average = "mean"
        self._average_y = None

    def fit(self, _x, y):
        """Compute the average."""
        self._average_y = numpy.mean(y)
        return self

    def predict(self, x):
        """Predict the average."""
        return numpy.full(x.shape[0], self._average_y)

    def set_params(self, **params):
        """Set the parameters."""
        self._average = params.get("average", self._average)
        return self

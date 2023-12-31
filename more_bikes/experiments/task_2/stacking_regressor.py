"""Stacking regressor."""

# pylint: disable=dangerous-default-value

from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingRegressor as StackingRegressor_

from more_bikes.data.model_loader import get_estimators


class StackingRegressor(StackingRegressor_):
    """A wrapper for `StackingRegressor` with pre-trained RLM models."""

    def __init__(
        self,
        final_estimator: BaseEstimator,
        models: list[str] = [
            "full",
            "full_temp",
            "short",
            "short_full",
            "short_full_temp",
            "short_temp",
        ],
    ):
        super().__init__(
            estimators=get_estimators(models), final_estimator=final_estimator
        )

        self.models = models

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

"""Stacking regressor."""

# pylint: disable=dangerous-default-value

from sklearn.ensemble import StackingRegressor as StackingRegressor_

from more_bikes.data.model_loader import get_estimators


class StackingRegressor(StackingRegressor_):
    """A wrapper for `StackingRegressor` with pre-trained RLM models."""

    def __init__(
        self,
        models: list[str] = [
            "full",
            "full_temp",
            "short",
            "short_full",
            "short_full_temp",
            "short_temp",
        ],
        **kwargs,
    ):
        super().__init__(estimators=get_estimators(models), **kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.models = models

        self._param_names = ["models"] + list(kwargs.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.estimators = get_estimators(self.models)

        return self

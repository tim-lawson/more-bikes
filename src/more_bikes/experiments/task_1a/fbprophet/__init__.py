"""Task 1A: Facebook's Prophet."""

from pandas import DataFrame
from prophet import Prophet
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FunctionTransformer, make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.util.array import NDArray
from more_bikes.util.dataframe import create_dropna_row
from more_bikes.util.datetime import utc_from_timestamp_data
from more_bikes.util.log import disable_logging

disable_logging("cmdstanpy")
disable_logging("fbprophet")


def make_fit(x: NDArray, y: DataFrame) -> DataFrame:
    """Make a DataFrame to fit a Prophet model."""
    return DataFrame(
        {
            "ds": x.flatten(),
            "y": y.to_numpy().reshape(-1, 1).flatten(),
        }
    )


def make_predict(x: NDArray) -> DataFrame:
    """Make a DataFrame to predict with a Prophet model."""
    return DataFrame(
        {
            "ds": x.flatten(),
        }
    )


class ProphetRegressor(BaseEstimator):
    """A wrapper for Prophet."""

    def __init__(self):
        super().__init__()
        self._model = Prophet()

    def set_params(self, **params):
        """Set parameters."""
        self._model = Prophet(**params)

    def fit(self, x: NDArray, y: DataFrame):
        """Fit the model."""
        self._model.fit(make_fit(x, y))

    def predict(self, x: NDArray):
        """Predict."""
        return self._model.predict(make_predict(x))["yhat"]


fbprophet = Task1AExperiment(
    model=Model(
        name="fbprophet",
        pipeline=make_pipeline(
            ColumnTransformer(
                [
                    (
                        "timestamp",
                        FunctionTransformer(utc_from_timestamp_data),
                        ["timestamp"],
                    ),
                ]
            ),
            ProphetRegressor(),
        ),
        params=[
            {
                "prophetregressor__changepoint_prior_scale": [
                    0.001,
                    0.01,
                    0.1,
                    0.5,
                ],
                "prophetregressor__seasonality_prior_scale": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
            }
        ],
        preprocessing=[create_dropna_row()],
    ),
    cv=TimeSeriesSplit(
        n_splits=5,
    ),
)


if __name__ == "__main__":
    fbprophet.run().save()

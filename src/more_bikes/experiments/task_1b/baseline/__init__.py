"""Task 1B: Baseline (average)."""

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.util import ParamGrid
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

params: ParamGrid = [
    {
        "regressor__dummyregressor__strategy": [
            "mean",
            # "median",
        ]
    }
]


def baseline():
    """Baseline (average)."""
    return Task1BExperiment(
        model=Model(
            name="baseline",
            pipeline=TransformedTargetRegressor(
                make_pipeline(DummyRegressor()),
                transformer=BikesFractionTransformer(),
            ),
            params=params,
        ),
    )

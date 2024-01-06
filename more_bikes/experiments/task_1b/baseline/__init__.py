"""Task 1B: Baseline (average)."""

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model, TaskExperiment
from more_bikes.preprocessing.bikes_fraction_transformer import BikesFractionTransformer
from more_bikes.preprocessing.transformed_target_regressor import (
    TransformedTargetRegressor,
)


def baseline():
    """Baseline (average)."""
    return TaskExperiment(
        task="1b",
        model=Model(
            name="baseline",
            pipeline=TransformedTargetRegressor(
                make_pipeline(DummyRegressor()),
                transformer=BikesFractionTransformer(),
            ),
            params=[
                {
                    "regressor__dummyregressor__strategy": [
                        "mean",
                        # "median",
                    ]
                }
            ],
        ),
    )

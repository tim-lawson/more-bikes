"""Task 1A: Baseline (average)."""

from sklearn.dummy import DummyRegressor

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.util import ParamGrid
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

params: ParamGrid = [
    {
        "regressor__strategy": [
            "mean",
            "median",
        ]
    }
]


def baseline():
    """Baseline (average)."""
    return Task1AExperiment(
        model=Model(
            name="baseline",
            pipeline=TransformedTargetRegressor(
                DummyRegressor(),
                transformer=BikesFractionTransformer(),
            ),
            params=params,
        ),
    )


if __name__ == "__main__":
    baseline().run().save()

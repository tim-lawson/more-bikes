"""Task 1B: Baseline (average)."""

from sklearn.dummy import DummyRegressor

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import HackTransformedTargetRegressor


def baseline():
    """Baseline (average)."""
    return Task1BExperiment(
        model=Model(
            name="baseline",
            pipeline=HackTransformedTargetRegressor(
                DummyRegressor(strategy="mean"),
                transformer=BikesFractionTransformer(),
            ),
        ),
    )


if __name__ == "__main__":
    baseline().run().save()

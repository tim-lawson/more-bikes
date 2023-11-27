"""Task 1B: Baseline (average)."""

from sklearn.pipeline import make_pipeline

from more_bikes.estimators.average_regressor import AverageRegressor
from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.bikes_fraction import proc_bikes_fraction
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment

param_grid = [
    {
        "averageregressor__average": ["mean"],
    }
]


def baseline():
    """Baseline (average)."""
    return Task1BExperiment(
        model=Model(
            name="baseline",
            pipeline=make_pipeline(
                AverageRegressor(),
            ),
            params=param_grid,
        ),
        processing=proc_bikes_fraction(True),
        cv=time_series_split,
    )


if __name__ == "__main__":
    baseline().run().save()

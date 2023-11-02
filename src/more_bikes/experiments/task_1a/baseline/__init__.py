"""Task 1A: baseline estimator (average)."""

from sklearn.pipeline import make_pipeline

from more_bikes.estimators.average_regressor import AverageRegressor
from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.util.dataframe import create_dropna_row

params = [
    {
        "averageregressor__average": ["mean"],
    }
]

baseline = Task1AExperiment(
    model=Model(
        name="baseline",
        pipeline=make_pipeline(
            AverageRegressor(),
        ),
        params=params,
        preprocessing=[create_dropna_row()],
    ),
    cv=time_series_split,
)


if __name__ == "__main__":
    baseline.run().save()

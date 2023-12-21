"""Task 1A experiments."""

from sklearn._config import set_config

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.task_1a.baseline import baseline
from more_bikes.experiments.task_1a.hgbr import hgbr

task_experiments: list[TaskExperiment] = [
    TaskExperiment(baseline, True),
    TaskExperiment(hgbr, True),
]


if __name__ == "__main__":
    set_config(transform_output="pandas")

    for task_experiment in task_experiments:
        if task_experiment.run:
            task_experiment.experiment().run().save()

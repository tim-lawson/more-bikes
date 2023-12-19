"""Task 1B experiments."""

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.task_1b.baseline import baseline
from more_bikes.experiments.task_1b.hgbr import hgbr
from more_bikes.experiments.task_1b.lightgbm import lightgbm
from more_bikes.experiments.task_1b.mlp import mlp

task_experiments: list[TaskExperiment] = [
    TaskExperiment(baseline, False),
    TaskExperiment(hgbr, True),
    TaskExperiment(lightgbm, False),
    TaskExperiment(mlp, False),
]

if __name__ == "__main__":
    for task_experiment in task_experiments:
        if task_experiment.run:
            task_experiment.experiment().run().save()

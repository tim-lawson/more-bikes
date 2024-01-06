"""Task 3 experiments."""

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.experiments import TaskExperiments, run_experiments

task_experiments: TaskExperiments[TaskExperiment] = {}


if __name__ == "__main__":
    run_experiments(task_experiments)

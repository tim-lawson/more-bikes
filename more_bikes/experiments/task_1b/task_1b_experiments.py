"""Task 1B experiments."""

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.experiments import TaskExperiments, run_experiments
from more_bikes.experiments.task_1b.baseline import baseline
from more_bikes.experiments.task_1b.decision_tree import decision_tree
from more_bikes.experiments.task_1b.hgbr import hgbr
from more_bikes.experiments.task_1b.lightgbm import lightgbm
from more_bikes.experiments.task_1b.mlp import mlp

task_experiments: TaskExperiments[TaskExperiment] = {
    "baseline": baseline,
    "decision_tree": decision_tree,
    "hgbr": hgbr,
    "lightgbm": lightgbm,
    "mlp": mlp,
}


if __name__ == "__main__":
    run_experiments(task_experiments)

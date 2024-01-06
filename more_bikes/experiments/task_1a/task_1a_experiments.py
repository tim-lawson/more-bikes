"""Task 1A experiments."""

from more_bikes.experiments.experiments import TaskExperiments, run_experiments
from more_bikes.experiments.task_1a.baseline import baseline
from more_bikes.experiments.task_1a.decision_tree import decision_tree
from more_bikes.experiments.task_1a.hgbr import hgbr
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment

task_experiments: TaskExperiments[Task1AExperiment] = {
    "baseline": baseline,
    "decision_tree": decision_tree,
    "hgbr": hgbr,
}


if __name__ == "__main__":
    run_experiments(task_experiments)

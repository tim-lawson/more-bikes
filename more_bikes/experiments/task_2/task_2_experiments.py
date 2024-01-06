"""Task 2 experiments."""

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.experiments import TaskExperiments, run_experiments
from more_bikes.experiments.task_2.stacking import stacking
from more_bikes.experiments.task_2.stacking_decision_tree import stacking_decision_tree
from more_bikes.experiments.task_2.stacking_hgbr import stacking_hgbr

task_experiments: TaskExperiments[TaskExperiment] = {
    "stacking": stacking,
    "stacking_decision_tree": stacking_decision_tree,
    "stacking_hgbr": stacking_hgbr,
}


if __name__ == "__main__":
    run_experiments(task_experiments)

"""Task 2B experiments."""

from typing import Callable

from sklearn import set_config

from more_bikes.experiments.task_2.stacking import stacking
from more_bikes.experiments.task_2.stacking_decision_tree import stacking_decision_tree
from more_bikes.experiments.task_2.stacking_hgbr import stacking_hgbr
from more_bikes.experiments.task_2.task_2_experiment import Task2Experiment
from more_bikes.util.args import get_task_args

task_experiments: dict[str, Callable[[], Task2Experiment]] = {
    "stacking": stacking,
    "stacking_decision_tree": stacking_decision_tree,
    "stacking_hgbr": stacking_hgbr,
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        task_experiments[arg]().run().save()

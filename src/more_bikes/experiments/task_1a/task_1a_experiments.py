"""Task 1A experiments."""

from typing import Callable

from sklearn._config import set_config

from more_bikes.experiments.task_1a.baseline import baseline
from more_bikes.experiments.task_1a.decision_tree import decision_tree
from more_bikes.experiments.task_1a.hgbr import hgbr
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.util.args import get_task_args

task_experiments: dict[str, Callable[[], Task1AExperiment]] = {
    "baseline": baseline,
    "decision_tree": decision_tree,
    "hgbr": hgbr,
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        task_experiments[arg]().run().save()

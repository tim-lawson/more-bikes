"""Task 1B experiments."""

from typing import Callable

from sklearn import set_config

from more_bikes.experiments.task_1b.baseline import baseline
from more_bikes.experiments.task_1b.hgbr import hgbr
from more_bikes.experiments.task_1b.lightgbm import lightgbm
from more_bikes.experiments.task_1b.mlp import mlp
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.util.args import get_task_args

task_experiments: dict[str, Callable[[], Task1BExperiment]] = {
    "baseline": baseline,
    "hgbr": hgbr,
    "lightgbm": lightgbm,
    "mlp": mlp,
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        task_experiments[arg]().run().save()

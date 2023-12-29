"""Task 2B experiments."""

from typing import Callable

from sklearn import set_config

from more_bikes.experiments.task_2.stacking import Task2Experiment, stacking
from more_bikes.util.args import get_task_args

task_experiments: dict[str, Callable[[], Task2Experiment]] = {
    "stacking": stacking,
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        task_experiments[arg]().run().save()

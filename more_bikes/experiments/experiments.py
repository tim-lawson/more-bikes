"""Utilities to run experiments."""

from typing import Callable, TypeVar

from sklearn import set_config

from more_bikes.experiments.experiment import Experiment
from more_bikes.util.args import get_task_args

TypeTaskExperiment = TypeVar("TypeTaskExperiment", bound=Experiment)

TaskExperiments = dict[str, Callable[[], TypeTaskExperiment]]


def run_experiments(task_experiments: TaskExperiments):
    """Run the experiments."""

    set_config(transform_output="pandas")

    for arg in get_task_args():
        task_experiments[arg]().run().save()

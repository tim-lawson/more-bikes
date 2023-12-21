"""Task 1A experiments."""

from sklearn._config import set_config

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.task_1a.baseline import baseline
from more_bikes.experiments.task_1a.hgbr import hgbr
from more_bikes.util.args import get_task_args

task_experiments: dict[str, TaskExperiment] = {
    "baseline": TaskExperiment(baseline),
    "hgbr": TaskExperiment(hgbr),
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        if task_experiments[arg].run:
            task_experiments[arg].experiment().run().save()

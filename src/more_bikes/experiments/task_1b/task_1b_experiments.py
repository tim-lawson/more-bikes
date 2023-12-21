"""Task 1B experiments."""

from sklearn import set_config

from more_bikes.experiments.experiment import TaskExperiment
from more_bikes.experiments.task_1b.baseline import baseline
from more_bikes.experiments.task_1b.hgbr import hgbr
from more_bikes.experiments.task_1b.lightgbm import lightgbm
from more_bikes.experiments.task_1b.mlp import mlp
from more_bikes.util.args import get_task_args

task_experiments: dict[str, TaskExperiment] = {
    "baseline": TaskExperiment(baseline),
    "hgbr": TaskExperiment(hgbr),
    "lightgbm": TaskExperiment(lightgbm),
    "mlp": TaskExperiment(mlp),
}


if __name__ == "__main__":
    set_config(transform_output="pandas")
    for arg in get_task_args():
        if task_experiments[arg].run:
            task_experiments[arg].experiment().run().save()

"""Task 1B experiment class."""

from typing import Self

from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator

from more_bikes.data.data_loader import DataLoaderTestN, DataLoaderTrainN
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.util import SearchStrategy
from more_bikes.preprocessing.util import split


class Task1BExperiment(Experiment):
    """A class to run task 1B experiments."""

    def __init__(
        self,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
        search: SearchStrategy = "grid",
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_1b/{model.name}"
        super().__init__(self._output_path, model, processing, cv, search)

    def run(self) -> Self:
        """Run the task 1B experiment."""
        super().run()

        results, _best_score, scores = self.__run()

        self.data = results

        self.scores = scores
        self.scores = self.scores.astype({"split": "int"})

        return self

    def __run(self) -> tuple[DataFrame, float, DataFrame]:
        x_train, y_train = split(
            self.pre(DataLoaderTrainN().data), self._processing.target
        )

        x_test = DataLoaderTestN().data

        return self._run(x_train, y_train, x_test)

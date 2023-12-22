"""Task 1A experiment class."""

from pandas import DataFrame, concat
from sklearn.model_selection import BaseCrossValidator

from more_bikes.data.data_loader import DataLoaderTest1, DataLoaderTrain1
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.util import SearchStrategy
from more_bikes.preprocessing.util import split


class Task1AExperiment(Experiment):
    """A class to run task 1A experiments."""

    def __init__(
        self,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
        search: SearchStrategy = "grid",
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_1a/{model.name}"
        super().__init__(self._output_path, model, processing, cv, search)

    def run(self, station_id_min=201, station_id_max=275):
        """Run the task 1A experiment."""
        super().run()

        resultss: list[DataFrame] = []

        best_scores: list[float] = []

        scoress: DataFrame = DataFrame({"station": [], "split": [], "score": []})

        for station_id in range(station_id_min, station_id_max + 1):
            results, best_score, scores = self.__run_station_id(station_id)

            resultss.append(results)

            best_scores.append(best_score)

            scores["station"] = station_id
            scoress = concat([scoress, scores], ignore_index=True)

        self.data = concat(resultss, ignore_index=True)

        self.scores = scoress
        self.scores = self.scores.astype({"station": "int", "split": "int"})

        self._logger.info("mean score %.3f", sum(best_scores) / len(best_scores))

        return self

    def __run_station_id(self, station_id: int) -> tuple[DataFrame, float, DataFrame]:
        self._logger.info("station id %s", station_id)

        x_train, y_train = split(
            self.pre(DataLoaderTrain1(station_id).data), self._processing.target
        )

        x_test = DataLoaderTest1(station_id).data

        return self._run(x_train, y_train, x_test)

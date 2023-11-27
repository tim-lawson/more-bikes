"""Task 1A experiment class."""

from pandas import DataFrame, Series, concat
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from more_bikes.data.data_loader import DataLoaderTest1, DataLoaderTrain1
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.util.processing import pre_chain, split


class Task1AExperiment(Experiment):
    """A class to run task 1A experiments."""

    def __init__(
        self, model: Model, processing: Processing, cv: BaseCrossValidator | None = None
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_1a/{model.name}"
        super().__init__(self._output_path, processing, model, cv)

    def run(
        self,
        station_id_min=201,
        station_id_max=275,
    ):
        """Run the task 1A experiment."""
        super().run()

        dataframes: list[DataFrame] = []
        scores: list[float] = []

        for station_id in range(station_id_min, station_id_max + 1):
            station, score = self.__run_station_id(station_id)

            dataframes.append(station)
            scores.append(score)

        self.data = concat(dataframes, ignore_index=True)

        self._logger.info("mean score %.3f", sum(scores) / len(scores))

        return self

    def __run_station_id(self, station_id: int) -> tuple[DataFrame, float]:
        self._logger.info("station id %s", station_id)

        pre = pre_chain(self._processing.pre)

        x_train, y_train = split(
            pre(DataLoaderTrain1(station_id).data), self._processing.target
        )

        x_test = DataLoaderTest1(station_id).data

        # If there is a parameter grid, search it.
        if self._cv and self._model.params is not None:
            grid_search_cv = GridSearchCV(
                estimator=self._model.pipeline,
                param_grid=self._model.params,
                scoring=[self._model.scoring],
                refit=self._model.scoring,
                cv=self._cv,
                verbose=1,
            )

            grid_search_cv.fit(x_train, y_train)

            self._logger.info("score %.3f", -grid_search_cv.best_score_)
            self._logger.info("params %s", grid_search_cv.best_params_)

            y_pred = grid_search_cv.predict(x_test)

            return self._output(x_test, y_pred, -grid_search_cv.best_score_)

        # If there is no parameter grid, run the pipeline.
        return self.__run_pipeline(x_train, y_train, x_test)

    def __run_pipeline(
        self,
        x_train: DataFrame,
        y_train: Series,
        x_test: DataFrame,
    ):
        self._model.pipeline.fit(x_train, y_train)

        y_pred = self._model.pipeline.predict(x_test)
        assert not isinstance(y_pred, tuple)

        score = float(self._model.pipeline.score(x_train, y_train))

        self._logger.info("score %.3f", score)

        return self._output(x_test, y_pred, score)

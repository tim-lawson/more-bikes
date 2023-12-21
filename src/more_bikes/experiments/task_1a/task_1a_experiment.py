"""Task 1A experiment class."""

from contextlib import redirect_stdout

from pandas import DataFrame, concat
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from more_bikes.data.data_loader import DataLoaderTest1, DataLoaderTrain1
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.util.processing import pre_chain, split


class Task1AExperiment(Experiment):
    """A class to run task 1A experiments."""

    def __init__(
        self,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_1a/{model.name}"
        super().__init__(self._output_path, processing, model, cv)

        self._pre = pre_chain(self._processing.pre)

    def run(
        self,
        station_id_min=201,
        station_id_max=275,
    ):
        """Run the task 1A experiment."""
        super().run()

        results_dataframes: list[DataFrame] = []
        scores: list[float] = []

        scores_dataframes: DataFrame = DataFrame(
            {"station": [], "split": [], "score": []}
        )

        for station_id in range(station_id_min, station_id_max + 1):
            results_dataframe, best_score, scores_dataframe = self.__run_station_id(
                station_id
            )

            results_dataframes.append(results_dataframe)
            scores.append(best_score)

            scores_dataframe["station"] = station_id
            scores_dataframes = concat(
                [scores_dataframes, scores_dataframe], ignore_index=True
            )

        self.data = concat(results_dataframes, ignore_index=True)
        self.scores = scores_dataframes
        self.scores = self.scores.astype({"station": "int", "split": "int"})

        self._logger.info("mean score %.3f", sum(scores) / len(scores))

        return self

    def __run_station_id(self, station_id: int) -> tuple[DataFrame, float, DataFrame]:
        self._logger.info("station id %s", station_id)

        x_train, y_train = split(
            self._pre(DataLoaderTrain1(station_id).data), self._processing.target
        )

        x_test = DataLoaderTest1(station_id).data

        # If there is a parameter grid, search it.
        if self._cv and self._model.params is not None:
            outfile = f"{self._output_path}/{self._model.name}_cv.log"
            with open(outfile, "w", encoding="utf-8") as out:
                with redirect_stdout(out):
                    grid_search_cv = GridSearchCV(
                        estimator=self._model.pipeline,
                        param_grid=self._model.params,
                        scoring=(self._model.scoring),
                        refit=self._model.scoring,
                        cv=self._cv,
                        verbose=10,
                    )
                    grid_search_cv.fit(x_train, y_train)

            self._logger.info("score %.3f", -grid_search_cv.best_score_)
            self._logger.info("params %s", grid_search_cv.best_params_)
            self._save_attrs(grid_search_cv.best_estimator_)

            best_scores: list[float] = []
            for index in range(grid_search_cv.n_splits_):
                best_score = -grid_search_cv.cv_results_[f"split{index}_test_score"][
                    grid_search_cv.best_index_
                ]
                self._logger.info("split %s", index)
                self._logger.info("split score %.3f", best_score)
                best_scores.append(best_score)

            y_pred = grid_search_cv.predict(x_test)

            return self._output(x_test, y_pred, grid_search_cv.best_score_, best_scores)

        # If there is no parameter grid, run the pipeline.
        return self._run_pipeline(x_train, y_train, x_test)

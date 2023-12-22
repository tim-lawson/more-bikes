"""Task 1B experiment class."""

from contextlib import redirect_stdout
from typing import Self

from pandas import DataFrame

# pylint: disable=unused-import
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    HalvingGridSearchCV,
)

from more_bikes.data.data_loader import DataLoaderTestN, DataLoaderTrainN
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.util import SearchStrategy
from more_bikes.preprocessing.util import pre_chain, split


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
        super().__init__(self._output_path, processing, model, cv)

        self._pre = pre_chain(self._processing.pre)
        self._search = search

    def run(
        self,
    ) -> Self:
        """Run the task 1B experiment."""
        super().run()

        results_dataframe, _best_score, scores_dataframe = self.__run()

        self.data = results_dataframe
        self.scores = scores_dataframe
        self.scores = self.scores.astype({"split": "int"})

        return self

    def __run(self) -> tuple[DataFrame, float, DataFrame]:
        x_train, y_train = split(
            self._pre(DataLoaderTrainN().data),
            self._processing.target,
        )

        x_test = DataLoaderTestN().data

        # If there is a parameter grid, search it.
        if self._cv is not None and self._model.params is not None:
            outfile = f"{self._output_path}/{self._model.name}_cv.log"
            with open(outfile, "w", encoding="utf-8") as out:
                with redirect_stdout(out):
                    # Halving grid search.
                    if self._search == "halving":
                        search = HalvingGridSearchCV(
                            estimator=self._model.pipeline,
                            param_grid=self._model.params,
                            scoring=self._model.scoring,
                            refit=True,
                            cv=self._cv,
                            verbose=3,
                        )
                    # Grid search.
                    else:
                        search = GridSearchCV(
                            estimator=self._model.pipeline,
                            param_grid=self._model.params,
                            scoring=(self._model.scoring),
                            refit=self._model.scoring,
                            cv=self._cv,
                            verbose=3,
                        )
                    search.fit(x_train, y_train)

            self._logger.info("score %.3f", -search.best_score_)
            self._logger.info("params %s", search.best_params_)
            self._save_attrs(search.best_estimator_)  # type: ignore

            best_scores: list[float] = []
            for index in range(search.n_splits_):
                best_score = -search.cv_results_[f"split{index}_test_score"][
                    search.best_index_
                ]
                self._logger.info("split %s", index)
                self._logger.info("split score %.3f", best_score)
                best_scores.append(best_score)

            y_pred = search.predict(x_test)

            return self._output(x_test, y_pred, search.best_score_, best_scores)

        # If there is no parameter grid, run the pipeline.
        return self._run_pipeline(x_train, y_train, x_test)

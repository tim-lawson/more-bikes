"""Task 1B experiment class."""

from contextlib import redirect_stdout
from typing import Self

from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn_genetic import GASearchCV

from more_bikes.data.data_loader import DataLoaderTestN, DataLoaderTrainN
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.preprocessing.util import pre_chain, split


class Task1BExperiment(Experiment):
    """A class to run task 1B experiments."""

    def __init__(
        self,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
        search: SearchStrategy = "grid",
        ga_search_cv_params: GASearchCVParams | None = None,
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_1b/{model.name}"
        super().__init__(self._output_path, processing, model, cv)

        self._pre = pre_chain(self._processing.pre)

        self._search = search
        self._ga_search_cv_params = ga_search_cv_params or GASearchCVParams()

    def run(
        self,
    ) -> Self:
        """Run the task 1B experiment."""
        super().run()

        results_dataframe, best_score, scores_dataframe = self.__run()

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
                    # Genetic algorithm.
                    if self._search == "genetic":
                        search = GASearchCV(
                            estimator=self._model.pipeline,
                            param_grid=self._model.params,
                            scoring=self._model.scoring,
                            refit=self._model.scoring,  # type: ignore
                            cv=self._cv,  # type: ignore
                            verbose=True,
                            **self._ga_search_cv_params.__dict__,
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
            self._save_attrs(search.best_estimator_)

            best_scores: list[float] = []
            for index in range(search.n_splits_):
                best_score = -search.cv_results_[f"split{index}_test_score"][
                    search.best_index_
                ]
                self._logger.info("split %s", index)
                self._logger.info("split score %.3f", best_score)
                best_scores.append(best_score)

            # Save the genetic-algorithm results.
            if self._search == "genetic":
                self.__to_csv(DataFrame(search.history), "history")  # type: ignore
                self.__to_csv(DataFrame(search.cv_results_), "cv_results")

            y_pred = search.predict(x_test)

            return self._output(x_test, y_pred, search.best_score_, best_scores)

        # If there is no parameter grid, run the pipeline.
        return self._run_pipeline(x_train, y_train, x_test)

    def __to_csv(self, data: DataFrame, name: str) -> None:
        data.to_csv(
            f"{self._output_path}/{self._model.name}/{self._model.name}_{name}.csv",
            index=False,
        )

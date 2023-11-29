"""Task 1B experiment class."""

from contextlib import redirect_stdout
from typing import Self

from pandas import DataFrame, Series
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn_genetic import GASearchCV

from more_bikes.data.data_loader import DataLoaderTestN, DataLoaderTrainN
from more_bikes.experiments.experiment import Experiment, Model, Processing
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.util.processing import pre_chain, split


class Task1BExperiment(Experiment):
    """A class to run task 1B experiments."""

    def __init__(
        self,
        model: Model,
        processing: Processing,
        cv: BaseCrossValidator | None = None,
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

        data, score = self.__run()

        self.data = data
        self._logger.info("score %.3f", score)

        return self

    def __run(self) -> tuple[DataFrame, float]:
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
                            scoring=[self._model.scoring],
                            refit=self._model.scoring,
                            cv=self._cv,
                            verbose=3,
                        )
                    search.fit(x_train, y_train)

            self._logger.info("score %.3f", -search.best_score_)
            self._logger.info("params %s", search.best_params_)

            # Save the genetic-algorithm results.
            if self._search == "genetic":
                self.__to_csv(DataFrame(search.history), "history")  # type: ignore
                self.__to_csv(DataFrame(search.cv_results_), "cv_results")

            y_pred = search.predict(x_test)

            return self._output(x_test, y_pred, search.best_score_)

        # If there is no parameter grid, run the pipeline.
        return self.__run_pipeline(x_train, y_train, x_test)

    def __to_csv(self, data: DataFrame, name: str) -> None:
        data.to_csv(
            f"{self._output_path}/{self._model.name}/{self._model.name}_{name}.csv",
            index=False,
        )

    def __run_pipeline(
        self,
        x_train: DataFrame,
        y_train: Series,
        x_test: DataFrame,
    ) -> tuple[DataFrame, float]:
        self._model.pipeline.fit(x_train, y_train)

        y_pred = self._model.pipeline.predict(x_test)
        assert not isinstance(y_pred, tuple)

        score = float(self._model.pipeline.score(x_train, y_train))

        self._logger.info("score %.3f", score)

        return self._output(x_test, y_pred, score)

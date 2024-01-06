"""Experiment classes."""

from abc import ABCMeta, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Self

from numpy import ndarray
from pandas import DataFrame, Series

# pylint: disable=unused-import
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    HalvingGridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from more_bikes.data.data_loader import DataLoaderTestN, DataLoaderTrainN
from more_bikes.data.feature import BIKES, Feature
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.util import ParamGrid, SearchStrategy
from more_bikes.preprocessing.transformed_target_regressor import (
    TransformedTargetRegressor,
)
from more_bikes.preprocessing.util import (
    PostProcessing,
    PreProcessing,
    Submission,
    post_identity,
    pre_chain,
    pre_dropna_row,
    split,
    submission,
)
from more_bikes.util.log import create_logger

SCORING = "neg_mean_absolute_error"


@dataclass
class Processing:
    """Processing specification."""

    # The target variable.
    target: Feature | str = BIKES

    # A sequence of fixed pre-processing steps to apply before the pipeline.
    pre: list[PreProcessing] = field(default_factory=lambda: [pre_dropna_row()])

    # A fixed post-processing step to apply after predicting.
    post: PostProcessing = field(default_factory=lambda: post_identity)

    # A fixed step to apply before submitting.
    submit: Submission = field(default_factory=lambda: submission)


@dataclass
class Model:
    """Model specification."""

    name: str

    # The pipeline to evaluate.
    pipeline: Pipeline | TransformedTargetRegressor

    # If the pipeline is parameterised, the grid to search.
    params: ParamGrid | None = None

    # The strategy to evaluate the model.
    scoring: str = SCORING


class Experiment(metaclass=ABCMeta):
    """Abstract experiment class."""

    # The results of the experiment.
    data: DataFrame | None = None

    # The cross-validation scores of the experiment.
    scores: DataFrame | None = None

    def __init__(
        self,
        output_path: str,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
        search: SearchStrategy = "grid",
    ) -> None:
        self._output_path = output_path
        self._processing = processing
        self._model = model
        self._cv = cv
        self._search = search

        self._logger = create_logger(self._model.name, self._output_path)

        self.pre = pre_chain(self._processing.pre)

    @abstractmethod
    def run(self) -> Self:
        """Run the experiment."""
        self._logger.info("run")

    def _output(
        self, x_test: DataFrame, y_pred: ndarray, score: float, scores: list[float]
    ) -> tuple[DataFrame, float, DataFrame]:
        y_pred = self._processing.post(x_test)(y_pred)

        return (
            self._processing.submit(x_test, y_pred),
            -score,
            DataFrame({"split": range(len(scores)), "score": scores}),
        )

    def save(self) -> None:
        """Save the results to a CSV."""
        if self.data is None:
            raise NotImplementedError("No results.")

        self.data.sort_values(by=["Id"]).to_csv(
            f"{self._output_path}/{self._model.name}_submission.csv",
            index=False,
        )

        if self.scores is not None:
            self.scores.to_csv(
                f"{self._output_path}/{self._model.name}_cv.csv",
                index=False,
            )

    def _run(
        self, x_train: DataFrame, y_train: Series, x_test: DataFrame
    ) -> tuple[DataFrame, float, DataFrame]:
        # If there is a parameter grid, search it.
        if self._cv is not None and self._model.params is not None:
            outfile = f"{self._output_path}/{self._model.name}_cv.log"
            open(outfile, "a", encoding="utf-8").close()
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
                            verbose=4,
                            aggressive_elimination=True,
                        )
                    # Grid search.
                    else:
                        search = GridSearchCV(
                            estimator=self._model.pipeline,
                            param_grid=self._model.params,
                            scoring=(self._model.scoring),
                            refit=self._model.scoring,
                            cv=self._cv,
                            verbose=4,
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

    def _run_pipeline(
        self, x_train: DataFrame, y_train: Series, x_test: DataFrame
    ) -> tuple[DataFrame, float, DataFrame]:
        self._model.pipeline.fit(x_train, y_train)

        y_pred = self._model.pipeline.predict(x_test)
        assert not isinstance(y_pred, tuple)

        scores = cross_val_score(
            self._model.pipeline,
            x_train,
            y_train,
            cv=self._cv,
            scoring=make_scorer(mean_absolute_error),
        )

        self._logger.info("score %.3f", scores.mean())

        return self._output(x_test, y_pred, -float(scores.mean()), list(scores))

    def _named_steps(self, estimator: Pipeline | TransformedTargetRegressor) -> Bunch:
        """The named steps of the pipeline."""
        return (
            estimator.named_steps
            if isinstance(estimator, Pipeline)
            # TODO: see `more_bikes/util/target.py`
            else estimator.regressor.named_steps  # type: ignore
            if isinstance(estimator, TransformedTargetRegressor)
            else Bunch()
        )

    def _save_attrs(self, estimator: Pipeline | TransformedTargetRegressor) -> None:
        """Save attributes of the pipeline to CSVs."""

        named_steps = self._named_steps(estimator)

        for named_step_name in named_steps:
            named_step = named_steps[named_step_name]

            if hasattr(named_step, "transformers_"):
                for _, transformer, *_args in named_step.transformers_:
                    transformer_name: str = transformer.__class__.__name__
                    transformer_name = transformer_name.lower()

                    if isinstance(transformer, VarianceThreshold):
                        DataFrame(
                            {
                                "feature": transformer.feature_names_in_,
                                "variance": transformer.variances_,
                                "support": transformer.get_support(),
                            }
                        ).to_csv(
                            f"{self._output_path}/{self._model.name}_{transformer_name}.csv",
                            index=False,
                        )


class TaskExperiment(Experiment):
    """A class to run task experiments."""

    def __init__(
        self,
        task: str,
        model: Model,
        processing: Processing = Processing(),
        cv: BaseCrossValidator = time_series_split,
        search: SearchStrategy = "grid",
        train=DataLoaderTrainN(),
    ) -> None:
        self._output_path = f"./more_bikes/experiments/task_{task}/{model.name}"
        self.train = train
        super().__init__(self._output_path, model, processing, cv, search)

    def run(self) -> Self:
        """Run the task experiment."""
        super().run()

        results, _best_score, scores = self.__run()

        self.data = results

        self.scores = scores
        self.scores = self.scores.astype({"split": "int"})

        return self

    def __run(self) -> tuple[DataFrame, float, DataFrame]:
        x_train, y_train = split(self.pre(self.train.data), self._processing.target)

        x_test = DataLoaderTestN().data

        return self._run(x_train, y_train, x_test)

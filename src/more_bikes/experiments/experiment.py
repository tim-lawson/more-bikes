"""Experiment classes."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Self

from pandas import DataFrame, Series
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from more_bikes.data.feature import BIKES, Feature
from more_bikes.experiments.params.util import ParamGrid, ParamSpace
from more_bikes.util.array import NDArray
from more_bikes.util.log import create_logger
from more_bikes.util.processing import (
    PostProcessing,
    PreProcessing,
    Submission,
    post_identity,
    pre_dropna_row,
    submission,
)
from more_bikes.util.target import TransformedTargetRegressor

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

    # If the pipeline is parameterised, the grid or space to search.
    params: ParamGrid | ParamSpace | None = None

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
        processing: Processing,
        model: Model,
        cv: BaseCrossValidator | None = None,
    ) -> None:
        self._output_path = output_path
        self._processing = processing
        self._model = model
        self._cv = cv

        self._logger = create_logger(self._model.name, self._output_path)

    @abstractmethod
    def run(
        self,
    ) -> Self:
        """Run the experiment."""
        self._logger.info("run")

    def _output(
        self, x_test: DataFrame, y_pred: NDArray, score: float, scores: list[float]
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

    def _run_pipeline(
        self,
        x_train: DataFrame,
        y_train: Series,
        x_test: DataFrame,
    ) -> tuple[DataFrame, float, DataFrame]:
        self._model.pipeline.fit(x_train, y_train)

        y_pred = self._model.pipeline.predict(x_test)
        assert not isinstance(y_pred, tuple)

        scores = cross_val_score(
            self._model.pipeline,
            x_train,
            y_train,
            cv=self._cv,
            scoring=mean_absolute_error,
        )

        self._logger.info("score %.3f", scores.mean())

        return self._output(x_test, y_pred, -float(scores.mean()), list(scores))

    def _named_steps(self, estimator: Pipeline | TransformedTargetRegressor) -> Bunch:
        return (
            estimator.named_steps
            if isinstance(estimator, Pipeline)
            else estimator.regressor.named_steps  # type: ignore
            if isinstance(estimator, TransformedTargetRegressor)
            else Bunch()
        )

    def _save_attrs(self, estimator: Pipeline | TransformedTargetRegressor) -> None:
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

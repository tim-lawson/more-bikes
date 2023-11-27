"""Experiment classes."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Self

from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from more_bikes.data.feature import TARGET, Feature
from more_bikes.experiments.params.util import ParamGrid, ParamSpace
from more_bikes.util.array import NDArray
from more_bikes.util.log import create_logger
from more_bikes.util.processing import (
    PostProcessing,
    PreProcessing,
    Submission,
    post_identity,
    pre_identity,
    submission,
)

SCORING = "neg_mean_absolute_error"


@dataclass
class Processing:
    """Processing specification."""

    # The target variable.
    target: Feature | str = TARGET

    # A sequence of fixed pre-processing steps to apply before the pipeline.
    pre: list[PreProcessing] = field(default_factory=lambda: [pre_identity])

    # A fixed post-processing step to apply after predicting.
    post: PostProcessing = field(default_factory=lambda: post_identity)

    # A fixed step to apply before submitting.
    submit: Submission = field(default_factory=lambda: submission)


@dataclass
class Model:
    """Model specification."""

    name: str

    # The pipeline to evaluate.
    pipeline: Pipeline

    # If the pipeline is parameterised, the grid or space to search.
    params: ParamGrid | ParamSpace | None = None

    # The strategy to evaluate the model.
    scoring: str = SCORING


class Experiment(metaclass=ABCMeta):
    """Abstract experiment class."""

    # The results of the experiment.
    data: DataFrame | None = None

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
        self,
        x_test: DataFrame,
        y_pred: NDArray,
        score: float,
    ) -> tuple[DataFrame, float]:
        y_pred = self._processing.post(x_test)(y_pred)
        return (
            self._processing.submit(x_test, y_pred),
            -score,
        )

    def save(self) -> None:
        """Save the results to a CSV."""
        if self.data is None:
            raise NotImplementedError("No results.")

        self.data.to_csv(
            f"{self._output_path}/{self._model.name}_submission.csv",
            index=False,
        )


@dataclass()
class TaskExperiment:
    """A convenience class to conditionally run experiments."""

    experiment: Callable[[], Experiment]
    run: bool = True

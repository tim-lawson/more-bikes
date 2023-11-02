"""Experiment classes."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Self

from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from more_bikes.experiments.params.util import ParamGrid, ParamSpace
from more_bikes.util.dataframe import DataFrameMap, identity
from more_bikes.util.log import create_logger

SCORING = "neg_mean_absolute_error"


@dataclass
class Model:
    """Model specification."""

    name: str

    # The pipeline to evaluate.
    pipeline: Pipeline

    # A sequence of fixed pre-processing steps to apply before the pipeline.
    preprocessing: list[DataFrameMap] = field(default_factory=lambda: [identity])

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
        model: Model,
        output_path: str,
        cv: BaseCrossValidator | None = None,
    ) -> None:
        self._model = model
        self._output_path = output_path
        self._cv = cv
        self._logger = create_logger(self._model.name, self._output_path)

    @abstractmethod
    def run(
        self,
    ) -> Self:
        """Run the experiment."""
        self._logger.info("run")

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

    experiment: Experiment
    run: bool = True

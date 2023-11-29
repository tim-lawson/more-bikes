"""Parameters of `GASearchCV`."""

from dataclasses import dataclass
from typing import Any, Literal

from sklearn_genetic.space import Categorical, Continuous, Integer

SearchStrategy = Literal["grid", "genetic"]

# Grid-search parameter space.
ParamGrid = list[dict[str, list[Any]]]

# Genetic-algorithm parameter search space.
ParamSpace = dict[str, Categorical | Continuous | Integer]


@dataclass
# pylint:disable=too-many-instance-attributes
class GASearchCVParams:
    """Optional parameters of `GASearchCV`."""

    population_size: int = 10
    generations: int = 40
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1
    tournament_size: int = 3
    elitism: bool = True
    n_jobs: int | None = -1
    keep_top_k: int = 1
    algorithm: Literal[
        "eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"
    ] = "eaMuPlusLambda"

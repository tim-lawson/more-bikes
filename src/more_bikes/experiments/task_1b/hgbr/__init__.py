"""Task 1B: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.bikes_fraction import proc_bikes_fraction
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.hgbr import hgbr_param_grid, hgbr_param_space
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import column_transformer_1b
from more_bikes.preprocessing.ordinal import ordinal_transformer

SEARCH: SearchStrategy = "grid"

params = hgbr_param_grid if SEARCH == "grid" else hgbr_param_space


def hgbr():
    """Histogram-based gradient-boosting regression tree."""
    return Task1BExperiment(
        model=Model(
            name="hgbr",
            pipeline=make_pipeline(
                ordinal_transformer.set_output(transform="pandas"),
                column_transformer_1b.set_output(transform="pandas"),
                HistGradientBoostingRegressor(
                    random_state=42,
                    verbose=1,
                ),
            ),
            params=params,
        ),
        processing=proc_bikes_fraction(True),
        cv=time_series_split,
        search=SEARCH,
        ga_search_cv_params=GASearchCVParams(
            generations=3,
            population_size=10,
        ),
    )


if __name__ == "__main__":
    hgbr().run().save()

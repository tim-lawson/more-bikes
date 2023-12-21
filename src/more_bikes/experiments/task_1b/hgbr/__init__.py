"""Task 1B: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.hgbr import hgbr_param_grid, hgbr_param_space
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.feature_selection.variance_threshold import feature_selection_variance
from more_bikes.preprocessing.ordinal import preprocessing_ordinal
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

SEARCH: SearchStrategy = "grid"

params = hgbr_param_grid if SEARCH == "grid" else hgbr_param_space


def hgbr():
    """Histogram-based gradient-boosting regression tree."""
    return Task1BExperiment(
        model=Model(
            name="hgbr",
            pipeline=TransformedTargetRegressor(
                make_pipeline(
                    # preprocessing
                    preprocessing_ordinal,
                    # feature selection
                    feature_selection_variance,
                    # regression
                    HistGradientBoostingRegressor(random_state=42),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
        search=SEARCH,
        ga_search_cv_params=GASearchCVParams(
            generations=3,
            population_size=10,
        ),
    )


if __name__ == "__main__":
    hgbr().run().save()

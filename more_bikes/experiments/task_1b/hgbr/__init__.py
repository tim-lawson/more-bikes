"""Task 1B: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.hgbr import hgbr_param_space, params
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.feature_selection.drop import feature_selection_drop
from more_bikes.feature_selection.variance_threshold import (
    feature_selection_variance_threshold,
)
from more_bikes.preprocessing.bikes_fraction_transformer import BikesFractionTransformer
from more_bikes.preprocessing.ordinal_transformer import preprocessing_ordinal
from more_bikes.preprocessing.transformed_target_regressor import (
    TransformedTargetRegressor,
)

SEARCH: SearchStrategy = "halving"

params = hgbr_param_space if SEARCH == "genetic" else params


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
                    feature_selection_variance_threshold,
                    feature_selection_drop(
                        [
                            "bikes_3h_diff_avg_short",
                            "bikes_avg_short",
                            "wind_speed_avg",
                        ]
                    ),
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

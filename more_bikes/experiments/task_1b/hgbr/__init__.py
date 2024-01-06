"""Task 1B: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model, TaskExperiment
from more_bikes.experiments.params.hgbr import best_params, params
from more_bikes.feature_selection.drop import feature_selection_drop
from more_bikes.feature_selection.variance_threshold import (
    feature_selection_variance_threshold,
)
from more_bikes.preprocessing.bikes_fraction_transformer import BikesFractionTransformer
from more_bikes.preprocessing.ordinal_transformer import preprocessing_ordinal
from more_bikes.preprocessing.transformed_target_regressor import (
    TransformedTargetRegressor,
)


def hgbr():
    """Histogram-based gradient-boosting regression tree."""
    return TaskExperiment(
        task="1b",
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
        search="halving",
    )

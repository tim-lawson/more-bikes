"""Task 1A: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.hgbr import hgbr_param_grid
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.feature_selection.variance_threshold import feature_selection_variance
from more_bikes.preprocessing.drop import make_preprocessing_drop
from more_bikes.preprocessing.ordinal import preprocessing_ordinal
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor


def hgbr():
    """Histogram-based gradient-boosting regression tree."""
    return Task1AExperiment(
        model=Model(
            name="hgbr",
            pipeline=TransformedTargetRegressor(
                make_pipeline(
                    # preprocessing
                    preprocessing_ordinal,
                    # feature selection
                    feature_selection_variance,
                    make_preprocessing_drop(
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
            params=hgbr_param_grid,
        ),
    )

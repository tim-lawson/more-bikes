"""
Stacking regressor with a histogram-based gradient-boosting regression tree as the
final estimator.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.hgbr import stacking_params
from more_bikes.experiments.task_2.stacking_regressor import StackingRegressor
from more_bikes.experiments.task_2.task_2_experiment import Task2Experiment
from more_bikes.feature_selection.drop import feature_selection_drop
from more_bikes.feature_selection.variance_threshold import (
    feature_selection_variance_threshold,
)
from more_bikes.preprocessing.ordinal_transformer import preprocessing_ordinal


def stacking_hgbr():
    """
    Stacking regressor with a histogram-based gradient-boosting regression tree as the
    final estimator.
    """
    return Task2Experiment(
        model=Model(
            name="stacking_hgbr",
            pipeline=make_pipeline(
                # preprocessing
                preprocessing_ordinal,
                SimpleImputer(keep_empty_features=True),
                # feature selection
                feature_selection_variance_threshold,
                feature_selection_drop(["wind_speed_avg"]),
                # regression
                StackingRegressor(
                    final_estimator=HistGradientBoostingRegressor(),
                ),
            ),
            params=stacking_params,
        ),
    )

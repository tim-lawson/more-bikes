"""Stacking regressor."""

from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from more_bikes.data.model_loader import get_estimators
from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_2.task_2_experiment import Task2Experiment
from more_bikes.feature_selection.drop import feature_selection_drop
from more_bikes.feature_selection.variance_threshold import (
    feature_selection_variance_threshold,
)
from more_bikes.preprocessing.ordinal_transformer import preprocessing_ordinal


def stacking():
    """Stacking regressor."""
    return Task2Experiment(
        model=Model(
            name="stacking",
            pipeline=make_pipeline(
                # preprocessing
                preprocessing_ordinal,
                SimpleImputer(keep_empty_features=True),
                # feature selection
                feature_selection_variance_threshold,
                feature_selection_drop(["wind_speed_avg"]),
                # regression
                StackingRegressor(estimators=get_estimators()),
            ),
        )
    )

"""Stacking regressor."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor as StackingRegressor_
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


class StackingRegressor(StackingRegressor_):
    def __init__(self, models: list[str] | None, **kwargs):
        super().__init__(estimators=get_estimators(models), **kwargs)

        self.models = models

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._param_names = ["models"] + list(kwargs.keys())

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self


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
                StackingRegressor(
                    models=None,
                    final_estimator=HistGradientBoostingRegressor(),
                ),
                # StackingRegressor(
                #     estimators=get_estimators(["short", "short_temp"]),
                #     final_estimator=HistGradientBoostingRegressor(),
                # ),
            ),
            params=[
                {
                    "stackingregressor__models": [
                        ["full"],
                        ["full_temp"],
                        ["short"],
                        ["short_full"],
                        ["short_full_temp"],
                        ["short_temp"],
                    ],
                    "stackingregressor__final_estimator__l2_regularization": [
                        # 0.1,
                        # 0.2,
                        0.5,
                        # 1.0,
                    ],
                    "stackingregressor__final_estimator__loss": ["absolute_error"],
                    "stackingregressor__final_estimator__max_depth": [
                        # None,
                        # 5,
                        # 10,
                        20,
                        # 50,
                        # 100,
                    ],
                    "stackingregressor__final_estimator__max_iter": [
                        # 50,
                        # 100,
                        200,
                        # 500,
                        # 1000,
                    ],
                    "stackingregressor__final_estimator__max_leaf_nodes": [
                        # None,
                        # 3,
                        # 7,
                        15,
                        # 31,
                        # 63,
                    ],
                    "stackingregressor__final_estimator__min_samples_leaf": [
                        # 2,
                        # 5,
                        10,
                        # 20,
                        # 50,
                    ],
                    "stackingregressor__final_estimator__scoring": [
                        "neg_mean_absolute_error"
                    ],
                }
            ],
        ),
    )

"""Stacking regressor with a gradient-boosted decision tree and pre-trained linear models."""

from pandas import DataFrame, concat
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline

from more_bikes.data.data_loader import DataLoaderFullN
from more_bikes.data.model_loader import get_estimators
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


def feature_eng_rlm(x: DataFrame):
    """For each pre-trained linear model, add a feature whose values are the model's predictions."""

    for name, estimator in get_estimators():
        x = concat([x, DataFrame(estimator.predict(x), columns=[name])], axis=1)

    return x


def stacking():
    """Stacking regressor with a gradient-boosted decision tree and pre-trained linear models."""
    return TaskExperiment(
        task="3",
        model=Model(
            name="stacking",
            pipeline=TransformedTargetRegressor(
                make_pipeline(
                    # preprocessing
                    preprocessing_ordinal,
                    # feature selection
                    feature_selection_variance_threshold,
                    feature_selection_drop(["wind_speed_avg"]),
                    # feature engineering
                    # FunctionTransformer(feature_eng_rlm),
                    # regression
                    HistGradientBoostingRegressor(random_state=42),
                    # DummyRegressor(),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
        search="halving",
        train=DataLoaderFullN(),
    )

"""Task 1A: Decision tree."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.decision_tree import params
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


def decision_tree():
    """Decision tree."""
    return Task1BExperiment(
        model=Model(
            name="decision_tree",
            pipeline=TransformedTargetRegressor(
                make_pipeline(
                    # preprocessing
                    preprocessing_ordinal,
                    # feature selection
                    feature_selection_variance_threshold,
                    feature_selection_drop(
                        [
                            "bikes_3h_diff_avg_full",
                            "bikes_3h_diff_avg_short",
                            "bikes_avg_full",
                            "bikes_avg_short",
                            "wind_speed_avg",
                        ]
                    ),
                    SimpleImputer(),
                    # regression
                    DecisionTreeRegressor(random_state=42),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
    )

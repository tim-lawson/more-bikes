"""Task 1B: LightGBM."""

from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
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

params = [
    {
        "regressor__lgbmregressor__boosting_type": [
            "gbdt",
            # "dart",
        ],
        "regressor__lgbmregressor__num_leaves": [
            # 15,
            31,
            # 63,
            # 127,
        ],
        "regressor__lgbmregressor__max_depth": [
            -1,
            # 5,
            # 10,
            # 20,
        ],
        "regressor__lgbmregressor__learning_rate": [
            # 1e-2,
            1e-1,
            # 1.0,
        ],
        "regressor__lgbmregressor__n_estimators": [
            100,
            # 200,
            # 500,
            # 1000,
        ],
    }
]


def lightgbm():
    """LightGBM."""
    return Task1BExperiment(
        model=Model(
            name="lightgbm",
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
                    LGBMRegressor(random_state=42),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
    )

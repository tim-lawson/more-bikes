"""Parameters for `HistGradientBoostingRegressor`."""

from more_bikes.data.feature import categorical_features
from more_bikes.experiments.experiment import SCORING
from more_bikes.experiments.params.util import ParamGrid

best_params = {
    "regressor__histgradientboostingregressor__categorical_features": [
        "weekday",
        "is_holiday",
    ],
    "regressor__histgradientboostingregressor__l2_regularization": 0.5,
    "regressor__histgradientboostingregressor__learning_rate": 0.1,
    "regressor__histgradientboostingregressor__loss": "absolute_error",
    "regressor__histgradientboostingregressor__max_depth": 20,
    "regressor__histgradientboostingregressor__max_iter": 200,
    "regressor__histgradientboostingregressor__max_leaf_nodes": 15,
    "regressor__histgradientboostingregressor__min_samples_leaf": 10,
    "regressor__histgradientboostingregressor__scoring": "neg_mean_absolute_error",
}


params: ParamGrid = [
    {
        "regressor__histgradientboostingregressor__loss": ["absolute_error"],
        "regressor__histgradientboostingregressor__learning_rate": [
            0.01,
            0.1,
            1,
        ],
        "regressor__histgradientboostingregressor__max_iter": [
            10,
            20,
            50,
            100,
            200,
            500,
        ],
        "regressor__histgradientboostingregressor__max_leaf_nodes": [
            None,
            15,
            31,
            63,
        ],
        "regressor__histgradientboostingregressor__max_depth": [
            None,
            5,
            10,
            20,
            50,
        ],
        "regressor__histgradientboostingregressor__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
            50,
        ],
        "regressor__histgradientboostingregressor__l2_regularization": [
            0.0,
            0.1,
            0.2,
            0.5,
            1.0,
        ],
        "regressor__histgradientboostingregressor__categorical_features": [
            categorical_features
        ],
        "regressor__histgradientboostingregressor__scoring": [SCORING],
    }
]

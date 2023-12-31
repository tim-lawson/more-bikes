"""Parameters for `HistGradientBoostingRegressor`."""

from more_bikes.data.feature import categorical_features
from more_bikes.experiments.experiment import SCORING
from more_bikes.experiments.params.stacking import stacking_fixed
from more_bikes.experiments.params.util import ParamGrid

l2_regularization = [0.1, 0.2, 0.5, 1.0]
learning_rate = [0.02, 0.05, 0.1, 0.2, 0.5]
max_depth = [None, 1, 2, 5, 10, 20, 50]
max_iter = [20, 50, 100, 200, 500]
max_leaf_nodes = [None, 7, 15, 31, 63]
min_samples_leaf = [1, 2, 5, 10, 20, 50, 100]


fixed = {
    "regressor__histgradientboostingregressor__categorical_features": [
        categorical_features
    ],
    "regressor__histgradientboostingregressor__loss": ["absolute_error"],
    "regressor__histgradientboostingregressor__scoring": [SCORING],
}

best_params: ParamGrid = [
    {
        **fixed,
        "regressor__histgradientboostingregressor__l2_regularization": [0.2],
        "regressor__histgradientboostingregressor__learning_rate": [0.1],
        "regressor__histgradientboostingregressor__max_depth": [50],
        "regressor__histgradientboostingregressor__max_iter": [100],
        "regressor__histgradientboostingregressor__max_leaf_nodes": [15],
        "regressor__histgradientboostingregressor__min_samples_leaf": [2],
    }
]


params: ParamGrid = [
    {
        **fixed,
        "regressor__histgradientboostingregressor__l2_regularization": l2_regularization,
        "regressor__histgradientboostingregressor__learning_rate": learning_rate,
        "regressor__histgradientboostingregressor__max_depth": max_depth,
        "regressor__histgradientboostingregressor__max_iter": max_iter,
        "regressor__histgradientboostingregressor__max_leaf_nodes": max_leaf_nodes,
        "regressor__histgradientboostingregressor__min_samples_leaf": min_samples_leaf,
    }
]

stacking_params: ParamGrid = [
    {
        **stacking_fixed,
        "stackingregressor__final_estimator__categorical_features": [
            categorical_features
        ],
        "stackingregressor__final_estimator__loss": ["absolute_error"],
        "stackingregressor__final_estimator__scoring": [SCORING],
        "stackingregressor__final_estimator__l2_regularization": [0.2],
        "stackingregressor__final_estimator__learning_rate": [0.1],
        "stackingregressor__final_estimator__max_depth": [50],
        "stackingregressor__final_estimator__max_iter": [100],
        "stackingregressor__final_estimator__max_leaf_nodes": [15],
        "stackingregressor__final_estimator__min_samples_leaf": [2],
    }
]

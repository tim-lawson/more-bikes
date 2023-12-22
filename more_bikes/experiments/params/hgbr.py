"""Parameter grid and search space for `HistGradientBoostingRegressor`."""

from sklearn_genetic.space import Categorical

from more_bikes.data.feature import categorical_features
from more_bikes.experiments.experiment import SCORING
from more_bikes.experiments.params.util import ParamGrid, ParamSpace

hgbr_param_grid: ParamGrid = [
    {
        "regressor__histgradientboostingregressor__loss": [
            # "squared_error",
            "absolute_error",
            # "poisson",
        ],
        # `quantile` is irrelevant for `loss="absolute_error"`.
        "regressor__histgradientboostingregressor__learning_rate": [
            # 0.01,
            0.1,
            # 1,
        ],
        "regressor__histgradientboostingregressor__max_iter": [
            # 10,
            # 20,
            # 50,
            100,
            # 200,
            # 500,
        ],
        "regressor__histgradientboostingregressor__max_leaf_nodes": [
            None,
            # 15,
            # 31,
            # 63,
        ],
        "regressor__histgradientboostingregressor__max_depth": [
            # None,
            5,
            # 10,
            # 20,
        ],
        "regressor__histgradientboostingregressor__min_samples_leaf": [
            # 10,
            # 20,
            50,
        ],
        "regressor__histgradientboostingregressor__l2_regularization": [
            # 0.0,
            # 0.1,
            # 0.2,
            0.5,
            # 1.0,
        ],
        # "regressor__histgradientboostingregressor__max_bins": [255],
        "regressor__histgradientboostingregressor__categorical_features": [
            categorical_features
        ],
        # "regressor__histgradientboostingregressor__monotonic_cst": [None],
        # "regressor__histgradientboostingregressor__interaction_cst": [None],
        # "regressor__histgradientboostingregressor__warm_start": [False],
        "regressor__histgradientboostingregressor__scoring": [SCORING],
        # "regressor__histgradientboostingregressor__validation_fraction": [0.1],
        # "regressor__histgradientboostingregressor__n_iter_no_change": [10],
        # "regressor__histgradientboostingregressor__tol": [1e-7],
    }
]

hgbr_param_space: ParamSpace = {
    "regressor__histgradientboostingregressor__loss": Categorical(["absolute_error"]),
    # `quantile` is irrelevant for `loss="absolute_error"`.
    # "regressor__histgradientboostingregressor__learning_rate": Categorical(
    #     [1e-6, 1e-4, 1e-2, 1.0]
    # ),
    "regressor__histgradientboostingregressor__max_iter": Categorical([500]),
    # "regressor__histgradientboostingregressor__max_leaf_nodes": Categorical(
    #     [15, 31, 63, 127]
    # ),
    "regressor__histgradientboostingregressor__max_depth": Categorical([5, 10, 20]),
    # "regressor__histgradientboostingregressor__min_samples_leaf": Categorical(
    #     [10, 20, 50]
    # ),
    "regressor__histgradientboostingregressor__l2_regularization": Categorical(
        # [1e-6, 1e-4, 1e-2, 1.0]
        [0.5]
    ),
    # "regressor__histgradientboostingregressor__max_bins": Integer(255, 255),
    "regressor__histgradientboostingregressor__categorical_features": Categorical(
        [categorical_features]
    ),
    # "regressor__histgradientboostingregressor__monotonic_cst": Categorical([None]),
    # "regressor__histgradientboostingregressor__interaction_cst": Categorical([None]),
    # "regressor__histgradientboostingregressor__warm_start": Categorical([False]),
    "regressor__histgradientboostingregressor__scoring": Categorical([SCORING]),
    # "regressor__histgradientboostingregressor__validation_fraction": Continuous(0.1, 0.1),
    # "regressor__histgradientboostingregressor__n_iter_no_change": Integer(10, 10),
    # "regressor__histgradientboostingregressor__tol": Continuous(1e-7, 1e-7),
}

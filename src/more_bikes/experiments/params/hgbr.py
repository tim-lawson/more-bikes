"""Parameter grid and search space for `HistGradientBoostingRegressor`."""

from sklearn_genetic.space import Categorical

from more_bikes.data.feature import categorical_features
from more_bikes.experiments.experiment import SCORING
from more_bikes.experiments.params.util import ParamGrid, ParamSpace

hgbr_param_grid: ParamGrid = [
    {
        "histgradientboostingregressor__loss": [
            # "squared_error",
            "absolute_error",
            # "poisson",
        ],
        # `quantile` is irrelevant for `loss="absolute_error"`.
        "histgradientboostingregressor__learning_rate": [
            # 0.01,
            0.1,
            # 1,
        ],
        "histgradientboostingregressor__max_iter": [
            100,
            # 200,
            # 500,
        ],
        "histgradientboostingregressor__max_leaf_nodes": [
            None,
            # 15,
            # 31,
            # 63,
        ],
        "histgradientboostingregressor__max_depth": [
            None,
            # 5,
            # 10,
            # 20,
        ],
        "histgradientboostingregressor__min_samples_leaf": [
            # 10,
            # 20,
            50,
        ],
        "histgradientboostingregressor__l2_regularization": [
            # 0.0,
            # 0.1,
            # 0.2,
            0.5,
            # 1.0,
        ],
        # "histgradientboostingregressor__max_bins": [255],
        "histgradientboostingregressor__categorical_features": [categorical_features],
        # "histgradientboostingregressor__monotonic_cst": [None],
        # "histgradientboostingregressor__interaction_cst": [None],
        # "histgradientboostingregressor__warm_start": [False],
        "histgradientboostingregressor__scoring": [SCORING],
        # "histgradientboostingregressor__validation_fraction": [0.1],
        # "histgradientboostingregressor__n_iter_no_change": [10],
        # "histgradientboostingregressor__tol": [1e-7],
    }
]

hgbr_param_space: ParamSpace = {
    "histgradientboostingregressor__loss": Categorical(["absolute_error"]),
    # `quantile` is irrelevant for `loss="absolute_error"`.
    # "histgradientboostingregressor__learning_rate": Categorical(
    #     [1e-6, 1e-4, 1e-2, 1.0]
    # ),
    "histgradientboostingregressor__max_iter": Categorical([500]),
    # "histgradientboostingregressor__max_leaf_nodes": Categorical(
    #     [15, 31, 63, 127]
    # ),
    "histgradientboostingregressor__max_depth": Categorical([5, 10, 20]),
    # "histgradientboostingregressor__min_samples_leaf": Categorical(
    #     [10, 20, 50]
    # ),
    "histgradientboostingregressor__l2_regularization": Categorical(
        # [1e-6, 1e-4, 1e-2, 1.0]
        [0.5]
    ),
    # "histgradientboostingregressor__max_bins": Integer(255, 255),
    "histgradientboostingregressor__categorical_features": Categorical(
        [categorical_features]
    ),
    # "histgradientboostingregressor__monotonic_cst": Categorical([None]),
    # "histgradientboostingregressor__interaction_cst": Categorical([None]),
    # "histgradientboostingregressor__warm_start": Categorical([False]),
    "histgradientboostingregressor__scoring": Categorical([SCORING]),
    # "histgradientboostingregressor__validation_fraction": Continuous(0.1, 0.1),
    # "histgradientboostingregressor__n_iter_no_change": Integer(10, 10),
    # "histgradientboostingregressor__tol": Continuous(1e-7, 1e-7),
}

"""Parameters for `DecisionTreeRegressor`."""

from more_bikes.experiments.params.util import ParamGrid

params: ParamGrid = [
    {
        "regressor__decisiontreeregressor__criterion": ["absolute_error"],
        "regressor__decisiontreeregressor__splitter": [
            "best",
            # "random",
        ],
        "regressor__decisiontreeregressor__max_depth": [
            None,
            # 1,
            # 2,
            # 5,
            10,
            20,
            50,
            # 100,
        ],
        # "regressor__decisiontreeregressor__min_samples_split": [
        # 2,
        # 5,
        # 10,
        # 20,
        # 50,
        # ],
        "regressor__decisiontreeregressor__min_samples_leaf": [
            # 1,
            # 2,
            5,
            10,
            20,
            # 50,
            # 100,
        ],
        "regressor__decisiontreeregressor__max_features": [
            None,
            # "sqrt",
            # "log2",
        ],
        "regressor__decisiontreeregressor__max_leaf_nodes": [
            None,
            7,
            15,
            31,
            # 63,
        ],
        "regressor__decisiontreeregressor__min_impurity_decrease": [
            0.0,
            # 0.1,
            # 0.2,
            # 0.5,
            # 1.0,
        ],
        "regressor__decisiontreeregressor__ccp_alpha": [
            0.0,
            # 0.1,
            # 0.2,
            # 0.5,
            # 1.0,
        ],
    }
]

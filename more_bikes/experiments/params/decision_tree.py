"""Parameters for `DecisionTreeRegressor`."""

from more_bikes.experiments.params.util import ParamGrid

params: ParamGrid = [
    {
        "regressor__decisiontreeregressor__criterion": ["absolute_error"],
        "regressor__decisiontreeregressor__max_depth": [
            None,
            10,
            20,
            50,
        ],
        "regressor__decisiontreeregressor__min_samples_leaf": [
            5,
            10,
            20,
        ],
        "regressor__decisiontreeregressor__max_leaf_nodes": [
            None,
            7,
            15,
            31,
        ],
    }
]

"""Parameters for `DecisionTreeRegressor`."""

from more_bikes.experiments.params.util import ParamGrid

max_depth = [None, 1, 2, 5, 10, 20, 50]
max_leaf_nodes = [None, 7, 15, 31, 63]
min_samples_leaf = [1, 2, 5, 10, 20, 50, 100]

fixed = {
    "regressor__decisiontreeregressor__criterion": ["absolute_error"],
}

params: ParamGrid = [
    {
        **fixed,
        "regressor__decisiontreeregressor__max_depth": max_depth,
        "regressor__decisiontreeregressor__min_samples_leaf": min_samples_leaf,
        "regressor__decisiontreeregressor__max_leaf_nodes": max_leaf_nodes,
    }
]

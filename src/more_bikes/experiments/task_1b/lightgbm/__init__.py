"""Task 1B: LightGBM."""

from lightgbm import LGBMRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.bikes_fraction import proc_bikes_fraction
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import column_transformer_1b
from more_bikes.preprocessing.ordinal import ordinal_transformer

params = [
    {
        "lgbmregressor__boosting_type": [
            "gbdt",
            # "dart",
        ],
        "lgbmregressor__num_leaves": [
            # 15,
            31,
            # 63,
            # 127,
        ],
        "lgbmregressor__max_depth": [
            -1,
            # 5,
            # 10,
            # 20,
        ],
        "lgbmregressor__learning_rate": [
            # 1e-2,
            1e-1,
            # 1.0,
        ],
        "lgbmregressor__n_estimators": [
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
            pipeline=make_pipeline(
                ordinal_transformer.set_output(transform="pandas"),
                column_transformer_1b.set_output(transform="pandas"),
                LGBMRegressor(random_state=42, verbosity=2),
            ),
            params=params,
        ),
        processing=proc_bikes_fraction(True),
        cv=time_series_split,
    )


if __name__ == "__main__":
    lightgbm().run().save()

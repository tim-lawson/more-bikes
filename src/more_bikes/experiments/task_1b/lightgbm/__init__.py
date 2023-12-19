"""Task 1B: LightGBM."""

from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import column_transformer_1b
from more_bikes.preprocessing.ordinal import ordinal_transformer
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

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
                    ordinal_transformer.set_output(transform="pandas"),
                    column_transformer_1b.set_output(transform="pandas"),
                    LGBMRegressor(random_state=42),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
    )


if __name__ == "__main__":
    lightgbm().run().save()

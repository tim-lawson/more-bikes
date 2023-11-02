"""Task 1B: LightGBM."""

from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import column_transformer_1b
from more_bikes.preprocessing.ordinal import ordinal_transformer
from more_bikes.util.dataframe import create_dropna_row

params = [
    {
        # "lgbmregressor__boosting_type": ["gbdt", "dart"],
        # "lgbmregressor__num_leaves": [15, 31, 63, 127],
        "lgbmregressor__max_depth": [-1],
        "lgbmregressor__learning_rate": [1e-1],
        "lgbmregressor__n_estimators": [100],
    }
]


lightgbm = Task1BExperiment(
    model=Model(
        name="lightgbm",
        pipeline=make_pipeline(
            ordinal_transformer.set_output(transform="pandas"),
            column_transformer_1b.set_output(transform="pandas"),
            LGBMRegressor(random_state=42, n_jobs=10),
        ),
        preprocessing=[create_dropna_row()],
        params=params,
    ),
    cv=time_series_split,
)


if __name__ == "__main__":
    lightgbm.run().save()

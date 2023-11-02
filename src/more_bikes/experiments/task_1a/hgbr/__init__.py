"""Task 1A: histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.params.hgbr import hgbr_param_grid
from more_bikes.experiments.task_1a.task_1a_experiment import Task1AExperiment
from more_bikes.preprocessing.column import column_transformer_1a
from more_bikes.preprocessing.ordinal import ordinal_transformer
from more_bikes.util.dataframe import create_dropna_row

hgbr = Task1AExperiment(
    model=Model(
        name="hgbr",
        pipeline=make_pipeline(
            ordinal_transformer.set_output(transform="pandas"),
            column_transformer_1a.set_output(transform="pandas"),
            HistGradientBoostingRegressor(random_state=42, verbose=0),
        ),
        preprocessing=[create_dropna_row()],
        params=hgbr_param_grid,
    ),
    cv=time_series_split,
)


if __name__ == "__main__":
    hgbr.run().save()

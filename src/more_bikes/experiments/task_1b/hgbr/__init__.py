"""Task 1B: Histogram-based gradient-boosting regression tree."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline

from more_bikes.data.feature import categorical_features, categories
from more_bikes.experiments.experiment import Model
from more_bikes.experiments.params.hgbr import hgbr_param_grid, hgbr_param_space
from more_bikes.experiments.params.util import GASearchCVParams, SearchStrategy
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import make_drop_column_transformer
from more_bikes.preprocessing.ordinal import make_ordinal_transformer
from more_bikes.util.datetime import make_datetime_columns
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import HackTransformedTargetRegressor

SEARCH: SearchStrategy = "grid"

params = hgbr_param_grid if SEARCH == "grid" else hgbr_param_space


def hgbr():
    """Histogram-based gradient-boosting regression tree."""
    return Task1BExperiment(
        model=Model(
            name="hgbr",
            pipeline=HackTransformedTargetRegressor(
                make_pipeline(
                    FunctionTransformer(make_datetime_columns),
                    make_ordinal_transformer(
                        "ordinal",
                        categorical_features + ["weekend", "period"],
                        categories
                        + [["True", "False"], ["night", "midday", "afternoon"]],
                    ).set_output(transform="pandas"),
                    make_drop_column_transformer(
                        [
                            "timestamp",
                            "year",
                            "month",
                            "day",
                            "hour",
                            "weekhour",
                            "wind_speed_avg",
                            "precipitation",
                            "bikes_3h_diff_avg_short",
                            "bikes_avg_short",
                        ],
                    ).set_output(transform="pandas"),
                    HistGradientBoostingRegressor(random_state=42),
                ),
                BikesFractionTransformer(),
            ),
            params=params,
        ),
        search=SEARCH,
        ga_search_cv_params=GASearchCVParams(
            generations=3,
            population_size=10,
        ),
    )


if __name__ == "__main__":
    hgbr().run().save()

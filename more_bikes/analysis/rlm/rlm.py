"""A script to compute the scores of RLM models."""

# pylint: disable=redefined-outer-name

from pandas import DataFrame, concat, read_csv
from sklearn import set_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from more_bikes.data.data_loader import DataLoaderTrainN
from more_bikes.data.model_loader import get_estimator_ids
from more_bikes.experiments.experiment import Model, Processing
from more_bikes.experiments.params.cv import time_series_split
from more_bikes.experiments.task_2.stacking import StackingRegressor
from more_bikes.feature_selection.drop import feature_selection_drop
from more_bikes.feature_selection.variance_threshold import (
    feature_selection_variance_threshold,
)
from more_bikes.preprocessing.ordinal_transformer import preprocessing_ordinal
from more_bikes.preprocessing.util import pre_chain, split


def get_rlm_score(model: Model, processing: Processing = Processing()):
    """Score the RLM model."""

    x_train, y_train = split(pre_chain(processing.pre)(DataLoaderTrainN().data))

    scores = cross_val_score(
        model.pipeline,
        x_train,
        y_train,
        cv=time_series_split,
        scoring=make_scorer(mean_absolute_error),
    )

    return scores, scores.mean()


def save_rlm_scores():
    """Score all RLM models."""

    rlm_scores = DataFrame(
        {
            "model": [],
            "station": [],
            "split": [],
            "score": [],
        }
    )

    rlm_mean_scores = DataFrame(
        {
            "model": [],
            "station": [],
            "score": [],
        }
    )

    for name, station, estimator in get_estimator_ids():
        model = Model(
            name=f"{name}_{station}",
            pipeline=make_pipeline(
                # preprocessing
                preprocessing_ordinal,
                SimpleImputer(keep_empty_features=True),
                # feature selection
                feature_selection_variance_threshold,
                feature_selection_drop(["wind_speed_avg"]),
                # regression
                estimator,
            ),
        )

        scores, mean_score = get_rlm_score(model)

        print(f"{name} {station}".ljust(20) + f"{mean_score:.3f}")

        rlm_scores = concat(
            [
                rlm_scores,
                DataFrame(
                    {
                        "model": name,
                        "station": station,
                        "split": range(len(scores)),
                        "score": scores,
                    }
                ),
            ]
        )

        rlm_mean_scores = concat(
            [
                rlm_mean_scores,
                DataFrame(
                    {
                        "model": name,
                        "station": station,
                        "score": [mean_score],
                    }
                ),
            ]
        )

    rlm_scores["station"] = rlm_scores["station"].astype(int)
    rlm_scores["split"] = rlm_scores["split"].astype(int)

    rlm_mean_scores["station"] = rlm_mean_scores["station"].astype(int)

    rlm_scores.to_csv(
        "more_bikes/analysis/rlm/rlm_scores.csv",
        index=False,
    )

    rlm_mean_scores.to_csv(
        "more_bikes/analysis/rlm/rlm_mean_scores.csv",
        index=False,
    )


def save_stacked_rlm_scores():
    """Score all stacked RLM models."""

    stacked_rlm_scores = DataFrame(
        {
            "model": [],
            "split": [],
            "score": [],
        }
    )

    stacked_rlm_mean_scores = DataFrame(
        {
            "model": [],
            "score": [],
        }
    )

    for rlm_models in [
        ["full"],
        ["full_temp"],
        ["short"],
        ["short_full"],
        ["short_full_temp"],
        ["short_temp"],
    ]:
        name = rlm_models[0]

        model = Model(
            name=f"stacking_{name}",
            pipeline=make_pipeline(
                # preprocessing
                preprocessing_ordinal,
                SimpleImputer(keep_empty_features=True),
                # feature selection
                feature_selection_variance_threshold,
                feature_selection_drop(["wind_speed_avg"]),
                # regression
                StackingRegressor(models=rlm_models),
            ),
        )

        scores, mean_score = get_rlm_score(model)

        print(name.ljust(20) + f"{mean_score:.3f}")

        stacked_rlm_scores = concat(
            [
                stacked_rlm_scores,
                DataFrame(
                    {
                        "model": name,
                        "split": range(len(scores)),
                        "score": scores,
                    }
                ),
            ]
        )

        stacked_rlm_mean_scores = concat(
            [
                stacked_rlm_mean_scores,
                DataFrame(
                    {
                        "model": name,
                        "score": [mean_score],
                    }
                ),
            ]
        )

    stacked_rlm_scores["split"] = stacked_rlm_scores["split"].astype(int)

    stacked_rlm_scores.to_csv(
        "more_bikes/analysis/rlm/stacked_rlm_scores.csv",
        index=False,
    )

    stacked_rlm_mean_scores.to_csv(
        "more_bikes/analysis/rlm/stacked_rlm_mean_scores.csv",
        index=False,
    )


def save_aggregates():
    """Save aggregate statistics for RLM scores."""

    rlm_mean_scores = read_csv(
        "more_bikes/analysis/rlm/rlm_mean_scores.csv",
        header=0,
        names=["model", "station", "score"],
    )

    for name, group_rlm_mean_scores in rlm_mean_scores.groupby("model"):
        group_rlm_mean_scores["score"].to_csv(
            f"more_bikes/analysis/rlm/rlm_mean_scores_{name}.csv",
            index=False,
        )

    groups_rlm_mean_scores = rlm_mean_scores.groupby("model")

    box_plots_rlm_mean_scores = DataFrame(
        {
            "mean": groups_rlm_mean_scores["score"].mean(),
            "std": groups_rlm_mean_scores["score"].std(),
            "min": groups_rlm_mean_scores["score"].min(),
            "max": groups_rlm_mean_scores["score"].max(),
            "median": groups_rlm_mean_scores["score"].median(),
            "first_quartile": groups_rlm_mean_scores["score"].quantile(0.25),
            "third_quartile": groups_rlm_mean_scores["score"].quantile(0.75),
        }
    )

    box_plots_rlm_mean_scores.to_csv(
        "more_bikes/analysis/rlm/rlm_box_plots.csv",
        index=True,
    )

    stacked_rlm_scores = read_csv(
        "more_bikes/analysis/rlm/rlm_mean_scores.csv",
        header=0,
        names=["model", "split", "score"],
    )

    groups_stacked_rlm_scores = stacked_rlm_scores.groupby("model")

    box_plots_stacked_rlm_scores = DataFrame(
        {
            "mean": groups_stacked_rlm_scores["score"].mean(),
            "std": groups_stacked_rlm_scores["score"].std(),
            "min": groups_stacked_rlm_scores["score"].min(),
            "max": groups_stacked_rlm_scores["score"].max(),
            "median": groups_stacked_rlm_scores["score"].median(),
            "first_quartile": groups_stacked_rlm_scores["score"].quantile(0.25),
            "third_quartile": groups_stacked_rlm_scores["score"].quantile(0.75),
        }
    )

    box_plots_stacked_rlm_scores.to_csv(
        "more_bikes/analysis/rlm/stacked_rlm_box_plots.csv",
        index=True,
    )


if __name__ == "__main__":
    set_config(transform_output="pandas")
    # save_rlm_scores()
    save_stacked_rlm_scores()
    save_aggregates()

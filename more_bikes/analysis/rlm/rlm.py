"""A script to compute the scores of RLM models."""

# pylint: disable=redefined-outer-name

from pandas import DataFrame, concat, read_csv
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from more_bikes.data.data_loader import DataLoaderTrainN
from more_bikes.data.model_loader import get_estimator_ids
from more_bikes.experiments.experiment import Model, Processing
from more_bikes.experiments.params.cv import time_series_split
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
            name="stacking",
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


def save_box_plots():
    """Save box plots for RLM scores."""

    rlm_mean_scores = read_csv(
        "more_bikes/analysis/rlm/rlm_mean_scores.csv",
        header=0,
        names=["model", "station", "score"],
    )

    # Group by model/name and save separate CSVs
    for name, grouped in rlm_mean_scores.groupby("model"):
        grouped["score"].to_csv(
            f"more_bikes/analysis/rlm/rlm_mean_scores_{name}.csv",
            index=False,
        )

    grouped = rlm_mean_scores.groupby("model")

    box_plots = DataFrame(
        {
            "mean": grouped["score"].mean(),
            "std": grouped["score"].std(),
            "min": grouped["score"].min(),
            "max": grouped["score"].max(),
            "median": grouped["score"].median(),
            "first_quartile": grouped["score"].quantile(0.25),
            "third_quartile": grouped["score"].quantile(0.75),
        }
    )

    box_plots.to_csv(
        "more_bikes/analysis/rlm/rlm_box_plots.csv",
        index=True,
    )


if __name__ == "__main__":
    set_config(transform_output="pandas")
    # save_rlm_scores()
    save_box_plots()

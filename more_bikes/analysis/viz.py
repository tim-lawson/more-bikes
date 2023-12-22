"""Visualisations."""

# pylint: disable=redefined-outer-name

from numpy import histogram
from pandas import CategoricalDtype, DataFrame, concat, set_option
from scipy.stats import describe

from more_bikes.data.data_loader import DataLoaderTrainN
from more_bikes.data.feature import WEEKDAY, numerical_features


def _get_columns_stats(
    data: DataFrame, columns: str | list[str], target: str = "bikes"
):
    grouped = data.groupby(columns, observed=False)
    return DataFrame(
        {
            "count": grouped[target].count(),
            "mean": grouped[target].mean(),
            "lower": grouped[target].mean() - grouped[target].std(),
            "upper": grouped[target].mean() + grouped[target].std(),
            # Box plot
            "minimum": grouped[target].min(),
            "maximum": grouped[target].max(),
            "median": grouped[target].median(),
            "first_quartile": grouped[target].quantile(0.25),
            "third_quartile": grouped[target].quantile(0.75),
        }
    )


def _get_column_histogram(data: DataFrame, column: str):
    counts, divisions = histogram(data[column].dropna().to_numpy(), density=True)
    return DataFrame({"counts": counts, "divisions": divisions[:-1]})


def _get_column_stats(data: DataFrame, column: str):
    stats = describe(data[column].dropna().to_numpy())
    return DataFrame(
        {
            "feature": [column],
            "n": [stats.nobs],
            "na": [data[column].isna().sum()],
            "min": [stats.minmax[0]],
            "max": [stats.minmax[1]],
            "mean": [stats.mean],
            "variance": [stats.variance],
            "skewness": [stats.skewness],
            "kurtosis": [stats.kurtosis],
        }
    )


if __name__ == "__main__":
    data = DataLoaderTrainN().data

    weekday = CategoricalDtype(categories=WEEKDAY, ordered=True)
    data["weekday"] = data["weekday"].astype(weekday)

    data["fraction"] = data["bikes"] / data["docks"]

    correlation = data[
        [
            "day",
            "hour",
            "weekhour",
            "wind_speed_max",
            "wind_speed_avg",
            "wind_direction",
            "temperature",
            "humidity",
            "pressure",
            "bikes",
            "bikes_avg_full",
            "bikes_avg_short",
            "bikes_3h",
            "bikes_3h_diff_avg_full",
            "bikes_3h_diff_avg_short",
        ]
    ].corr()
    correlation = correlation.reset_index().melt(id_vars="index")
    correlation.columns = ["x", "y", "value"]
    correlation.to_csv("more_bikes/analysis/csv/correlation.csv", index=False)

    correlation = correlation.sort_values("value", ascending=False)
    print(correlation[correlation["x"] != correlation["y"]].head(10))

    for columns in [
        ["day"],
        ["hour"],
        ["is_holiday"],
        ["weekday"],
        ["weekday", "hour"],
    ]:
        columns_stats = _get_columns_stats(data, columns, "fraction")
        columns_stats.to_csv(
            f"more_bikes/analysis/csv/fraction_{'_'.join(columns)}.csv",
        )

    stats: list[DataFrame] = []

    for column in [
        "wind_speed_max",
        "wind_speed_avg",
        "wind_direction",
        "temperature",
        "humidity",
        "pressure",
        "bikes",
        "bikes_avg_full",
        "bikes_avg_short",
        "bikes_3h",
        "bikes_3h_diff_avg_full",
        "bikes_3h_diff_avg_short",
    ]:
        hist = _get_column_histogram(data, column)
        hist.to_csv(f"more_bikes/analysis/csv/histogram_{column}.csv", index=False)

        stats.append(_get_column_stats(data, column))

    concat(stats, ignore_index=True).to_csv(
        "more_bikes/analysis/csv/stats.csv", index=False
    )

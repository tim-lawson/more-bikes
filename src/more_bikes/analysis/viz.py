"""Visualisations."""

# pylint: disable=redefined-outer-name

from matplotlib import pyplot as plt
from numpy import histogram
from pandas import CategoricalDtype, DataFrame
from scipy.stats import describe

from more_bikes.data.data_loader import DataLoaderAll
from more_bikes.data.feature import WEEKDAY


def _get_box_plot(data: DataFrame, column: str, target: str = "bikes"):
    grouped = data.groupby(column, observed=False)
    return DataFrame(
        {
            "minimum": grouped[target].min(),
            "maximum": grouped[target].max(),
            "median": grouped[target].median(),
            "first_quartile": grouped[target].quantile(0.25),
            "third_quartile": grouped[target].quantile(0.75),
        }
    )


def _get_average_fraction(data: DataFrame, columns: list[str]):
    return DataFrame(data.groupby(columns, observed=False)["fraction"].mean())


def _plot_average_fraction(data: DataFrame, columns: list[str]):
    _, ax = plt.subplots()
    average_fraction = _get_average_fraction(data, columns)
    average_fraction.plot(ax=ax)
    plt.show()


def _get_stats(data: DataFrame, column: str):
    stats = describe(data[column].dropna().to_numpy())
    return DataFrame([stats], columns=stats._fields)


def _get_histogram(data: DataFrame, column: str):
    counts, divisions = histogram(data[column].dropna().to_numpy(), density=True)
    return DataFrame({"counts": counts, "divisions": divisions[:-1]})


if __name__ == "__main__":
    data = DataLoaderAll().data

    weekday = CategoricalDtype(categories=WEEKDAY, ordered=True)
    data["weekday"] = data["weekday"].astype(weekday)

    data["fraction"] = data["bikes"] / data["docks"]

    for columns in [["weekday"], ["weekday", "hour"]]:
        average_fraction = _get_average_fraction(data, columns)
        average_fraction.to_csv(
            f"more_bikes/analysis/csv/average_fraction_{'_'.join(columns)}.csv",
        )

    for column in ["weekday", "is_holiday"]:
        box_plot = _get_box_plot(data, column)
        box_plot.to_csv(f"more_bikes/analysis/csv/box_plot_{column}.csv", index=False)

    for column in [
        "wind_speed_max",
        "wind_speed_avg",
        "wind_direction",
        "temperature",
        "humidity",
        "pressure",
        "precipitation",
        "bikes_3h",
        "bikes_3h_diff_avg_full",
        "bikes_avg_full",
        "bikes_3h_diff_avg_short",
        "bikes_avg_short",
        "bikes",
        "fraction",
    ]:
        hist = _get_histogram(data, column)
        hist.to_csv(f"more_bikes/analysis/csv/histogram_{column}.csv", index=False)

        stats = _get_stats(data, column)
        stats.to_csv(f"more_bikes/analysis/csv/stats_{column}.csv", index=False)

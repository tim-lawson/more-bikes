"""Visualisations."""

# pylint: disable=redefined-outer-name

from matplotlib import pyplot as plt
from pandas import CategoricalDtype, DataFrame

from more_bikes.data.data_loader import DataLoaderFullN
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


if __name__ == "__main__":
    data = DataLoaderFullN().data

    weekday = CategoricalDtype(categories=WEEKDAY, ordered=True)
    data["weekday"] = data["weekday"].astype(weekday)

    data["fraction"] = data["bikes"] / data["docks"]

    average_fraction_weekday = _get_average_fraction(data, ["weekday", "hour"])
    average_fraction_weekday.to_csv(
        "more_bikes/analysis/csv/average_fraction_weekday.csv"
    )

    average_fraction_hour = _get_average_fraction(data, ["hour"])
    average_fraction_hour.to_csv("more_bikes/analysis/csv/average_fraction_hour.csv")

    box_plot_weekday = _get_box_plot(data, "weekday", "fraction")
    box_plot_weekday.to_csv("more_bikes/analysis/csv/box_plot_weekday.csv")

    box_plot_hour = _get_box_plot(data, "hour", "fraction")
    box_plot_hour.to_csv("more_bikes/analysis/csv/box_plot_hour.csv")

    box_plot_is_holiday = _get_box_plot(data, "is_holiday", "fraction")
    box_plot_is_holiday.to_csv("more_bikes/analysis/csv/box_plot_is_holiday.csv")

    # _plot_average_fraction(data, ["weekday", "hour"])
    # _plot_average_fraction(data, ["hour"])

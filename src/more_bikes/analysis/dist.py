"""Distributional analyses."""

from matplotlib import pyplot
from pandas import DataFrame
from seaborn import heatmap, pairplot

from more_bikes.data.data_loader import DataLoaderTrainN
from more_bikes.data.feature import categorical_features, numerical_features

DIRECTORY_CSV = "more_bikes/analysis/csv"
DIRECTORY_PNG = "more_bikes/analysis/png"
PREFIX = "dist"

pairwise_features = [
    feature
    for feature in numerical_features
    if feature not in ["year", "month", "day", "hour", "weekhour", "precipitation"]
]


def save_categorical_dist(data: DataFrame) -> None:
    """Save the distributions of categorical features to a CSV."""

    for column in categorical_features:
        group = data.groupby(column)

        DataFrame(
            {
                "count": data[column].value_counts(),
                "mean": group["bikes"].mean(),
                "std": group["bikes"].std(),
            }
        ).to_csv(f"{DIRECTORY_CSV}/{PREFIX}_{column}.csv")


def save_numerical_dist(data: DataFrame) -> None:
    """Save the distributions of numerical features to a CSV."""

    DataFrame(
        {column: data[column].describe() for column in numerical_features}
    ).to_csv(f"{DIRECTORY_CSV}/{PREFIX}_numerical.csv")


def plot_pairwise(data: DataFrame) -> None:
    """Plot the pairwise distributions of numerical features."""

    plot = pairplot(data[pairwise_features])

    plot.savefig(f"{DIRECTORY_PNG}/{PREFIX}_pairplot.png", dpi=300)


def plot_correlations(data: DataFrame) -> None:
    """Plot the correlations between numerical features."""

    fig, ax = pyplot.subplots()
    fig.set_size_inches(16, 16)

    heatmap(data[pairwise_features].corr(), ax=ax, annot=True, center=0)

    fig.savefig(f"{DIRECTORY_PNG}/{PREFIX}_heatmap.png", dpi=300)


if __name__ == "__main__":
    data_n = DataLoaderTrainN().data

    print("Saving distributions...")
    save_categorical_dist(data_n)
    save_numerical_dist(data_n)

    print("Plotting distributions...")
    plot_pairwise(data_n)
    plot_correlations(data_n)

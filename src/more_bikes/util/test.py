"""Statistical tests."""

from pandas import read_csv
from scipy.stats import ttest_rel


def test_task_1b(experiment1: str, experiment2: str):
    """Dependent t-test for paired samples."""

    dataframe1 = read_csv(
        f"more_bikes/experiments/task_1b/{experiment1}/{experiment1}_cv.csv"
    )
    dataframe2 = read_csv(
        f"more_bikes/experiments/task_1b/{experiment2}/{experiment2}_cv.csv"
    )

    scores1 = dataframe1["score"].values
    scores2 = dataframe2["score"].values

    statistic, pvalue = ttest_rel(scores1, scores2)

    print(f"\n{experiment1} vs {experiment2}")
    print(f"statistic = {statistic:.3f}")
    print(f"p-value = {pvalue:.3f}")


if __name__ == "__main__":
    for pair in [
        ("baseline", "hgbr"),
        ("baseline", "lightgbm"),
        ("baseline", "mlp"),
        ("hgbr", "lightgbm"),
        ("hgbr", "mlp"),
        ("lightgbm", "mlp"),
    ]:
        test_task_1b(*pair)

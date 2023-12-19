"""Statistical tests."""

from pandas import read_csv
from scipy.stats import ranksums


def test_task_1b(experiment1: str, experiment2: str):
    """Wilcoxon rank-sum test."""

    dataframe_a = read_csv(
        f"more_bikes/experiments/task_1b/{experiment1}/{experiment1}_cv.csv"
    )
    dataframe_b = read_csv(
        f"more_bikes/experiments/task_1b/{experiment2}/{experiment2}_cv.csv"
    )

    scores1 = dataframe_a["score"].values
    scores2 = dataframe_b["score"].values

    result = ranksums(scores1, scores2)

    print(f"{experiment1} vs {experiment2}")
    print(f"statistic = {result.statistic:.3f}")
    print(f"p-value = {result.pvalue:.3f}")


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

"""Statistical tests."""

# pylint: disable=redefined-outer-name

from itertools import combinations

from numpy import greater
from pandas import DataFrame, concat, read_csv
from scipy.stats import ttest_rel


def mean_score(task: str, experiment: str):
    """Mean score and variance for a given experiment."""

    dataframe = read_csv(
        f"more_bikes/experiments/{task}/{experiment}/{experiment}_cv.csv"
    )

    return dataframe["score"].mean(), dataframe["score"].var()


def test_task_1a(experiment1: str, experiment2: str):
    """Dependent t-test for paired samples."""

    dataframe1 = read_csv(
        f"more_bikes/experiments/task_1a/{experiment1}/{experiment1}_cv.csv"
    )
    dataframe2 = read_csv(
        f"more_bikes/experiments/task_1a/{experiment2}/{experiment2}_cv.csv"
    )

    statistics: list[float] = []
    pvalues: list[float] = []

    for station in range(201, 276):
        station1 = dataframe1[dataframe1["station"] == station]
        station2 = dataframe2[dataframe2["station"] == station]

        scores1 = station1["score"].values
        scores2 = station2["score"].values

        statistic, pvalue = ttest_rel(scores1, scores2)

        statistics.append(statistic)
        pvalues.append(pvalue)

    avg_statistic = sum(statistics) / len(statistics)
    num_positive = sum([statistic > 0 for statistic in statistics])
    num_significant = sum([pvalue < 0.05 for pvalue in pvalues])

    print(
        f"{experiment1} vs {experiment2}: {avg_statistic:.3f} ({num_positive} positive, {num_significant} significant)"
    )

    statistics = [round(statistic, 3) for statistic in statistics]
    pvalues = [round(pvalue, 3) for pvalue in pvalues]

    return statistics, pvalues


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

    return statistic, pvalue


if __name__ == "__main__":
    print("task 1a")
    for experiment in ["baseline", "decision_tree", "hgbr"]:
        mean, variance = mean_score("task_1a", experiment)
        print(f"{experiment}: {mean:.2f} ({variance:.2f})")

    print("\ntask 1b")
    for experiment in ["baseline", "decision_tree", "hgbr", "lightgbm", "mlp"]:
        mean, variance = mean_score("task_1b", experiment)
        print(f"{experiment}: {mean:.2f} ({variance:.2f})")

    pair: tuple[str, str]

    P = 0.05

    results_1a = DataFrame(
        {
            "model1": [],
            "model2": [],
            "station": [],
            "statistic": [],
            "pvalue": [],
            "model": [],
            "significant": [],
        }
    )

    for pair in combinations(["baseline", "decision_tree", "hgbr"], 2):
        statistics, pvalues = test_task_1a(*pair)

        for station, statistic, pvalue in zip(range(201, 276), statistics, pvalues):
            results_1a = concat(
                [
                    results_1a,
                    DataFrame(
                        {
                            "model1": [pair[0]],
                            "model2": [pair[1]],
                            "station": [station],
                            "statistic": [statistic],
                            "pvalue": [pvalue],
                            "model": [pair[0] if statistic < 0 else pair[1]],
                            "significant": [pvalue < P],
                        }
                    ),
                ],
                ignore_index=True,
            )

    results_1a["station"] = results_1a["station"].astype(int)
    results_1a["significant"] = results_1a["significant"].astype(bool)
    results_1a.to_csv("more_bikes/util/test_results_1a.csv", index=False)

    results_1b = DataFrame(
        {
            "model1": [],
            "model2": [],
            "statistic": [],
            "pvalue": [],
            "model": [],
            "significant": [],
        }
    )

    for pair in combinations(
        ["baseline", "decision_tree", "hgbr", "lightgbm", "mlp"], 2
    ):
        statistic, pvalue = test_task_1b(*pair)

        results_1b = concat(
            [
                results_1b,
                DataFrame(
                    {
                        "model1": [pair[0]],
                        "model2": [pair[1]],
                        "statistic": [statistic],
                        "pvalue": [pvalue],
                        "model": [pair[0] if statistic < 0 else pair[1]],
                        "significant": [pvalue < P],
                    }
                ),
            ],
            ignore_index=True,
        )

    results_1b["significant"] = results_1b["significant"].astype(bool)
    results_1b.to_csv("more_bikes/util/test_results_1b.csv", index=False)

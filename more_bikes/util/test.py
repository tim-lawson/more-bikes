"""Statistical tests."""

# pylint: disable=redefined-outer-name

from itertools import combinations

from numpy import array
from pandas import DataFrame, concat, read_csv
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare, ttest_rel

UTIL = "more_bikes/util"


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

    num_positive = sum(statistic > 0 for statistic in statistics)

    num_significant = sum(pvalue < 0.05 for pvalue in pvalues)

    print(
        f"{experiment1} vs {experiment2}: {avg_statistic:.3f} ({num_positive} positive, {num_significant} significant)"
    )

    return statistics, pvalues


def read_task(task: str, experiment: str):
    """Read dataframe for a given experiment."""

    dataframe = read_csv(
        f"more_bikes/experiments/{task}/{experiment}/{experiment}_cv.csv"
    )

    return dataframe["score"].values


def t_test_task(task1: str, task2: str, experiment1: str, experiment2: str):
    """Dependent t-test for paired samples."""

    scores1 = read_task(task1, experiment1)
    scores2 = read_task(task2, experiment2)

    statistic, pvalue = ttest_rel(scores1, scores2)

    return statistic, pvalue


task_names = {
    "task_1a": "1 (a)",
    "task_1b": "1 (b)",
    "task_2": "2",
}

task_names_inverse = {v: k for k, v in task_names.items()}

experiment_names = {
    "baseline": "Baseline",
    "decision_tree": "Decision tree",
    "hgbr": "GBDT",
    "lightgbm": "LightGBM",
    "mlp": "Multi-layer perceptron",
    "stacking": "Stacked Ridge",
    "stacking_decision_tree": "Stacked Decision tree",
    "stacking_hgbr": "Stacked GBDT",
}

experiment_names_inverse = {v: k for k, v in experiment_names.items()}


def friedman_test_task(task_experiments: tuple[tuple[str, str], ...], name: str):
    """Friedman test for multiple experiments."""

    scores = []

    for task, experiment in task_experiments:
        scores.append(read_task(task, experiment))

    statistic, pvalue = friedmanchisquare(*scores)

    if pvalue < 0.05:
        posthoc = posthoc_nemenyi_friedman(array(scores).T)

        posthoc.columns = [
            f"{task}~{experiment}" for task, experiment in task_experiments
        ]
        posthoc.index = posthoc.columns

        # One row per comparison
        posthoc = posthoc.stack().reset_index()

        # Rename columns
        posthoc.columns = ["model1", "model2", "pvalue"]

        # Remove comparisons with self
        posthoc = posthoc[posthoc["model1"] != posthoc["model2"]]

        # Remove duplicates, i.e., keep A vs B but not B vs A
        posthoc = posthoc[
            posthoc.apply(lambda row: row["model1"] < row["model2"], axis=1)
        ]

        # Re-order columns
        posthoc = posthoc[
            [
                "model1",
                "model2",
                "pvalue",
            ]
        ]

        # Split model names into task and experiment
        posthoc["task1"] = posthoc["model1"].apply(
            lambda model: model.split("~")[0],
        )
        posthoc["experiment1"] = posthoc["model1"].apply(
            lambda model: model.split("~")[1],
        )
        posthoc["task2"] = posthoc["model2"].apply(
            lambda model: model.split("~")[0],
        )
        posthoc["experiment2"] = posthoc["model2"].apply(
            lambda model: model.split("~")[1],
        )

        # Add significant column
        posthoc["significant"] = posthoc["pvalue"] < 0.05

        # Add t-statistic column
        posthoc["tstatistic"] = posthoc.apply(
            lambda row: t_test_task(
                row["task1"], row["task2"], row["experiment1"], row["experiment2"]
            )[0],
            axis=1,
        )

        # Add descriptive names
        posthoc["task1"] = posthoc["task1"].apply(lambda task: task_names[task])
        posthoc["task2"] = posthoc["task2"].apply(lambda task: task_names[task])
        posthoc["experiment1"] = posthoc["experiment1"].apply(
            lambda experiment: experiment_names[experiment].replace("Stacked", "")
        )
        posthoc["experiment2"] = posthoc["experiment2"].apply(
            lambda experiment: experiment_names[experiment].replace("Stacked", "")
        )

        # Drop model columns
        posthoc.drop(columns=["model1", "model2"], inplace=True)

        # Re-order columns
        posthoc = posthoc[
            [
                "task1",
                "experiment1",
                "task2",
                "experiment2",
                "tstatistic",
                "pvalue",
                "significant",
            ]
        ]

        # Remove task columns if all tasks are the same
        all_task_1b = all(task == "task_1b" for task, _ in task_experiments)
        all_task_2 = all(task == "task_2" for task, _ in task_experiments)

        if all_task_1b or all_task_2:
            posthoc.drop(columns=["task1", "task2"], inplace=True)

        posthoc.to_csv(f"{UTIL}/test_results_posthoc_{name}.csv", index=False)

    return statistic, pvalue


if __name__ == "__main__":
    print("task 1a")

    for experiments in ["baseline", "decision_tree", "hgbr"]:
        mean, variance = mean_score("task_1a", experiments)

        print(f"{experiments}: {mean:.2f} ({variance:.2f})")

    print("\ntask 1b")

    for experiments in ["baseline", "decision_tree", "hgbr", "lightgbm", "mlp"]:
        mean, variance = mean_score("task_1b", experiments)

        print(f"{experiments}: {mean:.2f} ({variance:.2f})")

    print("\ntask 2")

    for experiments in ["stacking", "stacking_decision_tree", "stacking_hgbr"]:
        mean, variance = mean_score("task_2", experiments)

        print(f"{experiments}: {mean:.2f} ({variance:.2f})")

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

    results_1a.to_csv(f"{UTIL}/test_results_1a.csv", index=False)

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
        statistic, pvalue = t_test_task("task_1b", "task_1b", *pair)

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

    results_1b.to_csv(f"{UTIL}/test_results_1b.csv", index=False)

    results_2 = DataFrame(
        {
            "task1": [],
            "model1": [],
            "task2": [],
            "model2": [],
            "statistic": [],
            "pvalue": [],
            "model": [],
            "significant": [],
        }
    )

    experiments_1b = (
        ("task_1b", "baseline"),
        ("task_1b", "decision_tree"),
        ("task_1b", "hgbr"),
        # ("task_1b", "lightgbm"),
        # ("task_1b", "mlp"),
    )

    experiments_2 = (
        ("task_2", "stacking"),
        ("task_2", "stacking_decision_tree"),
        ("task_2", "stacking_hgbr"),
    )

    for pair2 in combinations(list(experiments_1b) + list(experiments_2), 2):
        statistic, pvalue = t_test_task(
            pair2[0][0],
            pair2[1][0],
            pair2[0][1],
            pair2[1][1],
        )

        results_2 = concat(
            [
                results_2,
                DataFrame(
                    {
                        "task1": [pair2[0][0]],
                        "model1": [pair2[0][1]],
                        "task2": [pair2[1][0]],
                        "model2": [pair2[1][1]],
                        "statistic": [statistic],
                        "pvalue": [pvalue],
                        "model": [pair2[0][1] if statistic < 0 else pair2[1][1]],
                        "significant": [pvalue < P],
                    }
                ),
            ],
            ignore_index=True,
        )

    results_2["significant"] = results_2["significant"].astype(bool)

    results_2.to_csv(f"{UTIL}/test_results_2.csv", index=False)

    friedman_test_task(experiments_1b, "1b")
    friedman_test_task(experiments_2, "2")
    friedman_test_task((*experiments_1b, *experiments_2), "all")

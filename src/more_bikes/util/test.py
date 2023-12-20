"""Statistical tests."""

from logging import Logger

from pandas import read_csv
from scipy.stats import ttest_rel

from more_bikes.util.log import create_logger


def test_task_1a(experiment1: str, experiment2: str, logger: Logger):
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

    statistics = [round(statistic, 3) for statistic in statistics]
    pvalues = [round(pvalue, 3) for pvalue in pvalues]

    logger.info("%s vs %s", experiment1, experiment2)
    logger.info("statistics = %s", statistics)
    logger.info("p-values = %s", pvalues)


def test_task_1b(experiment1: str, experiment2: str, logger: Logger):
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

    logger.info("%s vs %s", experiment1, experiment2)
    logger.info("statistic = %f", statistic)
    logger.info("p-value = %f", pvalue)


if __name__ == "__main__":
    logger_1a = create_logger("test_1a")
    logger_1b = create_logger("test_1b")

    for pair in [
        ("baseline", "hgbr"),
    ]:
        test_task_1a(*pair, logger=logger_1a)

    for pair in [
        ("baseline", "hgbr"),
        ("baseline", "lightgbm"),
        ("baseline", "mlp"),
        ("hgbr", "lightgbm"),
        ("hgbr", "mlp"),
        ("lightgbm", "mlp"),
    ]:
        test_task_1b(*pair, logger=logger_1b)

"""Model-loader classes."""

# pylint:disable=too-few-public-methods,invalid-name

from functools import reduce
from typing import Sequence, TypeVar

from pandas import DataFrame, read_csv
from sklearn.base import BaseEstimator, RegressorMixin

intercept = "(Intercept)"
bikes_3h = "bikes_3h_ago"
bikes_3h_diff_avg_full = "full_profile_3h_diff_bikes"
bikes_3h_diff_avg_short = "short_profile_3h_diff_bikes"
bikes_avg_full = "full_profile_bikes"
bikes_avg_short = "short_profile_bikes"
temperature = "temperature.C"

columns = ["feature", "weight"]


class ModelLoader(BaseEstimator, RegressorMixin):
    """Abstract model-loader class."""

    data: DataFrame

    def __init__(self, station_id: int):
        self.station_id = station_id

    def _get_weight(self, feature: str) -> float:
        return self.data.loc[self.data["feature"] == feature, "weight"].item()

    def fit(self, _x, _y):
        """No-op."""
        return self

    def predict(self, x: DataFrame):
        """Predict."""
        raise NotImplementedError


class ModelLoaderFull(ModelLoader):
    """`full` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_full.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_full = self._get_weight(bikes_3h_diff_avg_full)
        self.bikes_avg_full = self._get_weight(bikes_avg_full)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_full * x["bikes_3h_diff_avg_full"]
            + self.bikes_avg_full * x["bikes_avg_full"]
        ).to_numpy()


class ModelLoaderFullTemp(ModelLoader):
    """`full_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_full_temp.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_full = self._get_weight(bikes_3h_diff_avg_full)
        self.bikes_avg_full = self._get_weight(bikes_avg_full)
        self.temperature = self._get_weight(temperature)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_full * x["bikes_3h_diff_avg_full"]
            + self.bikes_avg_full * x["bikes_avg_full"]
            + self.temperature * x["temperature"]
        ).to_numpy()


class ModelLoaderShortFull(ModelLoader):
    """`short_full` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_short_full.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_full = self._get_weight(bikes_3h_diff_avg_full)
        self.bikes_3h_diff_avg_short = self._get_weight(bikes_3h_diff_avg_short)
        self.bikes_avg_full = self._get_weight(bikes_avg_full)
        self.bikes_avg_short = self._get_weight(bikes_avg_short)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_full * x["bikes_3h_diff_avg_full"]
            + self.bikes_3h_diff_avg_short * x["bikes_3h_diff_avg_short"]
            + self.bikes_avg_full * x["bikes_avg_full"]
            + self.bikes_avg_short * x["bikes_avg_short"]
        ).to_numpy()


class ModelLoaderShort(ModelLoader):
    """`short` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_short.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_short = self._get_weight(bikes_3h_diff_avg_short)
        self.bikes_avg_short = self._get_weight(bikes_avg_short)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_short * x["bikes_3h_diff_avg_short"]
            + self.bikes_avg_short * x["bikes_avg_short"]
        ).to_numpy()


class ModelLoaderShortFullTemp(ModelLoader):
    """`short_full_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_short_full_temp.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_full = self._get_weight(bikes_3h_diff_avg_full)
        self.bikes_3h_diff_avg_short = self._get_weight(bikes_3h_diff_avg_short)
        self.bikes_avg_full = self._get_weight(bikes_avg_full)
        self.bikes_avg_short = self._get_weight(bikes_avg_short)
        self.temperature = self._get_weight(temperature)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_full * x["bikes_3h_diff_avg_full"]
            + self.bikes_3h_diff_avg_short * x["bikes_3h_diff_avg_short"]
            + self.bikes_avg_full * x["bikes_avg_full"]
            + self.bikes_avg_short * x["bikes_avg_short"]
            + self.temperature * x["temperature"]
        ).to_numpy()


class ModelLoaderShortTemp(ModelLoader):
    """`short_temp` model-loader class."""

    def __init__(self, station_id: int):
        super().__init__(station_id)

        path = f"data/models/model_station_{station_id}_rlm_short_temp.csv"
        self.data = read_csv(path, header=0, names=columns)

        self.intercept = self._get_weight(intercept)
        self.bikes_3h = self._get_weight(bikes_3h)
        self.bikes_3h_diff_avg_short = self._get_weight(bikes_3h_diff_avg_short)
        self.bikes_avg_short = self._get_weight(bikes_avg_short)
        self.temperature = self._get_weight(temperature)

    def predict(self, x):
        return (
            self.intercept
            + self.bikes_3h * x["bikes_3h"]
            + self.bikes_3h_diff_avg_short * x["bikes_3h_diff_avg_short"]
            + self.bikes_avg_short * x["bikes_avg_short"]
            + self.temperature * x["temperature"]
        ).to_numpy()


def get_station_estimators(station_id: int) -> list[tuple[str, ModelLoader]]:
    """Get all estimators for a station."""
    return [
        (f"full_{station_id}", ModelLoaderFull(station_id)),
        (f"full_temp_{station_id}", ModelLoaderFullTemp(station_id)),
        (f"short_full_{station_id}", ModelLoaderShortFull(station_id)),
        (f"short_{station_id}", ModelLoaderShort(station_id)),
        (f"short_full_temp_{station_id}", ModelLoaderShortFullTemp(station_id)),
        (f"short_temp_{station_id}", ModelLoaderShortTemp(station_id)),
    ]


T = TypeVar("T")


def concat(list_of_lists: list[list[T]]) -> list[T]:
    """Concatenate a list of lists into a single list."""
    return reduce(lambda x, y: x + y, list_of_lists)


def get_estimators() -> Sequence[tuple[str, ModelLoader]]:
    """Get all estimators."""
    return concat([get_station_estimators(station_id) for station_id in range(1, 201)])

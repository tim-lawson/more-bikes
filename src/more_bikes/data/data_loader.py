"""Data-loader classes."""

from abc import ABCMeta, abstractmethod
from functools import cached_property

from pandas import DataFrame, concat, read_csv

from more_bikes.data.feature import FEATURE_DTYPE, FEATURE_TEST, FEATURE_TRAIN


class DataLoader(metaclass=ABCMeta):
    """Abstract data-loader class."""

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def data(self) -> DataFrame:
        """Read the data to a DataFrame."""


class DataLoaderTrain1(DataLoader):
    """Train data loader for a single station."""

    def __init__(self, station_id: int):
        super().__init__()
        self.station_id = station_id
        self.path = f"../data/train/station_{station_id}_deploy.csv"

    @cached_property
    def data(self):
        return read_csv(
            filepath_or_buffer=self.path,
            header=0,
            names=FEATURE_TRAIN,
            dtype=FEATURE_DTYPE,
            na_values="NA",
        ).sort_values(by=["timestamp"])


class DataLoaderTrainN(DataLoader):
    """Train data loader for multiple stations."""

    def __init__(self, station_ids: list[int] | None = None):
        super().__init__()
        self.station_ids = station_ids or list(range(201, 274))
        self.paths = [
            f"../data/train/station_{station_id}_deploy.csv"
            for station_id in self.station_ids
        ]

    @cached_property
    def data(self):
        return concat(
            [
                read_csv(
                    filepath_or_buffer=path,
                    header=0,
                    names=FEATURE_TRAIN,
                    dtype=FEATURE_DTYPE,
                    na_values="NA",
                )
                for path in self.paths
            ],
            ignore_index=True,
        ).sort_values(by=["timestamp"])


class DataLoaderTest1(DataLoader):
    """Test data loader for a single station."""

    def __init__(self, station_id: int):
        super().__init__()
        self.path = "../data/test.csv"
        self.station_id = station_id

    @cached_property
    def data(self) -> DataFrame:
        data = read_csv(
            filepath_or_buffer=self.path,
            header=0,
            names=FEATURE_TEST,
            dtype=FEATURE_DTYPE,
            na_values="NA",
        )
        return data[data["station"] == self.station_id].sort_values(by=["timestamp"])


class DataLoaderTestN(DataLoader):
    """Test data loader for multiple stations."""

    def __init__(self):
        super().__init__()
        self.path = "../data/test.csv"

    @cached_property
    def data(self) -> DataFrame:
        return read_csv(
            filepath_or_buffer=self.path,
            header=0,
            names=FEATURE_TEST,
            dtype=FEATURE_DTYPE,
            na_values="NA",
        ).sort_values(by=["timestamp"])


class DataLoaderFull1(DataLoader):
    """Full data loader for a single station."""

    def __init__(self, station_id: int):
        super().__init__()
        self.station_id = station_id
        self.path = f"../data/train/station_{station_id}_train.csv"

    @cached_property
    def data(self):
        return read_csv(
            filepath_or_buffer=self.path,
            header=0,
            names=FEATURE_TRAIN,
            dtype=FEATURE_DTYPE,
            na_values="NA",
        ).sort_values(by=["timestamp"])


class DataLoaderFullN(DataLoader):
    """Full data loader for multiple stations."""

    def __init__(self, station_ids: list[int] | None = None):
        super().__init__()
        self.station_ids = station_ids or list(range(1, 11))
        self.paths = [
            f"../data/train/station_{station_id}_train.csv"
            for station_id in self.station_ids
        ]

    @cached_property
    def data(self):
        return concat(
            [
                read_csv(
                    filepath_or_buffer=path,
                    header=0,
                    names=FEATURE_TRAIN,
                    dtype=FEATURE_DTYPE,
                    na_values="NA",
                )
                for path in self.paths
            ],
            ignore_index=True,
        ).sort_values(by=["timestamp"])


class DataLoaderAll(DataLoader):
    """Data loader for all sources."""

    def __init__(self):
        super().__init__()
        self.train = DataLoaderTrainN()
        self.full = DataLoaderFullN()

    @cached_property
    def data(self):
        return concat(
            [self.train.data, self.full.data], ignore_index=True, sort=False
        ).sort_values(by=["timestamp"])

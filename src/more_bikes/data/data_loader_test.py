"""Test the data-loader classes."""

from more_bikes.util.dataframe import split

from .data_loader import (
    DataLoaderTest1,
    DataLoaderTestN,
    DataLoaderTrain1,
    DataLoaderTrainN,
)


def test_data_loader_train_1():
    """Test the training data loader for a single station."""

    x, y = split(DataLoaderTrain1(station_id=201).data)

    assert x.shape == (745, 24)
    assert y.shape == (745,)


def test_data_loader_train_n():
    """Test the training data loader for multiple stations."""

    x, y = split(DataLoaderTrainN().data)

    assert x.shape == (54385, 24)
    assert y.shape == (54385,)


def test_data_loader_test_1():
    """Test the test data loader for a single station."""

    x = DataLoaderTest1(station_id=201).data

    assert x.shape == (30, 25)


def test_data_loader_test_n():
    """Test the test data loader for multiple stations."""

    x = DataLoaderTestN().data

    assert x.shape == (2250, 25)

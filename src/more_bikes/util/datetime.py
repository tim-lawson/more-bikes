"""Date-time utilities."""

from datetime import datetime

from pandas import DataFrame, Series


def utc_from_timestamp(timestamp: int) -> str:
    """Convert a timestamp to UTC."""
    return datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def utc_from_timestamp_series(series: Series) -> Series:
    """Convert a Series of timestamps to UTC."""
    return series.apply(utc_from_timestamp)


def utc_from_timestamp_data(data: DataFrame):
    """Convert a DataFrame of timestamp rows to UTC."""
    return data.apply(utc_from_timestamp_series, axis=1)

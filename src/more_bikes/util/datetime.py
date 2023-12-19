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


def make_datetime_columns(data: DataFrame):
    """Add date-time columns to a DataFrame."""
    data["weekend"] = data["weekday"].apply(
        lambda weekday: "True" if weekday in ["Saturday", "Sunday"] else "False"
    )
    data["period"] = data["hour"].apply(
        lambda hour: "night"
        if hour < 6 or hour > 22
        else "midday"
        if hour < 14
        else "afternoon"
    )
    return data

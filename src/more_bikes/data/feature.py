"""Feature utilities."""

from typing import Literal

Feature = Literal[
    "station",
    "latitude",
    "longitude",
    "docks",
    "timestamp",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "weekhour",
    "is_holiday",
    "wind_speed_max",
    "wind_speed_avg",
    "wind_direction",
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "bikes_3h",
    "bikes_3h_diff_avg_full",
    "bikes_avg_full",
    "bikes_3h_diff_avg_short",
    "bikes_avg_short",
    "bikes",
]


FEATURE = (
    "station",
    "latitude",
    "longitude",
    "docks",
    "timestamp",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "weekhour",
    "is_holiday",
    "wind_speed_max",
    "wind_speed_avg",
    "wind_direction",
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "bikes_3h",
    "bikes_3h_diff_avg_full",
    "bikes_avg_full",
    "bikes_3h_diff_avg_short",
    "bikes_avg_short",
)

FEATURE_TEST = ("id", *FEATURE)

FEATURE_TRAIN = (*FEATURE, "bikes")

DType = Literal["bool", "float", "int", "str"]

FEATURE_DTYPE: dict[Feature, DType] = {
    "station": "int",
    "latitude": "float",
    "longitude": "float",
    "docks": "int",
    "timestamp": "int",
    "year": "int",
    "month": "int",
    "day": "int",
    "hour": "int",
    "weekday": "str",
    "weekhour": "int",
    "is_holiday": "bool",
    "wind_speed_max": "float",
    "wind_speed_avg": "float",
    "wind_direction": "float",
    "temperature": "float",
    "humidity": "float",
    "pressure": "float",
    "precipitation": "float",
    "bikes_3h": "float",
    "bikes_3h_diff_avg_full": "float",
    "bikes_avg_full": "float",
    "bikes_3h_diff_avg_short": "float",
    "bikes_avg_short": "float",
    "bikes": "float",
}

BIKES = "bikes"

WEEKDAY = (
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ],
)


categorical_features = ["weekday", "is_holiday"]

categories = [
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    [False, True],
]

numerical_features = [
    "latitude",
    "longitude",
    "docks",
    "timestamp",
    "year",
    "month",
    "day",
    "hour",
    "weekhour",
    "wind_speed_max",
    "wind_speed_avg",
    "wind_direction",
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "bikes_3h",
    "bikes_3h_diff_avg_full",
    "bikes_avg_full",
    "bikes_3h_diff_avg_short",
    "bikes_avg_short",
]

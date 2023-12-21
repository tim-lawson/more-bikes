"""Feature selection by variance threshold."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

feature_selection_variance = ColumnTransformer(
    transformers=[
        (
            "variance",
            VarianceThreshold(),
            [
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
                # Ignore features that are `nan` for the first week.
                # "bikes_3h",
                # "bikes_3h_diff_avg_full",
                # "bikes_avg_full",
                # "bikes_3h_diff_avg_short",
                # "bikes_avg_short",
            ],
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

"""Column transformers."""

from sklearn.compose import ColumnTransformer

from more_bikes.data.feature import Feature


def make_drop_column_transformer(
    columns: list[Feature],
) -> ColumnTransformer:
    """Drop columns."""
    return ColumnTransformer(
        [
            (
                "drop",
                "drop",
                columns,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# Drop columns with zero variance and ~duplicates.
column_transformer_1a = make_drop_column_transformer(
    [
        "latitude",
        "longitude",
        "docks",
        "year",
        "month",
        "wind_speed_avg",
        "precipitation",
        "bikes_3h_diff_avg_short",
        "bikes_avg_short",
    ]
)

# Drop columns with zero variance and ~duplicates.
column_transformer_1b = make_drop_column_transformer(
    [
        "year",
        "month",
        "wind_speed_avg",
        "precipitation",
        "bikes_3h_diff_avg_short",
        "bikes_avg_short",
    ],
)

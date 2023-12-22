"""Transformer to manually drop columns."""

from sklearn.compose import ColumnTransformer

from more_bikes.data.feature import Feature


def feature_selection_drop(
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

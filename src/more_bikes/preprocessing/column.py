"""Column transformers."""

from sklearn.compose import ColumnTransformer

from more_bikes.data.feature import Feature


def make_preprocessing_drop(
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

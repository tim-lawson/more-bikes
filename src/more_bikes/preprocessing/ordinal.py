"""Ordinal transformer for categorical features."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from more_bikes.data.feature import categorical_features, categories


def make_ordinal_transformer(
    name: str, categorical_features_: list[str], categories_: list[list[Any]]
) -> ColumnTransformer:
    """Ordinal transformer for categorical features."""
    return ColumnTransformer(
        transformers=[
            (
                name,
                OrdinalEncoder(categories=categories_),
                categorical_features_,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


ordinal_transformer = make_ordinal_transformer(
    "ordinal",
    categorical_features,
    categories,
)

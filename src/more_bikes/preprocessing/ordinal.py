"""Ordinal transformer for categorical features."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from more_bikes.data.feature import categorical_features, categories


def make_ordinal_transformer(
    name: str = "ordinal",
    categorical_features_: list[str] | None = None,
    categories_: list[list[Any]] | None = None,
) -> ColumnTransformer:
    """Ordinal transformer for categorical features."""
    categorical_features_ = categorical_features_ or categorical_features
    categories_ = categories_ or categories
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


ordinal_transformer = make_ordinal_transformer()

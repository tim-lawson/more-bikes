"""A wrapper for `TransformedTargetRegressor` that allows x-dependent transforms."""

# pylint: disable=arguments-differ
# pyright: reportGeneralTypeIssues=false

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor


class HackTransformedTargetRegressor(TransformedTargetRegressor):
    """A wrapper for `TransformedTargetRegressor` that allows x-dependent transforms."""

    def __init__(
        self,
        regressor: BaseEstimator,
        transformer: TransformerMixin | None = None,
    ):
        super().__init__(
            regressor,
            transformer=transformer,
        )
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, X, y):
        if self.transformer is None:
            self.regressor.fit(X, y)
        else:
            self.transformer.fit(X, y)
            self.regressor.fit(X, self.transformer.transform(X, y))

        return self

    def predict(self, X):
        if self.transformer is None:
            return self.regressor.predict(X)
        return self.transformer.inverse_transform(X, self.regressor.predict(X))

    def score(self, X, y):
        if self.transformer is None:
            return self.regressor.score(X, y)
        return self.regressor.score(X, self.transformer.transform(X, y))

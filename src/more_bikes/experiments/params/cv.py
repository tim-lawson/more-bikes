"""Cross-validation."""

from sklearn.model_selection import TimeSeriesSplit

time_series_split = TimeSeriesSplit(n_splits=5)

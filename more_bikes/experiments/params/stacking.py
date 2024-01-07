"""Parameters for `StackingRegressor`."""

stacking_models = [
    "full",
    "full_temp",
    "short",
    "short_full",
    "short_full_temp",
    "short_temp",
]

stacking_fixed = {
    "stackingregressor__models": stacking_models,
}

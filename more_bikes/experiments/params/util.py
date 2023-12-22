"""Grid-search utilities."""

from typing import Any, Literal

SearchStrategy = Literal["grid", "halving"]

ParamGrid = list[dict[str, list[Any]]]

"""NumPy array utilities."""

from typing import Any, TypeVar

from numpy import dtype, generic, ndarray

DTYPE_co = TypeVar("DTYPE_co", covariant=True, bound=generic)

NDArray = ndarray[Any, dtype[DTYPE_co]]

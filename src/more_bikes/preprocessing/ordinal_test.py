"""Test the ordinal transformer."""

from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame

from .ordinal import make_ordinal_transformer


def test_ordinal_transformer():
    """Test the ordinal transformer."""

    data = DataFrame(
        [
            [1.0, "Monday", True],
            [2.0, "Tuesday", False],
            [3.0, "Wednesday", True],
            [1.0, "Thursday", False],
            [2.0, "Friday", True],
            [3.0, "Saturday", False],
            [1.0, "Sunday", True],
        ],
        columns=["bikes", "weekday", "is_holiday"],
    )

    transformer = make_ordinal_transformer(
        "ordinal",
        [
            "weekday",
            "is_holiday",
        ],
        [
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            [False, True],
        ],
    )

    assert_array_equal(
        array(transformer.fit_transform(data)),
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 0.0, 3.0],
            [6.0, 1.0, 1.0],
        ],
    )

    assert_array_equal(
        transformer.get_feature_names_out(),
        [
            "weekday",
            "is_holiday",
            "bikes",
        ],
    )

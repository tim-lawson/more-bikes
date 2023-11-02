"""Test the drop-NaN transformer."""

from numpy import NaN, array, float_
from numpy.testing import assert_array_equal

from .dropna import DropNaTransformer


def test_dropna_transformer():
    """Test the drop-NaN transformer."""

    data = array([[1.0, NaN], [3.0, 4.0], [NaN, 6.0]], float_)

    dropna_transformer_0 = DropNaTransformer(column_index=0)

    assert_array_equal(
        dropna_transformer_0.fit_transform(data),
        array([[1.0, NaN], [3.0, 4.0]]),
    )

    dropna_transformer_1 = DropNaTransformer(column_index=1)

    assert_array_equal(
        dropna_transformer_1.fit_transform(data),
        array([[3.0, 4.0], [NaN, 6.0]]),
    )

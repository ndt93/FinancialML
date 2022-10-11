import pandas as pd
import pytest

from data_structures.constants import BarCol
from features.frac_diff import frac_diff_expanding, frac_diff_fixed, run_adf_tests


@pytest.fixture
def linear_series():
    ts = pd.DataFrame(
        {BarCol.CLOSE: range(10)},
        index=pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 11)]),
        dtype=float
    )
    return ts


def test_frac_diff_expanding(linear_series):
    # Test order = 0
    diff_df = frac_diff_expanding(linear_series, 0, threshold=1)
    pd.testing.assert_series_equal(diff_df[BarCol.CLOSE], linear_series[BarCol.CLOSE])

    diff_df = frac_diff_expanding(linear_series, 0, threshold=0)
    pd.testing.assert_series_equal(diff_df[BarCol.CLOSE], linear_series[BarCol.CLOSE])

    # Test order = 1
    diff_df = frac_diff_expanding(linear_series, 1, threshold=1)
    pd.testing.assert_series_equal(
        diff_df[BarCol.CLOSE],
        pd.Series([0.] + [1.]*(linear_series.shape[0] - 1), index=linear_series.index),
        check_names=False
    )
    # Test order = 1, skip first point
    diff_df = frac_diff_expanding(linear_series, 1, threshold=0)
    pd.testing.assert_series_equal(
        diff_df[BarCol.CLOSE],
        pd.Series([1.]*(linear_series.shape[0] - 1), index=linear_series.index[1:]),
        check_names=False
    )

    # Test order = 0.5
    diff_df = frac_diff_expanding(linear_series, 0.5, threshold=1.0)
    pd.testing.assert_series_equal(
        diff_df[BarCol.CLOSE],
        pd.Series([0.0, 1.0, 1.5, 1.875, 2.1875, 2.4609, 2.707, 2.9326, 3.1421, 3.3385], index=linear_series.index),
        check_names=False,
        check_less_precise=3
    )


def test_frac_diff_fixed(linear_series):
    # Test order = 0
    diff_df = frac_diff_fixed(linear_series, 0, threshold=0)
    pd.testing.assert_series_equal(diff_df[BarCol.CLOSE], linear_series[BarCol.CLOSE].iloc[-1:])

    diff_df = frac_diff_fixed(linear_series, 0, threshold=0.01)
    pd.testing.assert_series_equal(diff_df[BarCol.CLOSE], linear_series[BarCol.CLOSE])

    # Test order = 1
    diff_df = frac_diff_fixed(linear_series, 1, threshold=0.01)
    pd.testing.assert_series_equal(
        diff_df[BarCol.CLOSE],
        pd.Series([1.]*(linear_series.shape[0] - 1), index=linear_series.index[1:]),
        check_names=False
    )

    # Test order = 0.5
    diff_df = frac_diff_fixed(linear_series, 0.5, threshold=0.05)
    pd.testing.assert_series_equal(
        diff_df[BarCol.CLOSE],
        pd.Series([1.875, 2.1875, 2.5, 2.8125, 3.125, 3.4375, 3.75], index=linear_series.index[3:]),
        check_names=False,
        check_less_precise=3
    )

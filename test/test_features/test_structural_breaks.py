import numpy as np
import pandas as pd
import pytest

from features.structural_breaks import _get_lag_df, _get_yx


@pytest.fixture
def timeseries():
    return pd.DataFrame(np.linspace(0.1, 1., 10), columns=['price'])


def test_lag_df(timeseries):
    print(_get_lag_df(timeseries, 3))


def test_get_yx(timeseries):
    y, x = _get_yx(timeseries, 'ctt', 3)
    print(timeseries)
    print(y)
    print(x)

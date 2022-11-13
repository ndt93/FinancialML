import numpy as np
import pandas as pd
from scipy.stats import norm
import pytest

from features.structural_breaks import (
    recursive_residuals_cusum_stats,
    levels_cusum_stats,
    _get_lag_df,
    _get_yx,
    _fit_adf,
    sadf_stat,
    sadf_stat_series
)


@pytest.fixture
def ts_stable_beta():
    size = 50
    X = np.linspace(0.01, 0.25, size)
    y = 1.5*X + norm.rvs(size=size, loc=0, scale=0.01, random_state=42)
    return X.reshape(-1, 1), y


@pytest.fixture
def ts_unstable_beta():
    size = 50
    X = np.linspace(0.01, 0.25, size)
    y = np.linspace(1., 5., size)*X + norm.rvs(size=size, loc=0, scale=0.01, random_state=42)
    return X.reshape(-1, 1), y


def test_recursive_cusum(ts_stable_beta, ts_unstable_beta):
    X, y = ts_stable_beta
    _, pvals, _ = recursive_residuals_cusum_stats(X, y)
    assert (pvals > 0.1).sum() == 49

    X, y = ts_unstable_beta
    _, pvals, _ = recursive_residuals_cusum_stats(X, y)
    assert (pvals < 0.1).sum() > 10


def test_levels_cusum(ts_stable_beta):
    _, y = ts_stable_beta
    stats, pvals = levels_cusum_stats(y, 0)
    assert (pvals < 0.1).sum() > 40

    y = norm.rvs(size=50, loc=0, scale=0.01, random_state=42)
    stats, pvals = levels_cusum_stats(y, 0)
    assert (pvals > 0.1).sum() == 49


def test_lag_df():
    ts = pd.DataFrame({'price': [0.1, 0.2, 0.4, 0.7, 0.11]})
    lag_df = _get_lag_df(ts, 3)
    np.testing.assert_array_equal(lag_df.columns, ['price_0', 'price_1', 'price_2', 'price_3'])
    pd.testing.assert_series_equal(lag_df['price_0'], ts['price'], check_names=False, check_index=False)
    pd.testing.assert_series_equal(lag_df['price_1'].dropna(), ts['price'][:-1], check_index=False, check_names=False)
    pd.testing.assert_series_equal(lag_df['price_2'].dropna(), ts['price'][:-2], check_index=False, check_names=False)
    pd.testing.assert_series_equal(lag_df['price_3'].dropna(), ts['price'][:-3], check_index=False, check_names=False)


def test_get_yx():
    ts = pd.DataFrame({'price': [0.1, 0.2, 0.4, 0.7, 1.1, 1.6]})
    y, x = _get_yx(ts, 'nc', 3)
    y = y.flatten()
    np.testing.assert_array_almost_equal(y, [0.4, 0.5])
    np.testing.assert_array_almost_equal(x.iloc[0, :].values, [0.7, 0.3, 0.2, 0.1])
    np.testing.assert_array_almost_equal(x.iloc[1, :].values, [1.1, 0.4, 0.3, 0.2])

    _, x = _get_yx(ts, 'c', 3)
    np.testing.assert_array_almost_equal(x[:, -1], [1., 1.])
    _, x = _get_yx(ts, 'ct', 3)
    np.testing.assert_array_almost_equal(x[:, -2], [1., 1.])
    np.testing.assert_array_almost_equal(x[:, -1], [0., 1.])
    _, x = _get_yx(ts, 'ctt', 3)
    np.testing.assert_array_almost_equal(x[:, -3], [1., 1.])
    np.testing.assert_array_almost_equal(x[:, -2], [0., 1.])
    np.testing.assert_array_almost_equal(x[:, -1], [0., 1.])


@pytest.fixture
def ts_pos_beta():
    y0 = 100
    size = 50
    b = 0.1
    a = 0
    sigma = 1
    series = [y0]
    rs = 42
    shocks = norm.rvs(size=size-1, scale=sigma, random_state=rs)
    for i in range(size - 1):
        dy = a + b*series[-1] + shocks[i]
        series.append(series[-1] + dy)
    return pd.DataFrame({'price': series})


@pytest.fixture
def ts_pos_beta_1lag():
    y0 = 100
    size = 100
    b = 0.1
    g1 = -0.01
    a = 0
    sigma = 1
    series = [y0]
    rs = 42
    shocks = norm.rvs(size=size-1, scale=sigma, random_state=rs)
    for i in range(size - 1):
        if i > 1:
            dy = a + b*series[-1] + g1*(series[-1] - series[-2]) + shocks[i]
        else:
            dy = a + b*series[-1] + shocks[i]
        series.append(series[-1] + dy)
    return pd.DataFrame({'price': series})


@pytest.fixture
def ts_zero_beta():
    y0 = 100
    size = 100
    b = 0
    a = 0
    sigma = 1
    series = [y0]
    rs = 42
    shocks = norm.rvs(size=size-1, scale=sigma, random_state=rs)
    for i in range(size - 1):
        dy = a + b*series[-1] + shocks[i]
        series.append(series[-1] + dy)
    return pd.DataFrame({'price': series})


@pytest.fixture
def ts_neg_beta():
    y0 = 100
    size = 100
    b = -0.1
    a = 0
    sigma = 1
    series = [y0]
    rs = 42
    shocks = norm.rvs(size=size-1, scale=sigma, random_state=rs)
    for i in range(size - 1):
        dy = a + b*series[-1] + shocks[i]
        series.append(series[-1] + dy)
    return pd.DataFrame({'price': series})


def test_fit_adf(ts_pos_beta, ts_pos_beta_1lag):
    y, x = _get_yx(ts_pos_beta, 'nc', 0)
    b_mean, b_var = _fit_adf(y, x)
    assert b_mean[0, 0] == pytest.approx(0.1, rel=1e-3)
    assert b_var[0, 0] == pytest.approx(0, abs=1e-3)

    y, x = _get_yx(ts_pos_beta_1lag, 'nc', 1)
    b_mean, _ = _fit_adf(y, x)
    assert b_mean[0, 0] == pytest.approx(0.1, abs=0.02)
    assert b_mean[1, 0] == pytest.approx(-0.01, abs=0.009)


def test_sadf_stat(ts_pos_beta, ts_pos_beta_1lag, ts_zero_beta, ts_neg_beta):
    out = sadf_stat(ts_pos_beta, 10, 'nc', 0)
    assert abs(out) > 3.43

    out = sadf_stat(ts_pos_beta_1lag, 10, 'nc', 1)
    assert abs(out) > 3.43

    out = sadf_stat(ts_zero_beta, 10, 'nc', 0)
    assert abs(out) < 3.43

    out = sadf_stat(ts_neg_beta, 10, 'nc', 0)
    assert abs(out) < 3.43


def test_sadf_stat_series(ts_zero_beta):
    out = sadf_stat_series(ts_zero_beta, 10, 'nc', 0)
    assert out.shape[0] == ts_zero_beta.shape[0] - 10
    assert (out < 3.43).sum() == out.shape[0]

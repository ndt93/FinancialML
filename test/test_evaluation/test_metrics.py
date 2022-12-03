import pytest
import pandas as pd
import numpy as np
from scipy.stats import norm

from financial_ml.data_structures.constants import PortfolioCol
from financial_ml.evaluation.metrics import (
    get_position_timings,
    compute_holding_period,
    compute_time_weighted_return,
    compute_hhi_returns_concentration,
    compute_dd_and_tuw,
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    annualized_sharpe_ratio,
    information_ratio
)


@pytest.fixture
def positions():
    dates = pd.date_range(start='2022-01-01', periods=10)
    pos = [1, 2, 0, -1, 1, 3, 2, -2, -3, 0]
    return pd.Series(pos, index=dates)


def test_get_position_timings(positions):
    out = get_position_timings(positions)
    expected = pd.DatetimeIndex(['2022-01-03', '2022-01-05', '2022-01-08', '2022-01-10'])
    pd.testing.assert_index_equal(out, expected)


def test_compute_holding_period(positions):
    out = compute_holding_period(positions)
    assert out == pytest.approx(1 + 2/3)


def test_twrr():
    returns = np.array([-0.058, 0.235, 0.008, -0.014, 0.112, -0.024, 0.084, -0.029, 0.357, 0.006])
    cashflows = np.random.choice(np.arange(-2, 2), size=10)
    account_values = []
    cur_value = 1.
    for ret, cf in zip(returns, cashflows):
        cur_value = cur_value*(1 + ret) + cf
        account_values.append(cur_value)

    out = compute_time_weighted_return(
        pd.DataFrame({
            PortfolioCol.BEGIN_VALUE: np.array(account_values) - cashflows,
            PortfolioCol.CASHFLOW: cashflows
        })
    )
    assert np.product(returns[1:] + 1) - 1 == pytest.approx(out)


def test_hhi_returns_concentration():
    np.random.seed(42)
    uniform_returns = np.random.uniform(low=-0.2, high=0.2, size=200)
    pos_hhi, neg_hhi = compute_hhi_returns_concentration(uniform_returns)
    assert pos_hhi < 0.01
    assert neg_hhi < 0.01

    norm_returns = np.random.normal(loc=0, scale=0.15, size=200)
    pos_hhi, neg_hhi = compute_hhi_returns_concentration(norm_returns)
    assert pos_hhi < 0.01
    assert neg_hhi < 0.01

    conc_returns = np.concatenate([
        np.random.normal(loc=0, scale=0.001, size=199),
        np.random.normal(loc=0.5, scale=0.01, size=1)
    ])
    pos_hhi, _ = compute_hhi_returns_concentration(conc_returns)
    assert pos_hhi > 0.5


def test_dd_and_tuw():
    values = np.array([1, 1.1, 1.2, 2, 1, 0.5, 0.3, 0.7, 0.8, 1.2, 2.1, 0.4, 0.1, 2.2])
    values = pd.Series(values, index=pd.date_range('2022-01-01', periods=len(values)))
    dd, tuw = compute_dd_and_tuw(values, dollars=True)
    pd.testing.assert_series_equal(dd, pd.Series([1.7, 2.0], index=pd.to_datetime(['2022-01-04', '2022-01-11'])))
    pd.testing.assert_series_equal(tuw, pd.Series([7.], index=pd.to_datetime(['2022-01-04'])))

    rets = values[1:]/values[:-1].values - 1
    rets = rets.iloc[1:]
    dd, tuw = compute_dd_and_tuw(rets, dollars=False)
    pd.testing.assert_series_equal(
        dd, pd.Series([1.75, 1.6071],index=pd.to_datetime(['2022-01-04', '2022-01-8'])), check_less_precise=3)
    pd.testing.assert_series_equal(tuw, pd.Series([4.], index=pd.to_datetime(['2022-01-04'])))


def test_sharpe_ratio():
    rets = norm.rvs(size=300, loc=0.08, scale=0.15, random_state=42)
    assert 0.5 < sharpe_ratio(rets) < 0.6


def test_probabilistic_sharpe_ratio():
    rets = norm.rvs(size=300, loc=0.0, scale=0.1, random_state=42)
    psr = probabilistic_sharpe_ratio(rets, benchmark=0)
    assert psr < 0.5

    rets = norm.rvs(size=300, loc=0.08, scale=0.1, random_state=42)
    psr = probabilistic_sharpe_ratio(rets, benchmark=0)
    assert psr > 0.99

    rets = norm.rvs(size=300, loc=-0.05, scale=0.1, random_state=42)
    psr = probabilistic_sharpe_ratio(rets, benchmark=0)
    assert psr < 0.01


def test_deflated_sharpe_ratio():
    rets = norm.rvs(size=(300, 10), loc=0.0, scale=0.1, random_state=42)
    psr, bsr = deflated_sharpe_ratio(rets, axis=0)
    assert len(psr) == 10
    assert bsr > 0

    rets = norm.rvs(size=(50, 10), loc=0.0, scale=0.1, random_state=42)
    _, bsr_higher_var = deflated_sharpe_ratio(rets, axis=0)
    assert bsr_higher_var > bsr

    rets = norm.rvs(size=(300, 30), loc=0.0, scale=0.1, random_state=42)
    _, bsr_more_trials = deflated_sharpe_ratio(rets, axis=0)
    assert bsr_more_trials > bsr


def test_annualized_sharpe_ratio():
    expected_asr = 1.0
    num_periods = 252
    period_sr = 1.0/np.sqrt(252)
    assert annualized_sharpe_ratio(period_sr, num_periods) == pytest.approx(expected_asr)


def test_information_ratio():
    market_rets = norm.rvs(size=300, loc=0.1, scale=0.1, random_state=42)
    rets = norm.rvs(size=300, loc=0.2, scale=0.2, random_state=1337)
    expected_excess = 0.2 - 0.1
    expected_tracking_error = np.sqrt(0.1**2 + 0.2**2)
    expected_ir = expected_excess/expected_tracking_error

    out = information_ratio(rets, market_rets)
    assert abs(expected_ir - out) < 0.05

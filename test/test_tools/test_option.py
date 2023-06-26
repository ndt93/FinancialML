import pytest
import pandas as pd
import numpy as np

from financial_ml.data_structures.constants import OptionCol, OptionType
from financial_ml.tools.option import (
    implied_risk_free_rate,
    implied_underlying_distribution,
    bsm_option_price,
    implied_volatility,
    binom_am_option_price,
    bsm_return_distribution
)


def _fmt_data(options):
    options = options.rename(columns={'strike': OptionCol.STRIKE})
    options[OptionCol.PRICE] = (options['bid'] + options['ask'])/2
    options['lastTradeDate'] = pd.to_datetime(options['lastTradeDate'], utc=True)
    options = options[
        (options['volume'] >= 100) &
        (options['lastTradeDate'] >= pd.to_datetime('2022-11-25T00:00:00Z')) &
        (options['ask'] > 0.01)
    ]
    options['mid'] = (options['bid'] + options['ask'])/2
    return options


@pytest.fixture
def data():
    dir = '../..'
    calls = _fmt_data(pd.read_csv(f'{dir}/data/SPY221202C.csv'))
    puts = _fmt_data(pd.read_csv(f'{dir}/data/SPY221202P.csv'))
    return calls, puts, 402.33, 7/365, 0.0


def test_implied_risk_free_rate(data):
    calls, puts, s0, t, d = data
    r, _ = implied_risk_free_rate(calls, puts, s0, t, d)
    assert r == pytest.approx(0.037444, abs=1e-3)


def test_implied_distribution_price_space(data):
    calls, puts, s0, t, d = data
    out = implied_underlying_distribution(
        calls, puts, t, r=None, s0=s0, d=d, n_interpolate=None,
        smooth_width=1e-3, volatility_space=False, random_state=42
    )
    print(out)
    rv, pdf = out['rv'], out['pdf']
    cdf_samples = rv.cdf([386, 392, 405, 406])
    pdf_samples = pdf(range(400, 410))
    assert out['r'] == pytest.approx(0.03744, abs=1e-3)
    np.testing.assert_array_almost_equal(cdf_samples, [0.00910268, 0.03869389, 0.59954005, 0.70131879])
    np.testing.assert_array_almost_equal(pdf_samples, [0.05012719, 0.02175576, 0.02381988, 0.05210559,
                                                       0.07070455, 0.08096232, 0.12420206, 0.12058565,
                                                       0.05888099, 0.01748028])


def test_implied_distribution_volatility_space(data):
    calls, puts, s0, t, d = data
    out = implied_underlying_distribution(
        calls, puts, t, r=None, s0=s0, d=d, n_interpolate=None,
        smooth_width=1e-3, volatility_space=True, random_state=42
    )
    rv, pdf = out['rv'], out['pdf']
    cdf_samples = rv.cdf([386, 392, 405, 406])
    pdf_samples = pdf(range(400, 410))
    assert out['r'] == pytest.approx(0.03744, abs=1e-3)
    np.testing.assert_array_almost_equal(cdf_samples, [0.010667, 0.037268, 0.635841, 0.737097])
    np.testing.assert_array_almost_equal(pdf_samples, [0.051657, 0.023868, 0.026152, 0.055514,
                                                       0.079277, 0.087429, 0.115507, 0.103345,
                                                       0.050047, 0.015746])


def test_bsm_option_pricing():
    price = bsm_option_price(
        OptionType.CALL, s0=1.3, k=1.347, r=0.04, sigma=0.15, T=3/12, div_yield=0.05
    )
    assert price == pytest.approx(0.0191968, abs=1e-4)

    price = bsm_option_price(
        OptionType.PUT, s0=1000, k=1000, r=0.05, sigma=0.15, T=10, div_yield=0.01
    )
    assert price == pytest.approx(38.4597, abs=1e-4)

    price = bsm_option_price(OptionType.CALL, 60, 60, 0.05, 0.22, 6, divs=[(1, 0.5 + i) for i in range(6)])
    assert price == pytest.approx(16.49211, abs=1e-4)

    price = bsm_option_price(OptionType.PUT, 50, 50, 0.1, 0.3, 0.25, divs=[(1.5, 2/12)])
    assert price == pytest.approx(3.030194, abs=1e-4)

    price = bsm_option_price(OptionType.CALL, 500, 550, 0.03, 0.2, 9/12, is_futures=True)
    assert price == pytest.approx(16.195803, abs=1e-4)

    price = bsm_option_price(OptionType.PUT, 500, 550, 0.03, 0.2, 9/12, is_futures=True)
    assert price == pytest.approx(65.083365, abs=1e-4)


def test_binom_option_pricing():
    price = binom_am_option_price(OptionType.PUT, 50, 50, 5/12, 0.1, 0.4, n=5)
    assert price == pytest.approx(4.48845, abs=1e-3)

    price = binom_am_option_price(OptionType.PUT, 50, 50, 5/12, 0.1, 0.4, n=100)
    assert price == pytest.approx(4.278, abs=1e-3)

    price = binom_am_option_price(OptionType.PUT, 300, 300, 4/12, 0.08, 0.3, is_futures=True)
    assert price == pytest.approx(20.22, abs=1e-2)

    price = binom_am_option_price(OptionType.PUT, 1.61, 1.6, 1, 0.08, 0.12, div_yield=0.09)
    assert price == pytest.approx(0.0738, abs=1e-3)

    price = binom_am_option_price(OptionType.PUT, 50, 50, 5/12, 0.1, 0.4, n=5, control_variates=True)
    assert price == pytest.approx(4.25, abs=1e-2)

    price = binom_am_option_price(OptionType.CALL, 50, 50, 5/12, 0.1, 0.4, n=500)
    bsm_eur_price = bsm_option_price(OptionType.CALL, 50, 50, 0.1, 0.4, 5/12)
    assert price == pytest.approx(bsm_eur_price, abs=1e-2)


def test_implied_volatility():
    iv = implied_volatility(observed_price=2.5, option_type=OptionType.CALL, s0=15, k=13, r=0.05, T=3 / 12)
    assert iv == pytest.approx(0.3964355, abs=1e-4)

    iv = implied_volatility(
        observed_price=4.278, pricing_fn=binom_am_option_price, option_type=OptionType.PUT, s0=50, k=50, r=0.1, T=5 / 12
    )
    assert iv == pytest.approx(0.4, abs=1e-2)


def test_bsm_return_distribution():
    dist = bsm_return_distribution(0.16, 0.2, 0.5)
    logret_dist = dist['log_ret']
    logprice_mean = np.log(40) + logret_dist.mean()
    logprice_var = logret_dist.var()
    assert logprice_mean == pytest.approx(3.759, abs=1e-3)
    assert logprice_var == pytest.approx(0.02, abs=1e-3)

    dist = bsm_return_distribution(0.17, 0.2, 3)
    ret_dist = dist['annual_ret']
    assert ret_dist.mean() == pytest.approx(0.15, abs=1e-3)
    assert ret_dist.std() == pytest.approx(0.1155, abs=1e-3)

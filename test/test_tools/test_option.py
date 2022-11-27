import pytest
import pandas as pd
import numpy as np

from data_structures.constants import OptionCol
from tools.option import (
    implied_risk_free_rate,
    implied_underlying_distribution,
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
    dir = '../'
    calls = _fmt_data(pd.read_csv(f'{dir}/data/SPY221202C.csv'))
    puts = _fmt_data(pd.read_csv(f'{dir}/data/SPY221202P.csv'))
    return calls, puts, 402.33, 7/365, 0.0


def test_implied_risk_free_rate(data):
    calls, puts, s0, t, d = data
    r, _ = implied_risk_free_rate(calls, puts, s0, t, d)
    assert r == pytest.approx(0.037444, abs=1e-3)


def test_implied_underlying_distribution(data):
    calls, puts, s0, t, d = data
    rv, pdf = implied_underlying_distribution(
        calls, puts, t, r=None, s0=s0, d=d, n_interpolate=None, smooth_width=1e-3, random_state=42
    )
    cdf_samples = rv.cdf([386, 392, 405, 406])
    pdf_samples = pdf(range(400, 410))
    np.testing.assert_array_almost_equal(cdf_samples, [0.00910268, 0.03869389, 0.59954005, 0.70131879])
    np.testing.assert_array_almost_equal(pdf_samples, [0.05012719, 0.02175576, 0.02381988, 0.05210559,
                                                       0.07070455, 0.08096232, 0.12420206, 0.12058565,
                                                       0.05888099, 0.01748028])

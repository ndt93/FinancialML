from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from financial_ml.data_structures.constants import TickCol, QuoteCol, BarCol
from financial_ml.data_structures.bars import aggregate_time_bars, aggregate_volume_bars
from financial_ml.features.microstructure import (
    tick_rule,
    roll_effective_spread,
    high_low_volatility,
    corwin_schultz_spread,
    kyle_lambda,
    amihud_lambda,
    hasbrouck_lambda,
    volume_synchronized_pin
)
from financial_ml.utils.simulation import gen_geometric_brownian


def test_tick_rule():
    ticks = pd.DataFrame({TickCol.PRICE: [1., 1.1, 1.2, 1., 1., 0.9, 0.8, 1.3]})
    aggressor = tick_rule(ticks, 1)
    np.testing.assert_array_almost_equal(aggressor, [1, 1, 1, -1, -1, -1, -1, 1])


def test_roll_effective_spread():
    np.random.seed(42)
    n_obs = 1000
    mid_price = np.array(list(gen_geometric_brownian(100, 0.0, 0.15, n_obs, random_state=42)))
    half_spread = 2.0
    observed_price = np.random.choice([1, -1], n_obs)*half_spread + mid_price
    ticks = pd.DataFrame({TickCol.PRICE: observed_price})
    spread, _ = roll_effective_spread(ticks)
    assert spread == pytest.approx(half_spread*2, abs=0.5)


@pytest.fixture
def random_walk_prices():
    n_obs = 1000
    prices = list(gen_geometric_brownian(100, 0.0, 0.15, n_obs, random_state=42))
    interval_sec = 0.001*365*24*60*60
    tick_times = [datetime(year=2021, month=1, day=1) + timedelta(seconds=interval_sec*i) for i in range(len(prices))]
    ticks = pd.DataFrame({TickCol.PRICE: prices, TickCol.TIMESTAMP: tick_times, TickCol.VOLUME: 1})
    bars = aggregate_time_bars(ticks, 'D')
    return ticks, bars


def test_hi_lo_volatility(random_walk_prices):
    ticks, bars = random_walk_prices
    est_vol = high_low_volatility(bars, 14).dropna()
    assert 0.075 <= np.mean(est_vol)*np.sqrt(365) <= 0.15


def test_corwin_schultz_spread(random_walk_prices):
    ticks, bars = random_walk_prices
    half_spread = 2.0
    np.random.seed(42)
    ticks[TickCol.PRICE] = np.random.choice([1, -1], ticks.shape[0])*half_spread + ticks[TickCol.PRICE]
    bars = aggregate_time_bars(ticks, 'D')
    spread, volatility = corwin_schultz_spread(bars, 30)
    dollar_spread = spread[QuoteCol.SPREAD]*bars[BarCol.CLOSE]
    assert 1.5 < dollar_spread.mean() <= 2.0
    assert 0.30 < volatility.mean()*np.sqrt(365) <= 0.33


def test_kyle_lambda(random_walk_prices):
    ticks, _ = random_walk_prices
    ld, ols = kyle_lambda(ticks, 1)
    assert ld == pytest.approx(0.3678, abs=1e-3)


def test_amihud_lambda(random_walk_prices):
    ticks, bars = random_walk_prices
    ld, ols = amihud_lambda(bars)
    assert ld == pytest.approx(0.0, abs=1e-3)


def test_hasbrouck_lambda(random_walk_prices):
    ticks, bars = random_walk_prices
    bar_ids = ticks.set_index(TickCol.TIMESTAMP)[TickCol.PRICE]\
        .groupby(pd.Grouper(freq='D')).transform(lambda x: x.index[0])
    ticks[TickCol.BAR_ID] = bar_ids.values
    ld, ols = hasbrouck_lambda(ticks, 1)
    assert ld == pytest.approx(0.0, abs=1e-3)


def test_vpin(random_walk_prices):
    ticks, _ = random_walk_prices
    bars, ticks = aggregate_volume_bars(ticks, 5, return_bar_id=True)
    vpin = volume_synchronized_pin(ticks, 5, n=10, b0=1).dropna()
    assert vpin.mean()[0] == pytest.approx(0.114062, abs=1e-3)

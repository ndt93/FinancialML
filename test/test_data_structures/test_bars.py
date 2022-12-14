import numpy as np
import pandas as pd
import pytest

import financial_ml.data_structures.bars as bars
from financial_ml.data_structures.constants import BarUnit, BarCol, TickCol


@pytest.fixture(scope='session')
def ticks_sample():
    ticks = pd.read_csv('../../data/XBTUSD_20220920.csv')
    ticks = ticks[ticks['symbol'] == 'XBTUSD'][['timestamp', 'price', 'foreignNotional']]
    ticks['timestamp'] = pd.to_datetime(ticks['timestamp'].str.slice(0, -3), format='%Y-%m-%dD%H:%M:%S.%f')
    ticks = ticks.rename(columns={
        'timestamp': TickCol.TIMESTAMP,
        'price': TickCol.PRICE,
        'foreignNotional': TickCol.VOLUME,
    })
    ticks = ticks.reset_index(drop=True)
    return ticks


@pytest.fixture(scope='session')
def bars_sample():
    bars_sample = pd.read_csv('../../data/SPY_1h_20221222.csv')
    bars_sample[BarCol.VOLUME] = np.arange(1, bars_sample.shape[0] + 1)
    bars_sample[BarCol.VWAP] = bars_sample[BarCol.CLOSE]
    bars_sample[BarCol.TIMESTAMP] = pd.to_datetime(bars_sample['Datetime'])
    bars_sample = bars_sample.drop(columns=['Adj Close'])
    bars_sample = bars_sample.set_index(BarCol.TIMESTAMP)
    return bars_sample


def test_time_bars(ticks_sample):
    time_bars = bars.aggregate_time_bars(ticks_sample, '15min')
    assert list(time_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(time_bars) == 96
    assert time_bars.index.freqstr == '15T'


def test_tick_bars(ticks_sample):
    tick_bars = bars.aggregate_tick_bars(ticks_sample, 1000)
    assert list(tick_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(tick_bars) == 86


def test_volume_bars(ticks_sample):
    volume_freq = 5*10**6
    volume_bars = bars.aggregate_volume_bars(ticks_sample, volume_freq)
    assert list(volume_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(volume_bars) == 86
    assert abs(volume_bars[BarCol.VOLUME].mean() - volume_freq) < 100000


def test_dollar_bars(ticks_sample):
    dollars_freq = 90*10**9
    dollars_bars = bars.aggregate_dollar_bars(ticks_sample, dollars_freq)
    assert list(dollars_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(dollars_bars) == 92
    print((dollars_bars[BarCol.VOLUME]*dollars_bars[BarCol.VWAP]).mean())
    assert abs((dollars_bars[BarCol.VOLUME]*dollars_bars[BarCol.VWAP]).mean() - dollars_freq) < 10**9


def test_tick_imbalance_bars(ticks_sample):
    min_bar_size = 500
    max_bar_size = 5000
    tick_imbalance_bars, ticks_ext = bars.aggregate_imblance_bars(
        ticks_sample,
        bar_unit=BarUnit.TICK,
        min_bar_size=min_bar_size,
        max_bar_size=max_bar_size,
        b0=-1,
        E_T_init=1000,
        abs_E_b_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
        debug=True
    )
    assert list(tick_imbalance_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(tick_imbalance_bars) == 157
    bar_sizes = ticks_ext[TickCol.BAR_ID].value_counts().sort_index().values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_volume_imbalance_bars(ticks_sample):
    min_bar_size = 1*10**6
    max_bar_size = 5*10**6
    volume_imbalance_bars = bars.aggregate_imblance_bars(
        ticks_sample,
        bar_unit=BarUnit.VOLUME,
        min_bar_size=min_bar_size,
        max_bar_size=max_bar_size,
        b0=-1,
        E_T_init=1000,
        abs_E_b_init=None,
        T_ewma_span=None,
        b_ewma_span=None
    )
    assert list(volume_imbalance_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(volume_imbalance_bars) == 395
    bar_sizes = volume_imbalance_bars[BarCol.VOLUME].values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_dollars_imbalance_bars(ticks_sample):
    min_bar_size = 1*10**6*20000
    max_bar_size = 5*10**6*20000
    dollars_imbalance_bars = bars.aggregate_imblance_bars(
        ticks_sample,
        bar_unit=BarUnit.DOLLARS,
        min_bar_size=min_bar_size,
        max_bar_size=max_bar_size,
        b0=-1,
        E_T_init=1000,
        abs_E_b_init=None,
        T_ewma_span=None,
        b_ewma_span=None
    )
    assert list(dollars_imbalance_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(dollars_imbalance_bars) == 379
    bar_sizes = dollars_imbalance_bars[BarCol.VOLUME]*dollars_imbalance_bars[BarCol.VWAP].values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_tick_runs_bars(ticks_sample):
    min_bar_size = 500
    max_bar_size = 5000
    tick_runs_bars, ticks_ext = bars.aggregate_runs_bars(
        ticks_sample,
        bar_unit=BarUnit.TICK,
        min_bar_size=min_bar_size,
        max_bar_size=max_bar_size,
        b0=-1,
        E_T_init=1000,
        P_b_buy_init=None,
        E_v_buy_init=None,
        E_v_sell_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
        debug=True
    )
    assert list(tick_runs_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(tick_runs_bars) == 148
    bar_sizes = ticks_ext[TickCol.BAR_ID].value_counts().sort_index().values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_volume_runs_bars(ticks_sample):
    min_bar_size = 1*10**6
    max_bar_size = 5*10**6
    volume_runs_bars = bars.aggregate_runs_bars(
        ticks_sample,
        bar_unit=BarUnit.VOLUME,
        min_bar_size=1*10**6,
        max_bar_size=5*10**6,
        b0=-1,
        E_T_init=1000,
        P_b_buy_init=None,
        E_v_buy_init=None,
        E_v_sell_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
    )
    assert list(volume_runs_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(volume_runs_bars) == 212
    bar_sizes = volume_runs_bars[BarCol.VOLUME].values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_dollars_runs_bars(ticks_sample):
    min_bar_size = 1*10**6*20000
    max_bar_size = 5*10**6*20000
    dollar_runs_bars = bars.aggregate_runs_bars(
        ticks_sample,
        bar_unit=BarUnit.DOLLARS,
        min_bar_size=1*10**6*20000,
        max_bar_size=5*10**6*20000,
        b0=-1,
        E_T_init=1000,
        P_b_buy_init=None,
        E_v_buy_init=None,
        E_v_sell_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
    )
    assert list(dollar_runs_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(dollar_runs_bars) == 218
    bar_sizes = dollar_runs_bars[BarCol.VOLUME]*dollar_runs_bars[BarCol.VWAP].values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_time_to_volume_bars_resample(bars_sample):
    resampled = bars.resample_time_to_volume_bars(bars_sample, freq=10)
    np.testing.assert_array_almost_equal(resampled[BarCol.VOLUME], [6, 9, 13])
    pd.testing.assert_index_equal(bars_sample.index[[0, 3, 5]], resampled.index)
    pd.testing.assert_series_equal(
        resampled[BarCol.OPEN], bars_sample[BarCol.OPEN].iloc[[0, 3, 5]], check_index=False, check_less_precise=2
    )
    pd.testing.assert_series_equal(
        resampled[BarCol.CLOSE], bars_sample[BarCol.CLOSE].iloc[[2, 4, 6]], check_index=False, check_less_precise=2
    )
    np.testing.assert_array_almost_equal(resampled[BarCol.LOW], [382.69, 385.63, 385.24], decimal=2)
    np.testing.assert_array_almost_equal(resampled[BarCol.HIGH], [387.3, 387.41, 386.86], decimal=2)
    np.testing.assert_array_almost_equal(resampled[BarCol.VWAP], [385.834, 386.137, 385.921], decimal=2)

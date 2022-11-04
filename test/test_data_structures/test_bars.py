import pandas as pd
import pytest

import data_structures.bars as bars
from data_structures.constants import BarUnit, BarCol, TickCol

ticks = pd.read_csv('../../data/trade_20220920.csv')
ticks = ticks[ticks['symbol'] == 'XBTUSD'][['timestamp', 'price', 'foreignNotional']]
ticks['timestamp'] = pd.to_datetime(ticks['timestamp'].str.slice(0, -3), format='%Y-%m-%dD%H:%M:%S.%f')
ticks = ticks.rename(columns={
    'timestamp': TickCol.TIMESTAMP,
    'price': TickCol.PRICE,
    'foreignNotional': TickCol.VOLUME,
})
ticks = ticks.reset_index(drop=True)


@pytest.fixture
def ticks_sample():
    return ticks.copy()


def test_time_bars(ticks_sample):
    time_bars = bars.aggregate_time_bars(ticks, '15min')
    print(time_bars)
    assert list(time_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(time_bars) == 96
    assert time_bars.index.freqstr == '15T'


def test_tick_bars(ticks_sample):
    tick_bars = bars.aggregate_tick_bars(ticks, 1000)
    assert list(tick_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(tick_bars) == 86


def test_volume_bars(ticks_sample):
    volume_freq = 5*10**6
    volume_bars = bars.aggregate_volume_bars(ticks, volume_freq)
    assert list(volume_bars.columns.values) == [
        BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP
    ]
    assert len(volume_bars) == 86
    assert abs(volume_bars[BarCol.VOLUME].mean() - volume_freq) < 100000


def test_dollar_bars(ticks_sample):
    dollars_freq = 90*10**9
    dollars_bars = bars.aggregate_dollar_bars(ticks, dollars_freq)
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
        ticks,
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
    bar_sizes = ticks_ext['bar_id'].value_counts().sort_index().values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_volume_imbalance_bars(ticks_sample):
    min_bar_size = 1*10**6
    max_bar_size = 5*10**6
    volume_imbalance_bars = bars.aggregate_imblance_bars(
        ticks,
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
        ticks,
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
        ticks,
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
    bar_sizes = ticks_ext['bar_id'].value_counts().sort_index().values
    assert max(bar_sizes) <= max_bar_size
    assert min(bar_sizes[:-1]) >= min_bar_size


def test_volume_runs_bars(ticks_sample):
    min_bar_size = 1*10**6
    max_bar_size = 5*10**6
    volume_runs_bars = bars.aggregate_runs_bars(
        ticks,
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
        ticks,
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

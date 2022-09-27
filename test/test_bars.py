from datetime import datetime

import pandas as pd

import data_structures.bars as bars
from data_structures.constants import BarUnit

ticks = pd.read_csv('../data/trade_20220920.csv')
ticks = ticks[ticks['symbol'] == 'XBTUSD'][['timestamp', 'price', 'foreignNotional']]
ticks['timestamp'] = ticks['timestamp'].map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
ticks = ticks.rename(columns={'foreignNotional': 'volume'})
ticks = ticks.reset_index(drop=True)

time_bars = bars.aggregate_time_bars(ticks, '15min')
print(time_bars)
tick_bars = bars.aggregate_tick_bars(ticks, 1000)
print(tick_bars)
volume_bars = bars.aggregate_volume_bars(ticks, 20000)
print(volume_bars)
dollar_bars = bars.aggregate_dollar_bars(ticks, 20000*20000)
print(dollar_bars)

tick_imbalance_bars = bars.aggregate_imblance_bars(
    ticks,
    bar_unit=BarUnit.TICK,
    min_bar_size=500,
    max_bar_size=5000,
    b0=-1,
    E_T_init=1000,
    # abs_E_b_init=0.1,
    T_ewma_window=None,
    b_ewma_window=None
)
print(tick_imbalance_bars)

volume_imbalance_bars = bars.aggregate_imblance_bars(
    ticks,
    bar_unit=BarUnit.VOLUME,
    min_bar_size=500000,
    max_bar_size=5000000,
    b0=-1,
    E_T_init=1000,
    # abs_E_b_init=10000,
    T_ewma_window=None,
    b_ewma_window=None
)
print(volume_imbalance_bars)

dollars_imbalance_bars = bars.aggregate_imblance_bars(
    ticks,
    bar_unit=BarUnit.DOLLARS,
    min_bar_size=20000*20000,
    max_bar_size=50000*50000,
    b0=-1,
    E_T_init=1000,
    # abs_E_b_init=10000,
    T_ewma_window=None,
    b_ewma_window=None
)
print(dollars_imbalance_bars)

tick_runs_bars = bars.aggregate_runs_bars(
    ticks,
    bar_unit=BarUnit.TICK,
    min_bar_size=500,
    max_bar_size=5000,
    b0=-1,
    E_T_init=1000,
    P_b_buy_init=None,
    E_v_buy_init=None,
    E_v_sell_init=None,
    T_ewma_window=None,
    b_ewma_window=None,
)
print(tick_runs_bars)

volume_runs_bars = bars.aggregate_runs_bars(
    ticks,
    bar_unit=BarUnit.VOLUME,
    min_bar_size=500000,
    max_bar_size=5000000,
    b0=-1,
    E_T_init=1000,
    P_b_buy_init=None,
    E_v_buy_init=None,
    E_v_sell_init=None,
    T_ewma_window=None,
    b_ewma_window=None,
)
print(volume_runs_bars)

dollar_runs_bars = bars.aggregate_runs_bars(
    ticks,
    bar_unit=BarUnit.DOLLARS,
    min_bar_size=20000*20000,
    max_bar_size=50000*50000,
    b0=-1,
    E_T_init=1000,
    P_b_buy_init=None,
    E_v_buy_init=None,
    E_v_sell_init=None,
    T_ewma_window=None,
    b_ewma_window=None,
)
print(dollar_runs_bars)

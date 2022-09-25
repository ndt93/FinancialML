from datetime import datetime

import pandas as pd

import data_structures.bars as bars
from data_structures.constants import BarUnit

ticks = pd.read_csv('../data/trade_20220920.csv')
ticks = ticks[ticks['symbol'] == 'XBTUSD'][['timestamp', 'price', 'foreignNotional']]
ticks['timestamp'] = ticks['timestamp'].map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
ticks = ticks.rename(columns={'foreignNotional': 'volume'})
ticks = ticks.reset_index(drop=True)

# time_bars = bars.aggregate_time_bars(ticks, '15min')
# print(time_bars)
# tick_bars = bars.aggregate_tick_bars(ticks, 1000)
# print(tick_bars)
# volume_bars = bars.aggregate_volume_bars(ticks, 20000)
# print(volume_bars)
# dollar_bars = bars.aggregate_dollar_bars(ticks, 20000*20000)
# print(dollar_bars)

tick_imbalance_bars = bars.aggregate_imblance_bars(
    ticks,
    bar_unit=BarUnit.TICK,
    min_bar_size=500,
    max_bar_size=5000,
    b0=-1,
    E_T_init=1000,
    # abs_E_b_init=0.1,
    b_ewma_window=1000*3
)
print(tick_imbalance_bars)

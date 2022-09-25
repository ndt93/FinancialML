from datetime import datetime

import pandas as pd

import data_structures.bars as bars


ticks = pd.read_csv('../data/trade_20220920.csv')
ticks = ticks[ticks['symbol'] == 'XBTUSD'][['timestamp', 'price', 'foreignNotional']]
ticks['timestamp'] = ticks['timestamp'].map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))
ticks = ticks.rename(columns={'foreignNotional': 'volume'})
ticks = ticks.reset_index(drop=True)

# time_bars = bars.aggregate_time_bars(ticks, '15min')
# print(time_bars)
tick_bars = bars.aggregate_tick_bars(ticks, 1000)
print(tick_bars)
# volume_bars = bars.aggregate_volume_bars(ticks, 20000)
# print(volume_bars)
# dollar_bars = bars.aggregate_dollar_bars(ticks, 20000*20000)
# print(dollar_bars)

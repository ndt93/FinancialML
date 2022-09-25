import numpy as np


def aggregate_tick_bars(ticks, frequency):
    times = ticks[:, 0]
    prices = ticks[:, 1]
    volumes = ticks[:, 2]
    bars = np.zeros(shape=(len(prices) // frequency, 6))
    bar_idx = 0
    for i in range(frequency, len(prices) + 1, frequency):
        bars[bar_idx][0] = times[i - 1]                        # Time
        bars[bar_idx][1] = prices[i - frequency]               # Open
        bars[bar_idx][2] = np.max(prices[(i - frequency):i])   # High
        bars[bar_idx][3] = np.min(prices[(i - frequency):i])   # Low
        bars[bar_idx][4] = prices[i - 1]                       # Close
        bars[bar_idx][5] = np.sum(volumes[(i - frequency):i])  # Volume
        bar_idx += 1
    return bars

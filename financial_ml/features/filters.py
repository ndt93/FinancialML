import numpy as np
import pandas as pd
from numba import jit


def filter_symmetric_cusum(series, threshold):
    """
    :param series: time series of a selected market feature with a DateTimeIndex
    :param threshold: threshold above which an event is sampled
    :return: a DateTimeIndex of event times
    """
    t_events = []
    pos_cumsum = 0
    neg_cumsum = 0
    diff = series.diff()
    for t in diff.index[1:]:
        cur_diff = diff.loc[t]
        pos_cumsum = max(0, pos_cumsum + cur_diff)
        neg_cumsum = min(0, neg_cumsum + cur_diff)
        if neg_cumsum < -threshold:
            neg_cumsum = 0
            t_events.append(t)
        elif pos_cumsum > threshold:
            pos_cumsum = 0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


@jit(nopython=True, nogil=True)
def get_mmi(series):
    """
    Calculate the market meanness index
    """
    median = np.median(series)
    diff = np.diff(series)
    n_hi = ((series[1:] > median) & (diff > 0)).sum()
    n_lo = ((series[1:] < median) & (diff < 0)).sum()
    return (n_hi + n_lo)/(series.shape[0] - 1)

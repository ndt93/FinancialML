"""
This module implements several filtering methods to obtain a subset of bars
at which there is a higher likelihood of an actionable event
"""
import pandas as pd


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

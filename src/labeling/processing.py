import pandas as pd

from data_structures.constants import EventCol


def drop_rare_labels(events: pd.DataFrame, min_pct=0.05, min_classes=2) -> pd.DataFrame:
    """
    Recursively drop labels with insufficient samples

    :param events: DataFrame with EventCol.LABEL column
    :param min_pct: minimum % of label value in the samples
    :param min_classes: minimum number of label classes
    :return: events DataFrame with rare labels removed
    """
    while True:
        df = events[EventCol.LABEL].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] <= min_classes:
            break
        events = events[events[EventCol.LABEL] != df.index[df.argmin()]]
    return events


def count_events_per_bar(bar_times: pd.DatetimeIndex, event_end_times: pd.Series) -> pd.Series:
    """
    Count number of concurrent events in each bar

    :param bar_times: Series of times of bars
    :param event_end_times: Series of event end times, indexed by event start times
    :return: a Series of concurrent events count index by bar times from the earliest to latest event times
    """
    event_end_times = event_end_times.fillna(bar_times[-1])
    event_times_iloc = bar_times.searchsorted([event_end_times.index[0], event_end_times.max()])
    res = pd.Series(0, index=bar_times[event_times_iloc[0]:event_times_iloc[1] + 1])
    for event_start_time, event_end_time in event_end_times.iteritems():
        res[event_start_time:event_end_time] += 1
    return res


def compute_label_avg_uniqueness(bars, events):
    """
    At each bar, a label's uniqueness is 1/#concurrent_events. This method find
    the average uniqueness of each label over its event's duration.

    :param bars: a Series of bars with DateTimeIndex
    :param events: a DataFrame of [EventCol.END_TIMES] and DateTimeIndex
    :return: a Series of average uniqueness for each event, indexed by the event start times
    """
    event_end_times = events[EventCol.END_TIME]
    events_counts = count_events_per_bar(bars.index, event_end_times)
    events_counts = events_counts.loc[~events_counts.index.duplicated(keep='last')]
    events_counts = events_counts.reindex(bars.index).fillna(0)

    res = pd.Series(index=event_end_times.index)
    for event_start_time, event_end_time in event_end_times.iteritems():
        res.loc[event_start_time] = (1./events_counts.loc[event_start_time:event_end_times]).mean()
    return res

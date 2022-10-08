import pandas as pd
import numpy as np

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


# --- Sequential Bootstrap ---

def get_event_indicators(bar_times: pd.DatetimeIndex, event_end_times: pd.Series) -> pd.DataFrame:
    """
    :param bar_times: Series of times of bars
    :param event_end_times: Series of event end times, indexed by event start times
    :return: DataFrame with 1 column per event, indexed by bar_times. Set to 1 if the event span the bar
    """
    res = pd.DataFrame(0, index=bar_times, columns=range(event_end_times.shape[0]))
    for i, (event_start, event_end) in enumerate(event_end_times.iteritems()):
        res.loc[event_start:event_end, i] = 1
    return res


def _get_avg_uniqueness(event_indicators: pd.DataFrame) -> pd.Series:
    """
    :param event_indicators: see output of _get_event_indicators
    :return: Series of average uniqueness for each event
    """
    concurrency = event_indicators.sum(axis=1)
    uniqueness = event_indicators.div(concurrency, axis=0)
    avg_uniqueness = uniqueness[uniqueness > 0].mean()
    return avg_uniqueness


def sample_sequential_boostrap(event_indicators: pd.DataFrame, size=None) -> list:
    """
    :param event_indicators: see output of _get_event_indicators
    :param size: number of samples to be drawn. If None, default to total number of events
    :return: a list of integer index of sampled events
    """
    if size is None:
        size = event_indicators.shape[1]
    samples = []
    while len(samples) < size:
        trial_avg_uniq = pd.Series()
        for event_id in event_indicators:
            trial_event_indicators = event_indicators[samples + [event_id]]
            trial_avg_uniq.loc[event_id] = _get_avg_uniqueness(trial_event_indicators).iloc[-1]
        probs = trial_avg_uniq / trial_avg_uniq.sum()
        samples += [np.random.choice(event_indicators.columns, p=probs)]
    return samples

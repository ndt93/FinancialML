import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


def _get_purged_train_times(event_times: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Remove any event time interval that overlaps with test time intervals to form a training set

    :param event_times: a Series of event end times indexed by event start times
    :param test_times: a Series of end times for each test period indexed by start times
    :return: a Series of event_times that doesn't overlap with test_times
    """
    train_times = event_times.copy(deep=True)
    for test_start, test_end in test_times.iteritems():
        overlap1 = train_times[(test_start <= train_times.index) & (train_times.index <= test_end)].index
        overlap2 = train_times[(test_start <= train_times) & (train_times <= test_end)].index
        overlap3 = train_times[(train_times.index <= test_start) & (train_times >= test_end)].index
        train_times = train_times.drop(overlap1.union(overlap2).union(overlap3))
    return train_times


def _get_embargo_times(bar_times: np.ndarray | list, embargo_pct: float):
    """
    This function should be used before applying purging to remove leakage from test set to train set
    due to serial correlations.
    Each event time interval will be extended by a small interval h and make any train event interval immediately
    follow a test event to be purged.

    :param bar_times: a Series of bar timestamps
    :param embargo_pct: each timestamp will be shifted forward by (embargo_pc * total_bars) number of bars
    :return: a Series with mapping from original timestamps (in index) to shifted embargo timestamps (in values)
    """
    step = 0 if bar_times is None else int(bar_times.shape[0] * embargo_pct)
    if step == 0:
        res = pd.Series(bar_times, index=bar_times)
    else:
        res = pd.Series(bar_times[step:], index=bar_times[:-step])
        res = res.append(pd.Series(bar_times[-1], index=bar_times[-step:]))
    return res


def apply_purging_and_embargo(
        event_times: pd.Series, test_times: pd.Series, bar_times=None, embargo_pct=0.,
) -> pd.Series:
    """
    Apply purging and embargo on train set labels that span intervals in a time series.
    - Purge observations in train set whose labels overlap with test set labels
    - Embargo: Drop a portion of observations at the start of train set that follows a test set to prevent
        leakage from serial correlations

    :param event_times: a Series of event end times indexed by event start times
    :param test_times: a Series of end times for each test period indexed by start times
    :param bar_times: a Series of bar timestamps. If none, no embargo is applied
    :param embargo_pct: each timestamp will be shifted forward by (embargo_pc * total_bars) number of bars
    :return: a Series of purged and embargo event_times for the train set
    """
    embargo_times = _get_embargo_times(bar_times, embargo_pct)
    adj_test_times = pd.Series(embargo_times[test_times].values, index=test_times.index)
    train_times = _get_purged_train_times(event_times, adj_test_times)
    return train_times


class PurgedKFold(_BaseKFold):
    """
    Perform KFold splitting on time series for cross validation, supporting labels that span across intervals.
    Purge observations in train set with labels that overlap with test set's labels
    Assume test set is contiguous (no training samples in between) i.e. shuffle = False
    """

    def __init__(self, n_splits=3, event_times: pd.Series=None, embargo_pct=0.):
        """
        :param n_splits:
        :param event_times: a Series of event end times indexed by event start time.
            An event defines the interval that each label spans
        :param embargo_pct: Used to prevent leakage from serial correlation. see apply_purging_and_embargo function
        """
        if not isinstance(event_times, pd.Series):
            raise ValueError('event_times must be a pandas Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.event_times = event_times
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        if (X.index == self.event_times.index).sum() != len(self.event_times):
            raise ValueError('X and event_times must have the same index')

        num_obs = X.shape[0]
        indices = np.arange(num_obs)
        embargo = int(num_obs * self.embargo_pct)
        test_splits = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_splits:
            test_indices = indices[i:j]
            test_start_time = self.event_times.index[i]
            test_end_time = self.event_times.iloc[test_indices].max()
            test_end_time_idx = self.event_times.index.searchsorted(test_end_time)

            train_start_times = self.event_times[self.event_times <= test_start_time].index
            train_indices = self.event_times.index.searchsorted(train_start_times)

            if test_end_time_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[test_end_time_idx + embargo:]))

            yield train_indices, test_indices

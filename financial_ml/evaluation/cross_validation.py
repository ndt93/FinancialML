import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from financial_ml.data_structures.constants import EventCol
from financial_ml.evaluation.metrics import Metrics


def _get_purged_train_indices(event_times: pd.Series, test_times: pd.Series) -> np.ndarray:
    """
    Remove any event time interval that overlaps with test time intervals to form a training set.
    Events are assumed to be uniquely identified by their start times.

    :param event_times: a Series of event end times indexed by event start times
    :param test_times: a Series of end times for each test period indexed by start times
    :return: train set indices array
    """
    train_times = event_times.rename_axis(index=EventCol.START_TIME).reset_index(name=EventCol.END_TIME)
    train_starts = train_times[EventCol.START_TIME]
    train_ends = train_times[EventCol.END_TIME]
    for test_start, test_end in test_times.iteritems():
        overlap1 = train_times[(test_start <= train_starts) & (train_starts <= test_end)].index
        overlap2 = train_times[(test_start <= train_ends) & (train_ends <= test_end)].index
        overlap3 = train_times[(train_starts <= test_start) & (train_ends >= test_end)].index
        train_times = train_times.drop(overlap1.union(overlap2).union(overlap3))
    return train_times.index.values


def _get_embargo_times(bar_times: np.ndarray | list, embargo_pct: float):
    """
    This function should be used before applying purging to remove leakage from test set to train set
    due to serial correlations.
    Each event time interval will be extended by a small interval h and make any train event interval immediately
    follow a test event to be purged.

    :param bar_times: an array of bar timestamps
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
) -> np.ndarray:
    """
    Apply purging and embargo on train set labels that span intervals in a time series.
    - Purge observations in train set whose labels overlap with test set labels
    - Embargo: Drop a portion of observations at the start of train set that follows a test set to prevent
        leakage from serial correlations
    Events are assumed to be uniquely identified by their start times.

    :param event_times: a Series of event end times indexed by event start times
    :param test_times: a Series of end times for each test period indexed by start times
    :param bar_times: an array of bar timestamps. If none, no embargo is applied
    :param embargo_pct: each timestamp will be shifted forward by (embargo_pc * total_bars) number of bars
    :return: train set indices array
    """
    if bar_times is not None:
        embargo_times = _get_embargo_times(bar_times, embargo_pct)
        adj_test_times = pd.Series(embargo_times[test_times].values, index=test_times.index)
    else:
        adj_test_times = test_times
    train_indices = _get_purged_train_indices(event_times, adj_test_times)
    return train_indices


class PurgedKFold:
    """
    KFold splitting on time series for cross validation, supporting labels that span across intervals.
    Purge observations in train set with labels that overlap with test set's labels.
    Apply a gap between train set that follows a test set based on an embargo parameter to prevent leakage
    from serial correlations.
    """

    def __init__(self, n_splits=3, embargo_pct=0.):
        """
        :param n_splits: number of k-fold splits
        :param embargo_pct: % of observations to discard from a train set that follows a test set.
            % is over the entire dataset and not just the train set.
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, event_times: pd.Series):
        """
        :param event_times: a Series of event end times indexed by event start time.
            An event defines the interval that each label spans.
        :return: (train indices, test indices)
        """
        if not isinstance(event_times, pd.Series):
            raise ValueError('event_times must be a pandas Series')
        if not isinstance(event_times.index, pd.DatetimeIndex):
            raise ValueError('event_times must have a DateTimeIndex')

        num_obs = event_times.shape[0]
        indices = np.arange(num_obs)
        embargo = int(num_obs * self.embargo_pct)
        test_splits = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_splits:
            test_indices = indices[i:j]
            test_start_time = event_times.index[i]
            train_1_indices = indices[event_times < test_start_time]
            train_indices = train_1_indices

            test_end_time = event_times.iloc[test_indices].max()
            train_2_start_idx = event_times.index.searchsorted(test_end_time, side='right') + 1
            if train_2_start_idx < num_obs:
                train_indices = np.concatenate((train_1_indices, indices[train_2_start_idx + embargo:]))

            yield train_indices, test_indices


def timeseries_cv_score(
        clf, X, y, sample_weight, scoring=Metrics.NEG_LOG_LOSS,
        event_times=None, cv=None, cv_splitter=None, embargo_pct=None
):
    """
    Perform cross validation scoring on time series data. See also: sklearn cross_val_score

    :param clf: the classifier model
    :param X: input features DataFrame
    :param y: label Series
    :param sample_weight: array of sample weights
    :param scoring: scoring function name
    :param event_times: used for PurgedKFold. Can be None if cv_splitter is provided
    :param cv: number of cv folds
    :param cv_splitter: default to using PurgedKFold if None
    :param embargo_pct: embargo parameter for PurgedKFold
    :return: an array of score for each CV split. Can be None if cv_splitter is provided
    """

    if scoring not in ['neg_log_loss', 'accuracy']:
        raise NotImplementedError(f'scoring: {scoring}')
    if cv_splitter is None:
        cv_splitter = PurgedKFold(n_splits=cv, embargo_pct=embargo_pct)

    scores = []
    for train_indices, test_indices in cv_splitter.split(event_times):
        X_train = X.iloc[train_indices, :]
        y_train = y.iloc[train_indices]
        train_sample_weight = sample_weight.iloc[train_indices].values
        X_test = X.iloc[test_indices, :]
        y_test = y.iloc[test_indices]
        test_sample_weight = sample_weight.iloc[test_indices].values

        fitted = clf.fit(X=X_train, y=y_train, sample_weight=train_sample_weight)

        if scoring == Metrics.NEG_LOG_LOSS:
            prob = fitted.predict_proba(X_test)
            split_score = -log_loss(y_test, prob, sample_weight=test_sample_weight, labels=clf.classes_)
        else:
            pred = fitted.predict(X_test)
            split_score = accuracy_score(y_test, pred, sample_weight=test_sample_weight)

        scores.append(split_score)

    return np.array(scores)

from datetime import datetime

import numpy as np
import pytest
import pandas as pd

from evaluation.purged_cv import apply_purging_and_embargo, PurgedKFold

days = pd.DatetimeIndex([datetime(year=2022, month=1, day=i) for i in range(1, 32)])


def indices_to_intervals(indices):
    starts = [days[i[0]] for i in indices]
    ends = [days[i[1]] for i in indices]
    return pd.Series(ends, index=starts)


@pytest.fixture
def event_times():
    event_indices = [(0, 6), (3, 6), (5, 8), (7, 10), (11, 15), (16, 24), (17, 23), (16, 24), (23, 27), (22, 29)]
    return indices_to_intervals(event_indices)


def test_apply_purging(event_times):
    test_indices = [(8, 10), (18, 22)]
    test_times = indices_to_intervals(test_indices)
    train_times = apply_purging_and_embargo(event_times, test_times)
    expected_train_indices = [0, 1, 4, 8]
    expected_train_times = event_times.iloc[expected_train_indices]
    pd.testing.assert_series_equal(train_times, expected_train_times)


def test_apply_embargo(event_times):
    test_indices = [(8, 10), (18, 22)]
    test_times = indices_to_intervals(test_indices)
    train_times = apply_purging_and_embargo(event_times, test_times, bar_times=days.values, embargo_pct=0.1)
    expected_train_indices = [0, 1]
    expected_train_times = event_times.iloc[expected_train_indices]
    pd.testing.assert_series_equal(train_times, expected_train_times)


def test_purged_kfold(event_times):
    n_splits = 3
    kfold = PurgedKFold(n_splits=n_splits, event_times=event_times, embargo_pct=0)

    X = pd.DataFrame(range(10), index=event_times.index)
    splits = list(kfold.split(X))
    assert len(splits) == n_splits

    for train_indices, test_indices in splits:
        assert len(train_indices) > 0
        assert len(test_indices) > 0

        train_events = event_times.iloc[train_indices]
        test_events = event_times.iloc[test_indices]
        test_from = test_events.index.min()
        test_to = test_events.max()

        for s, e in train_events.iteritems():
            assert e < test_from or s > test_to


def test_purged_kfold_embargo(event_times):
    n_splits = 3
    kfold = PurgedKFold(n_splits=n_splits, event_times=event_times, embargo_pct=0.1)

    X = pd.DataFrame(range(10), index=event_times.index)
    splits = list(kfold.split(X))
    assert len(splits) == n_splits

    train_split_1 = splits[0][0]
    test_split_1 = splits[0][1]
    np.testing.assert_equal(train_split_1, [5, 6, 7, 8, 9])
    np.testing.assert_array_equal(test_split_1, [0, 1, 2, 3])

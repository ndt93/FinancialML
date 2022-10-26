from itertools import combinations, chain

import pytest
import pandas as pd

from evaluation.backtest import CombinatorialPurgedCV


@pytest.fixture
def bar_times():
    return pd.date_range(start='2022-01-01', periods=7*30)


@pytest.fixture
def event_times():
    event_starts = pd.date_range(start='2022-01-01', periods=30, freq='7D')
    event_ends = event_starts + pd.DateOffset(6)
    return pd.Series(event_ends, index=event_starts)


def test_combinatorial_purged_cv_splits(bar_times, event_times):
    grp_size = 5
    expected_test_grps = [list(c) for c in combinations(range(6), 2)]

    comb_cv = CombinatorialPurgedCV(6, 2, bar_times, event_times, embargo_pct=0.)
    actual_test_grps = []
    for train_indices, test_indices in comb_cv.split():
        assert len(train_indices) + len(test_indices) == event_times.shape[0]
        assert len(set(train_indices) & set(test_indices)) == 0

        split_test_grps = sorted(list(set(i // grp_size for i in test_indices)))
        actual_test_grps.append(split_test_grps)

    assert expected_test_grps == actual_test_grps


def test_combinatorial_purged_cv_paths(bar_times, event_times):
    expected_path_splits = [
        [0, 0, 1, 2, 3, 4],
        [1, 5, 5, 6, 7, 8],
        [2, 6, 9, 9, 10, 11],
        [3, 7, 10, 12, 12, 13],
        [4, 8, 11, 13, 14, 14]
    ]

    comb_cv = CombinatorialPurgedCV(6, 2, bar_times, event_times, embargo_pct=0.)
    list(comb_cv.split())
    for p, backtest_path in enumerate(comb_cv.get_backtest_paths()):
        indices = list(chain.from_iterable(i[0] for i in backtest_path))
        splits = [i[1] for i in backtest_path]
        assert indices == list(range(event_times.shape[0]))
        assert splits == expected_path_splits[p]

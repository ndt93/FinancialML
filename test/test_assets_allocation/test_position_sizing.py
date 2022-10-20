import pytest
import pandas as pd

from assets_allocation.position_sizing import (
    compute_gaussian_mixture_position_size
)


def indices_to_intervals(bar_times, indices):
    starts = [bar_times[i[0]] for i in indices]
    ends = [bar_times[i[1]] for i in indices]
    return pd.Series(ends, index=starts)


@pytest.fixture
def bar_times():
    return pd.date_range(start='2022-01-01', end='2022-01-31')


@pytest.fixture
def event_times(bar_times):
    event_indices = [(0, 6), (3, 6), (5, 8), (7, 10), (11, 15), (16, 24), (17, 23), (16, 24), (23, 27), (22, 29)]
    return indices_to_intervals(bar_times, event_indices)


def test_gaussian_mixture_position_size(bar_times, event_times):
    sides = pd.Series([-1, -1,  1,  1,  1, -1,  1,  1, -1,  1], index=event_times.index)
    out = compute_gaussian_mixture_position_size(bar_times, event_times, sides, random_state=0)
    print(out)

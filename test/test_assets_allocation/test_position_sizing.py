import numpy as np
import pytest
import pandas as pd
from scipy.stats import norm

from assets_allocation.position_sizing import (
    compute_gaussian_mixture_position_size,
    get_sigmoid_coeff,
    compute_position_size_from_divergence,
    compute_limit_price,
    compute_budgeted_position_size,
    compute_position_size_from_probabilities
)
from data_structures.constants import EventCol
from labeling.weighting import count_events_per_bar, get_event_indicators


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


@pytest.fixture
def sides(event_times):
    return pd.Series([-1, -1,  1,  1,  1, -1,  1,  1, -1,  1], index=event_times.index)


def test_gaussian_mixture_position_size(bar_times, event_times, sides):
    events_l = count_events_per_bar(bar_times, event_times[sides == 1])
    events_s = count_events_per_bar(bar_times, event_times[sides == -1])
    net_events = ((pd.Series(0, index=bar_times) + events_l).fillna(0) - events_s).fillna(0)

    out = compute_gaussian_mixture_position_size(bar_times, event_times, sides, random_state=0)
    assert ((out >= -1) & (out <= 1)).sum() == out.shape[0]
    assert np.corrcoef(net_events.values, out.values)[0, 0] > 0.99


def test_budgeted_position_size(bar_times, event_times, sides):
    events_l = count_events_per_bar(bar_times, event_times[sides == 1])
    events_s = count_events_per_bar(bar_times, event_times[sides == -1])
    joined = pd.concat(
        {'bars': bar_times.to_series(), 'long': events_l, 'short': events_s}, axis=1
    ).fillna(0).drop(columns=['bars'])
    expected = joined['long']/joined['long'].max() - joined['short']/joined['short'].max()
    out = compute_budgeted_position_size(bar_times, event_times, sides)
    pd.testing.assert_series_equal(expected, out, check_less_precise=3)


def test_position_size_by_probabilities_no_side(bar_times, event_times, sides):
    events = event_times.to_frame(EventCol.END_TIME)
    prob = pd.Series([0.102, 0.789, 0.691, 0.872, 0.376, 0.569, 0.646, 0.209, 0.106, 0.15], index=events.index)
    pred = sides

    z = (prob - 0.5)/np.sqrt(prob*(1 - prob))
    pos_sizes = pred*(2*norm.cdf(z) - 1)
    event_indicators = get_event_indicators(bar_times, event_times)
    expected_sizes = pd.DataFrame(dtype=float, index=event_indicators.index)
    for event_id, indicators in event_indicators.iteritems():
        expected_sizes.loc[:, event_id] = indicators.where(indicators == 0, pos_sizes.iloc[event_id])
    expected_sizes = expected_sizes.sum(axis=1)/(expected_sizes != 0).sum(axis=1)
    expected_sizes = np.round(expected_sizes/0.1)*0.1
    expected_sizes = expected_sizes.loc[events.index.union(events[EventCol.END_TIME].values).drop_duplicates()]

    out = compute_position_size_from_probabilities(
        events, 0.1, prob, pred, 2
    )
    pd.testing.assert_series_equal(expected_sizes, out, check_less_precise=1, check_names=False)


def test_position_size_by_probabilities_with_side(bar_times, event_times, sides):
    events = event_times.to_frame(EventCol.END_TIME)
    events[EventCol.SIDE] = sides
    prob = pd.Series([0.102, 0.789, 0.691, 0.872, 0.376, 0.569, 0.646, 0.209, 0.106, 0.15], index=events.index)
    pred = pd.Series([-1,  1, -1,  1,  1,  1, -1, -1,  1, -1], index=events.index)

    z = (prob - 0.5)/np.sqrt(prob*(1 - prob))
    pos_sizes = pred*(2*norm.cdf(z) - 1)
    event_indicators = get_event_indicators(bar_times, event_times)
    expected_sizes = pd.DataFrame(dtype=float, index=event_indicators.index)
    for event_id, indicators in event_indicators.iteritems():
        expected_sizes.loc[:, event_id] = indicators.where(
            indicators == 0, pos_sizes.iloc[event_id]*sides.iloc[event_id]
        )
    expected_sizes = expected_sizes.sum(axis=1)/(expected_sizes != 0).sum(axis=1)
    expected_sizes = np.round(expected_sizes/0.1)*0.1
    expected_sizes = expected_sizes.loc[events.index.union(events[EventCol.END_TIME].values).drop_duplicates()]

    out = compute_position_size_from_probabilities(
        events, 0.1, prob, pred, 2
    )
    pd.testing.assert_series_equal(expected_sizes, out, check_less_precise=1, check_names=False)


def test_divergence_position_size():
    w = get_sigmoid_coeff(10, 0.95)
    pos_size = compute_position_size_from_divergence(110, 100, 100, w)
    assert pos_size == 95
    pos_size = compute_position_size_from_divergence(115, 100, 100, w)
    assert pos_size == 97


def test_divergence_position_limit_price():
    w = get_sigmoid_coeff(10, 0.95)
    pos_size = compute_position_size_from_divergence(115, 100, 100, w)
    limit_price = compute_limit_price(0, pos_size, 100, 115, w)
    assert limit_price == pytest.approx(112.3657)

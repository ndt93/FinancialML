from datetime import datetime, timedelta

import pandas as pd
import pytest
import numpy as np

from data_structures.constants import EventCol, BarCol
from labeling.weighting import (
    drop_rare_labels,
    compute_label_avg_uniqueness,
    get_event_indicators,
    sample_sequential_boostrap,
    count_events_per_bar,
    compute_weights_by_returns,
    apply_time_decay_to_weights
)


def test_drop_rare_labels():
    events = pd.DataFrame({EventCol.LABEL: [1]*5 + [2]*150 + [3]*8})
    filtered_events = drop_rare_labels(events, min_pct=0.05, min_classes=1)
    label_counts = filtered_events[EventCol.LABEL].value_counts().sort_index()
    pd.testing.assert_series_equal(label_counts, pd.Series([150, 8], index=[2, 3]), check_names=False)


@pytest.fixture
def data_sample():
    first_date = datetime(year=2022, month=1, day=1)
    event_starts = [first_date + timedelta(days=d) for d in [0, 2, 4]]
    event_ends = [first_date + timedelta(days=d) for d in [2, 3, 5]]
    events = pd.DataFrame({EventCol.END_TIME: event_ends}, index=event_starts)
    bars = pd.DataFrame(
        {BarCol.CLOSE: 1.1**np.arange(0, 6)},
        index=[first_date + timedelta(days=d) for d in range(6)]
    )
    return {
        'events': events,
        'bars': bars
    }


def test_compute_label_avg_uniqueness(data_sample):
    events = data_sample['events']
    bars = data_sample['bars']
    avg_uniqueness = compute_label_avg_uniqueness(bars, events)
    expected = pd.Series([0.833, 0.75, 1.0], index=events.index)
    pd.testing.assert_series_equal(avg_uniqueness, expected, check_less_precise=2)


def test_sequential_bootstrap(data_sample):
    np.random.seed(42)
    events = data_sample['events']
    bars = data_sample['bars']

    events_indicators = get_event_indicators(bars.index, events[EventCol.END_TIME])
    sample = sample_sequential_boostrap(events_indicators, 100)
    freqs = pd.Series(sample).value_counts(normalize=True).sort_index().values
    np.testing.assert_array_almost_equal(freqs, [0.34, 0.3, 0.36])


def test_compute_weights_by_returns(data_sample):
    events = data_sample['events']
    bars = data_sample['bars']

    event_times = events[EventCol.END_TIME]
    events_counts = count_events_per_bar(bars.index, event_times)
    weights_by_ret = compute_weights_by_returns(event_times, events_counts, bars[BarCol.CLOSE])
    np.testing.assert_array_almost_equal([0.9, 0.9, 1.2], weights_by_ret.values)


def test_apply_time_decay_to_weights(data_sample):
    events = data_sample['events']
    bars = data_sample['bars']
    avg_uniqueness = compute_label_avg_uniqueness(bars, events)
    print()

    time_decay_weights = apply_time_decay_to_weights(avg_uniqueness, oldest_weight=1.)
    np.testing.assert_array_almost_equal([1.0]*3, time_decay_weights.values)

    time_decay_weights = apply_time_decay_to_weights(avg_uniqueness, oldest_weight=0.5)
    np.testing.assert_array_almost_equal([0.6613, 0.8065, 1.0], time_decay_weights.values, decimal=3)

    time_decay_weights = apply_time_decay_to_weights(avg_uniqueness, oldest_weight=0.0)
    np.testing.assert_array_almost_equal([0.3226, 0.6129, 1.0], time_decay_weights.values, decimal=3)

    time_decay_weights = apply_time_decay_to_weights(avg_uniqueness, oldest_weight=-0.5)
    np.testing.assert_array_almost_equal([0.0, 0.2258, 1.0], time_decay_weights.values, decimal=3)

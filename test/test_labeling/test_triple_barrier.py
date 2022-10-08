import pandas as pd
import pytest

from data_structures.constants import BarCol, EventCol
from labeling.weighting import drop_rare_labels
from labeling.triple_barrier import get_event_end_times, get_event_labels

prices = pd.DataFrame({
    BarCol.TIMESTAMP: pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 21)]),
    BarCol.CLOSE: [
        100., 100.47, 99.99, 99.04, 99.02, 99.12, 99.4, 101.17, 100.51, 100.92,
        100.4, 100.54, 99.55, 98.93, 100.02, 99.39, 98.72, 98.31, 98.35, 97.45
    ]
}).set_index(BarCol.TIMESTAMP)

event_start_times = pd.to_datetime(['2022-01-05', '2022-01-09', '2022-01-15'])
events = pd.DataFrame({
    EventCol.EXPIRY: pd.to_datetime(['2022-01-09', '2022-01-12', '2022-01-20'])
}, index=event_start_times)

targets = pd.Series([0.0098, 0.05, 0.01], index=event_start_times)


def test_get_event_labels_no_side():
    end_times = get_event_end_times(
        prices[BarCol.CLOSE],
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
        0.0,
        sides=None
    )
    pd.testing.assert_index_equal(end_times.index, event_start_times)
    pd.testing.assert_series_equal(end_times[EventCol.EXPIRY], events[EventCol.EXPIRY])
    pd.testing.assert_series_equal(end_times[EventCol.TARGET], targets, check_names=False)
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-08', '2022-01-12', '2022-01-17']), index=event_start_times)
    pd.testing.assert_series_equal(end_times[EventCol.END_TIME], expected_end_times, check_names=False)

    labels = get_event_labels(end_times, prices)
    expected_returns = pd.Series([101.17/99.02 - 1, 100.54/100.51 - 1, 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 1.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_labels_known_side():
    sides = pd.Series([1, -1, 1], index=event_start_times)
    end_times = get_event_end_times(
        prices[BarCol.CLOSE],
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
        0.0,
        sides=sides
    )
    pd.testing.assert_index_equal(end_times.index, event_start_times)
    pd.testing.assert_series_equal(end_times[EventCol.EXPIRY], events[EventCol.EXPIRY])
    pd.testing.assert_series_equal(end_times[EventCol.TARGET], targets, check_names=False)
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-08', '2022-01-12', '2022-01-17']), index=event_start_times)
    pd.testing.assert_series_equal(end_times[EventCol.END_TIME], expected_end_times, check_names=False)

    labels = get_event_labels(end_times, prices)
    expected_returns = pd.Series([101.17/99.02 - 1, -1*(100.54/100.51 - 1), 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 0.0, 0.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_labels_cancel_expired_event():
    end_times = get_event_end_times(
        prices[BarCol.CLOSE],
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
        0.0,
        sides=None
    )
    labels = get_event_labels(end_times, prices, cancel_expired_event=True)
    expected_labels = pd.Series([1.0, 0.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_end_times_no_pt_sl():
    end_times = get_event_end_times(
        prices[BarCol.CLOSE],
        event_start_times,
        targets,
        (0.0, 0.0),
        events[EventCol.EXPIRY],
        0.0,
        sides=None
    )
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-09', '2022-01-12', '2022-01-20']), index=event_start_times)
    pd.testing.assert_series_equal(end_times[EventCol.END_TIME], expected_end_times, check_names=False)


def test_get_event_labels_multipliers():
    end_times = get_event_end_times(
        prices[BarCol.CLOSE],
        event_start_times,
        targets/2.0,
        (2.0, 2.0),
        events[EventCol.EXPIRY],
        0.0,
        sides=None
    )
    labels = get_event_labels(end_times, prices)

    expected_returns = pd.Series([101.17/99.02 - 1, 100.54/100.51 - 1, 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 1.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_drop_rare_labels():
    events = pd.DataFrame({EventCol.LABEL: [1]*5 + [2]*150 + [3]*8})
    filtered_events = drop_rare_labels(events, min_pct=0.05, min_classes=1)
    label_counts = filtered_events[EventCol.LABEL].value_counts().sort_index()
    pd.testing.assert_series_equal(label_counts, pd.Series([150, 8], index=[2, 3]), check_names=False)

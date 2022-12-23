import pandas as pd
import pytest

from financial_ml.data_structures.constants import BarCol, EventCol
from financial_ml.labeling.triple_barrier import get_event_returns, get_event_labels

event_start_times = pd.to_datetime(['2022-01-05', '2022-01-09', '2022-01-15'])
events = pd.DataFrame({
    EventCol.EXPIRY: pd.to_datetime(['2022-01-09', '2022-01-12', '2022-01-20'])
}, index=event_start_times)

targets = pd.Series([0.0098, 0.05, 0.01], index=event_start_times)


@pytest.fixture
def prices():
    prices_arr = [
        100., 100.47, 99.99, 99.04, 99.02, 99.12, 99.4, 101.17, 100.51, 100.92,
        100.4, 100.54, 99.55, 98.93, 100.02, 99.39, 98.72, 98.31, 98.35, 97.45
    ]
    prices = pd.DataFrame({
        BarCol.TIMESTAMP: pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 21)]),
        BarCol.CLOSE: prices_arr,
        BarCol.OPEN: prices_arr,
        BarCol.HIGH: prices_arr,
        BarCol.LOW: prices_arr
    }).set_index(BarCol.TIMESTAMP)
    return prices


@pytest.fixture
def prices_with_hi_lo():
    closes = [100., 100.47, 99.99, 99.04, 99.02, 99.12, 99.4, 101.17, 100.51, 100.92,
              100.4, 100.54, 99.55, 98.93, 100.02, 99.39, 98.72, 98.31, 98.35, 97.45]
    opens = [99.64, 100.51, 100.85, 99.02, 99.83, 99.71, 99.31, 101.16, 100.74, 101.61,
             100.01, 101.25, 99.33, 98.12, 100.03, 99.0, 97.88, 97.69, 98.85, 97.79]
    lows = [99.09, 100.3, 99.78, 98.32, 98.3, 98.24, 99.01, 100.91, 99.79, 100.71, 99.75,
            100.4, 98.8, 97.83, 99.26, 99.06, 97.01, 97.58, 97.88, 97.05]
    highs = [100.83, 101.33, 101.39, 99.79, 100.23, 100.36, 100.24, 101.81, 100.9, 102.28,
             101.29, 102.22, 99.71, 99.16, 100.73, 101.2, 99.46, 98.6, 99.28, 98.1]

    prices = pd.DataFrame({
        BarCol.TIMESTAMP: pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 21)]),
        BarCol.CLOSE: closes,
        BarCol.OPEN: opens,
        BarCol.HIGH: highs,
        BarCol.LOW: lows
    }).set_index(BarCol.TIMESTAMP)
    return prices


def test_get_event_labels_no_side(prices):
    event_returns = get_event_returns(
        prices,
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY]
    )
    pd.testing.assert_index_equal(event_returns.index, event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.EXPIRY], events[EventCol.EXPIRY])
    pd.testing.assert_series_equal(event_returns[EventCol.TARGET], targets, check_names=False)
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-08', '2022-01-12', '2022-01-17']), index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.END_TIME], expected_end_times, check_names=False)

    labels = get_event_labels(event_returns)
    expected_returns = pd.Series([101.17/99.02 - 1, 100.54/100.51 - 1, 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 1.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_labels_known_side(prices):
    sides = pd.Series([1, -1, 1], index=event_start_times)
    event_returns = get_event_returns(
        prices,
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
        sides=sides
    )
    pd.testing.assert_index_equal(event_returns.index, event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.EXPIRY], events[EventCol.EXPIRY])
    pd.testing.assert_series_equal(event_returns[EventCol.TARGET], targets, check_names=False)
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-08', '2022-01-12', '2022-01-17']), index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.END_TIME], expected_end_times, check_names=False)

    labels = get_event_labels(event_returns)
    expected_returns = pd.Series([101.17/99.02 - 1, 1*(100.54/100.51 - 1), 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 1.0, 0.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_labels_cancel_expired_event(prices):
    even_returns = get_event_returns(
        prices,
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY]
    )
    labels = get_event_labels(even_returns, cancel_expired_event=True)
    expected_labels = pd.Series([1.0, 0.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_get_event_end_times_no_pt_sl(prices):
    event_returns = get_event_returns(
        prices,
        event_start_times,
        targets,
        (0.0, 0.0),
        events[EventCol.EXPIRY]
    )
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-09', '2022-01-12', '2022-01-20']), index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.END_TIME], expected_end_times, check_names=False)


def test_get_event_labels_multipliers(prices):
    event_returns = get_event_returns(
        prices,
        event_start_times,
        targets/2.0,
        (2.0, 2.0),
        events[EventCol.EXPIRY]
    )
    labels = get_event_labels(event_returns)

    expected_returns = pd.Series([101.17/99.02 - 1, 100.54/100.51 - 1, 98.72/100.02 - 1], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.RETURN], expected_returns, check_names=False)
    expected_labels = pd.Series([1.0, 1.0, -1.0], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_labeling_with_hi_lo_prices(prices_with_hi_lo):
    event_returns = get_event_returns(
        prices_with_hi_lo,
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
    )
    expected_end_times = pd.Series(pd.to_datetime(['2022-01-05', '2022-01-12', '2022-01-16']), index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.END_TIME], expected_end_times, check_names=False)

    expected_returns = pd.Series([-0.015326, -0.001985, 0.011696], index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.RETURN], expected_returns,
                                   check_names=False, check_less_precise=4)

    labels = get_event_labels(event_returns)
    expected_labels = pd.Series([-1., -1., 1.], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)


def test_labeling_with_hi_lo_prices_and_sides(prices_with_hi_lo):
    sides = pd.Series([-1, 1, 1], index=event_start_times)
    event_returns = get_event_returns(
        prices_with_hi_lo,
        event_start_times,
        targets,
        (1.0, 1.0),
        events[EventCol.EXPIRY],
        sides=sides
    )
    expected_returns = pd.Series([0.015326, -0.001985, 0.011696], index=event_start_times)
    pd.testing.assert_series_equal(event_returns[EventCol.RETURN], expected_returns,
                                   check_names=False, check_less_precise=4)

    labels = get_event_labels(event_returns)
    expected_labels = pd.Series([1., 0., 1.], index=event_start_times)
    pd.testing.assert_series_equal(labels[EventCol.LABEL], expected_labels, check_names=False)

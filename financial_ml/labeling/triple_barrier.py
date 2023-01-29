import pandas as pd
import numpy as np

from financial_ml.data_structures.constants import EventCol, BarCol


def _get_triple_barrier_returns(prices: pd.DataFrame, events: pd.DataFrame, multipliers: (float, float)):
    """
    Get timestamps when price breaks the horizontal barriers (e.g. profit taking and stop loss limits)
    and vertical barrier (e.g. order cancelled or closed after some time limit) after each event

    :param prices: time series of Open, High and Low prices
    :param events: DataFrame of [EventCol.EXPIRY, EventCol.TARGET, EventCol.SIDE] and DateTimeIndex
        - An event is sampled/filtered from bars which seeds the onset of some trading actions
        - An event's expiry is the timestamp of the vertical barrier
        - Target should be in absolute return values
        - Side should be either 1 or -1 corresponding to a long or short trade
    :param multipliers: a tuple of 2 floats, 1 for each of the horizontal barrier.
        The barrier's width = multiplier * target
    :return: events DataFrame
    """
    # TODO: parallelization
    # TODO: handle non-unique event start times
    pt_mulp, sl_mulp = multipliers
    assert pt_mulp >= 0
    assert sl_mulp >= 0
    pt_limits = pt_mulp * events[EventCol.TARGET] if pt_mulp > 0 else pd.Series(index=events.index)
    st_limits = -sl_mulp * events[EventCol.TARGET] if sl_mulp > 0 else pd.Series(index=events.index)
    opens = prices[BarCol.OPEN]
    closes = prices[BarCol.CLOSE]
    lows = prices[BarCol.LOW]
    highs = prices[BarCol.HIGH]

    out = events.copy()
    for event_start, event_expiry in events[EventCol.EXPIRY].fillna(prices.index[-1]).items():
        start_price = opens.loc[event_start]
        path_lo_prices = lows[event_start:event_expiry]
        path_hi_prices = highs[event_start:event_expiry]
        path_lo_returns = path_lo_prices/start_price - 1
        path_hi_returns = path_hi_prices/start_price - 1
        side = int(events.at[event_start, EventCol.SIDE])
        assert side in [-1, 1]
        path_pt_returns = path_hi_returns if side == 1 else -path_lo_returns
        path_sl_returns = path_lo_returns if side == 1 else -path_hi_returns

        pt_time = path_pt_returns[path_pt_returns > pt_limits.loc[event_start]].index.min()
        sl_time = path_sl_returns[path_sl_returns < st_limits.loc[event_start]].index.min()
        out.loc[event_start, EventCol.PT_TIME] = pt_time
        out.loc[event_start, EventCol.SL_TIME] = sl_time
        if pd.isna(pt_time) and pd.isna(sl_time):
            out.loc[event_start, EventCol.END_TIME] = event_expiry
            out.loc[event_start, EventCol.RETURN] = closes[event_expiry]/start_price - 1
        elif pd.isna(sl_time) or pt_time < sl_time:
            out.loc[event_start, EventCol.END_TIME] = pt_time
            out.loc[event_start, EventCol.RETURN] = path_pt_returns[pt_time]
        else:
            out.loc[event_start, EventCol.END_TIME] = sl_time
            out.loc[event_start, EventCol.RETURN] = path_sl_returns[sl_time]

    return out


def get_event_returns(
        prices: pd.DataFrame,
        event_starts: pd.Series | pd.DatetimeIndex,
        targets: pd.Series,
        multipliers: tuple[float, float],
        event_expiries: pd.Series,
        min_target_return: float = 0.,
        sides=None
) -> pd.DataFrame:
    """
    Get events' end times when price reaches 1 of the horizontal barriers (e.g. profit taking and stop loss limits)
    or vertical barrier (e.g. order cancelled or closed after some time limit)

    :param prices: time series of Open, High and Low prices
    :param event_starts: a datetime series sampled from bars which seed the onset of some trading actions
    :param targets: a series of targets in absolute returns
    :param multipliers: a tuple of 2 floats, 1 for each of the horizontal barrier.
        The barrier's width = multiplier * target. If side=None, only the first multiplier is used for both barrier
    :param event_expiries: a series of timestamp that defined the max holding period after an event (vertical barrier)
        Set to None to disable all vertical barriers
    :param min_target_return: minimum target return
    :param sides: a Series of the side (1: long, 0: short) of each trade. Set to None if side is unknown.
    :return: events DataFrame
    """
    targets = targets.loc[event_starts]
    targets = targets[targets > min_target_return]
    if event_expiries is None:
        event_expiries = pd.Series(pd.NaT, index=event_starts)

    price_indices = prices.index.union(event_starts).union(event_expiries).dropna().drop_duplicates()
    prices = prices.reindex(labels=price_indices, method='bfill')

    if sides is None:
        event_sides = pd.Series(1.0, index=targets.index)
        multipliers = (multipliers[0], multipliers[0])
    else:
        event_sides = sides.loc[targets.index]

    events_df = pd.concat({
        EventCol.EXPIRY: event_expiries,
        EventCol.TARGET: targets,
        EventCol.SIDE: event_sides
    }, axis=1)
    events_df = _get_triple_barrier_returns(prices, events_df, multipliers)
    if sides is None:
        events_df = events_df.drop(columns=[EventCol.SIDE])
    return events_df


def get_event_labels(event_returns: pd.DataFrame, cancel_expired_event=False):
    """
    Get {-1, 0, 1} label for each event using the triple barriers labeling method.
    If the side of the event is specified, only {0, 1} labels are returned

    :param event_returns: events DataFrame output from `get_event_returns`.
    :param cancel_expired_event: if True, label 0 for events that exceed the vertical barrier.
        if False, use the sign of the returns
    :return: a DataFrame of [EventCol.RETURN, EventCol.LABEL] and DateTimeIndex corresponding to the events
    """
    event_returns = event_returns.dropna(subset=[EventCol.END_TIME])
    out = event_returns.copy()

    out[EventCol.LABEL] = np.sign(out[EventCol.RETURN])
    if EventCol.SIDE in event_returns.columns:
        out.loc[out[EventCol.RETURN] <= 0, EventCol.LABEL] = 0
    if cancel_expired_event:
        out.loc[event_returns[EventCol.END_TIME] >= event_returns[EventCol.EXPIRY], EventCol.LABEL] = 0.

    return out


def get_fixed_window_events(bars: pd.DataFrame, window: pd.Timedelta, feature_event_gap: pd.Timedelta,
                            drop_partial_events=True, align_with_bars=True, with_return=True):
    """
    Get windows of events with fixed duration

    :param bars: DataFrame with DateTimeIndex
    :param window: event's duration including the gap between the feature's bar and event's start bar
    :param feature_event_gap: interval between the feature's bar and event's start bar
    :param drop_partial_events: drop events that ends after the last bar
    :param align_with_bars: match event's times with the closest bars on or after them. Assume bars' index is sorted.
        Events that start after the last bar are removed. Events that end after the last bar are clipped.
    :param with_return: calculate log-return for each event
    :return: DataFrame of event start times, end times and optional returns
    """

    def _ret_in_period(rets, start, end):
        period_rets = rets.loc[start:end]
        return period_rets['c'].sum() + period_rets['oc'].iloc[0]

    event_starts = bars.index + feature_event_gap
    event_ends = bars.index + window
    events = pd.DataFrame({
        EventCol.START_TIME: event_starts,
        EventCol.END_TIME: event_ends
    }, index=bars.index)

    if drop_partial_events:
        is_completed = events[EventCol.END_TIME] <= bars.index[-1]
        events = events[is_completed]

    if align_with_bars:
        start_indices = bars.index.searchsorted(events[EventCol.START_TIME])
        is_started = start_indices < len(bars)
        start_indices = start_indices[is_started]
        events = events[is_started]
        events[EventCol.START_TIME] = bars.index[start_indices]

        end_indices = bars.index.searchsorted(events[EventCol.END_TIME])
        end_indices = np.minimum(end_indices, len(bars) - 1)
        events[EventCol.END_TIME] = bars.index[end_indices]

    if with_return:
        rets = pd.DataFrame({
            'c': np.log(bars[BarCol.CLOSE]).diff(),
            'oc': np.log(bars[BarCol.CLOSE]/bars[BarCol.OPEN])
        })
        event_rets = events.apply(
            lambda e: _ret_in_period(rets, e[EventCol.START_TIME], e[EventCol.END_TIME]), axis=1
        )
        events[EventCol.RETURN] = event_rets

    return events

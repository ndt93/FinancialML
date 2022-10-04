import pandas as pd
import numpy as np

from data_structures.constants import EventCol


def get_barrier_break_times(prices: pd.Series, events: pd.DataFrame, multipliers: (float, float)) -> pd.DataFrame:
    """
    Get timestamps when price breaks the horizontal barriers (e.g. profit taking and stop loss limits)
    and vertical barrier (e.g. order cancelled or closed after some time limit) after each event

    :param prices: a time series of prices with a DataTimeIndex
    :param events: a DataFrame of [EventCol.EXPIRY, EventCol.TARGET, EventCol.SIDE] and DateTimeIndex
        - An event is sampled/filtered from bars which seeds the onset of some trading actions
        - An event's expiry is the timestamp of the vertical barrier
        - Target should be in absolute return values
        - Side should be either 1 or -1 corresponding to a long or short trade
    :param multipliers: a tuple of 2 floats, 1 for each of the horizontal barrier.
        The barrier's width = multiplier * target
    :return: a DataFrame of [EventCol.EXPIRY, EventCol.PT_TIME, EventCol.SL_TIME] timestamps
    """
    res = events[[EventCol.EXPIRY]].copy()
    pt_mulp, sl_mulp = multipliers
    assert pt_mulp > 0
    assert sl_mulp > 0
    pt_limits = pt_mulp * events[EventCol.TARGET] if pt_mulp > 0 else pd.Series(index=events.index)
    st_limits = -sl_mulp * events[EventCol.TARGET] if sl_mulp > 0 else pd.Series(index=events.index)

    for event_time, event_expiry in events[EventCol.EXPIRY].fillna(prices.index[-1]).iteritems():
        path_prices = prices[event_time:event_expiry]
        path_returns = (path_prices/prices.loc[event_time] - 1) * events.at[event_time, EventCol.SIDE]
        res.loc[event_time, EventCol.PT_TIME] = path_returns[path_returns > pt_limits.loc[event_time]].index.min()
        res.loc[event_time, EventCol.SL_TIME] = path_returns[path_returns < st_limits.loc[event_time]].index.min()

    return res


def get_event_end_times(
        prices: pd.Series,
        events: pd.DatetimeIndex,
        targets: pd.Series,
        multipliers: (float, float),
        expiries: pd.Series,
        min_return: float,
        sides=None
) -> pd.DataFrame:
    """
    Get events' end times when price reaches 1 of the horizontal barriers (e.g. profit taking and stop loss limits)
    or vertical barrier (e.g. order cancelled or closed after some time limit)

    :param prices: a time series of prices with a DataTimeIndex
    :param events: a timeindex sampled from bars which seed the onset of some trading actions
    :param targets: a series of targets in absolute returns
    :param multipliers: a tuple of 2 floats, 1 for each of the horizontal barrier.
        The barrier's width = multiplier * target. If side=None, only the first multipler is used for both barrier
    :param expiries: a series of timestamp that defined the max holding period after an event (vertical barrier)
        Set to None to disable all vertical barriers
    :param min_return: minimum target return
    :param sides: a Series of the side (1: long, 0: short) of each trade. Set to None if side is unknown.
    :return: a DataFrame of EventCol.EXPIRY, EventCol.TARGET and EventCol.END_TIME.
        The end_time is the first timestamp where 1 of the horizontal or vertical barriers are reached
    """
    targets = targets.loc[events]
    targets = targets[targets > min_return]
    if expiries is None:
        expiries = pd.Series(pd.NaT, index=events)
    if sides is None:
        event_sides = pd.Series(1.0, index=targets.index)
        multipliers = (multipliers[0], multipliers[0])
    else:
        event_sides = sides.loc[targets.index]

    events_df = pd.concat({
        EventCol.EXPIRY: expiries,
        EventCol.TARGET: targets,
        EventCol.SIDE: event_sides
    }, axis=1)
    events_df = events_df.dropna(subset=[EventCol.TARGET])
    touch_times = get_barrier_break_times(prices, events_df, multipliers)
    events_df[EventCol.END_TIME] = touch_times.dropna(how='all').min(axis=1)
    if sides is None:
        events_df = events_df.drop(columns=[EventCol.SIDE])
    return events_df


def get_event_labels(events: pd.DataFrame, prices: pd.Series, cancel_expired_event=False):
    """
    Get {-1, 0, 1} label for each event using the triple barriers labeling method.
    If the side of the event is specified, only {0, 1} labels are returned

    :param events: DataFrame of [EventCol.EXPIRY, EventCol.END_TIME, EventCol.SIDE (optional)] and DateTimeIndex.
        See `get_event_end_times` function for more details
    :param prices: a time series of prices with DateTimeIndex
    :param expiries: a series of timestamp that defined the max holding period after an event (vertical barrier)
    :param cancel_expired_event: if True, label 0 for events that exceed the vertical barrier.
        if False, use the sign of the returns
    :return: a DataFrame of [EventCol.RETURN, EventCol.LABEL] and DateTimeIndex corresponding to the events
    """
    events = events.dropna(subset=[EventCol.END_TIME])
    all_start_end_times = events.index.union(events[EventCol.END_TIME].values).drop_duplicates()
    event_prices = prices.reindex(all_start_end_times, method='bill')
    event_start_prices = events.loc[events.index]
    event_end_prices = event_prices.loc[events[EventCol.END_TIME].values].values
    res = pd.DataFrame(index=events.index)

    res[EventCol.RETURN] = event_end_prices / event_start_prices - 1
    if EventCol.SIDE in events.columns:
        res[EventCol.RETURN] *= events[EventCol.SIDE]

    res[EventCol.LABEL] = np.sign(res[EventCol.RETURN])
    if EventCol.SIDE in events.columns:
        res.loc[res[EventCol.RETURN] <= 0, EventCol.LABEL] = 0
    if cancel_expired_event:
        res[EventCol.LABEL][events[EventCol.END_TIME] >= events[EventCol.EXPIRY]] = 0.

    return res

import numpy as np
import pandas as pd

from data_structures.constants import TickCol, BarCol, BarUnit
from utils.time_series import fast_ewma


def _compute_vwap(ticks):
    """
    :param ticks: DataFrame of TickCols
    :return: volume weighted average price over all ticks
    """
    p = ticks[TickCol.PRICE]
    v = ticks[TickCol.VOLUME]
    vwap = np.sum(p * v) / np.sum(v)
    return vwap


def _compute_bar_stats(ticks, timestamp='last'):
    """
    :param ticks: DataFrame of TickCols
    :param timestamp: set to None to exclude timestamp column in the result
    'first' or 'last' to use the timestamp of the first or last tick in the bar
    :return: DataFrame of BarCol columns
    """
    bar_cols = [BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP]
    bar = pd.Series([
        ticks[TickCol.PRICE].iloc[0],
        np.max(ticks[TickCol.PRICE]),
        np.min(ticks[TickCol.PRICE]),
        ticks[TickCol.PRICE].iloc[-1],
        np.sum(ticks[TickCol.VOLUME]),
        _compute_vwap(ticks)
    ], index=bar_cols)
    if timestamp == 'first':
        bar[BarCol.TIMESTAMP] = ticks[TickCol.TIMESTAMP].iloc[0]
    elif timestamp == 'last':
        bar[BarCol.TIMESTAMP] = ticks[TickCol.TIMESTAMP].iloc[-1]
    return bar


def _group_ticks_to_bars(df):
    df_grp = df.groupby('bar_id')
    df_res = df_grp.apply(_compute_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res


def aggregate_time_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: time interval per bar.
    see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :return: DataFrame of BarCol columns
    """
    df = ticks.set_index(TickCol.TIMESTAMP)
    df_grp = df.groupby(pd.Grouper(freq=frequency))
    df_res = df_grp.apply(_compute_bar_stats, timestamp=None)
    return df_res


def aggregate_tick_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: number of trades per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    df['bar_id'] = df.index // frequency
    return _group_ticks_to_bars(df)


def aggregate_volume_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: volume traded per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    cm_vol = df[TickCol.VOLUME].cumsum()
    df['bar_id'] = cm_vol // frequency
    return _group_ticks_to_bars(df)


def aggregate_dollar_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: dollars amount traded per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    cm_dollars = (df[TickCol.VOLUME]*df[TickCol.PRICE]).cumsum()
    df['bar_id'] = cm_dollars // frequency
    return _group_ticks_to_bars(df)


def _compute_tick_rule(ticks, b0):
    price_change = ticks[TickCol.PRICE].diff().values
    if len(price_change) == 0:
        return np.array([])
    tick_directions = np.zeros(len(price_change))
    tick_directions[0] = b0

    for i in range(1, len(price_change)):
        if price_change[i] == 0:
            tick_directions[i] = tick_directions[i - 1]
        else:
            tick_directions[i] = abs(price_change[i])/price_change[i]

    return tick_directions.astype(np.float64)


def _compute_imbalance_bar_sizes(
        b_arr, min_bar_size, max_bar_size, E_T_init, abs_E_b_init, T_ewma_window, b_ewma_window
):
    """
    b: tick direction or signed flow of dollars/volume. see _compute_tick_rule
    T: number of ticks in a bar
    E_T: expected value for T
    E_b: expected value for b
    theta: imbalance measure. sum of b in a bar

    For tick imbalance bars:
    T* = argmin_T {|theta| >= E_T*|2*P[b=1] - 1|}
    For dollars or volume imbalance bars:
    T* = argmin_T {|theta| >= E_T*|2*v_positive - E[v]|}
    E_T and |2*P[b=1] - 1| = |E_b| are estimated as EWMA of previous bars' T and b

    :param b_arr: array of tick directions or signed flow of dollars/volume
    :param min_bar_size: measured in same unit as b_arr
    :param max_bar_size: measured in same unit as b_arr
    :param E_T_init: initial value for E_T
    :param abs_E_b_init: initial value for |E_b|
    :param T_ewma_window: number of bars in a window used to calculate the EWMA of T. If None, all bars are used
    :param b_ewma_window: number of ticks in a window used to calculate the EWMA of b
    :return: end tick index for each bar, bar sizes, theta list, imbalance threshold list
    """

    T_arr = []
    bar_end_indices = []
    cur_bar_start = 0
    cur_E_T = E_T_init
    cur_abs_E_b = abs_E_b_init

    num_ticks = b_arr.shape[0]
    abs_theta_arr = np.zeros(num_ticks)
    thresholds = np.zeros(num_ticks)
    abs_theta_arr[0] = np.abs(b_arr[0])
    cur_theta = b_arr[0]
    cur_bar_size = abs(b_arr[0])
    print(f'Init: E_T={cur_E_T},abs_E_b={cur_abs_E_b},threshold={cur_E_T*cur_abs_E_b}')

    for i in range(1, num_ticks):
        cur_bar_size += abs(b_arr[i])
        cur_theta += b_arr[i]
        abs_theta = np.abs(cur_theta)
        abs_theta_arr[i] = abs_theta

        threshold = cur_E_T * cur_abs_E_b
        thresholds[i] = threshold

        if (abs_theta >= threshold and cur_bar_size >= min_bar_size) or cur_bar_size >= max_bar_size:
            T_arr.append(np.float64(i - cur_bar_start + 1))
            bar_end_indices.append(i)
            print(f'Bar: size={T_arr[-1]},end_tick={bar_end_indices[-1]}')

            cur_theta = 0
            cur_bar_size = 0
            cur_bar_start = i + 1
            cur_T_ewma_window = len(T_arr) if not T_ewma_window else T_ewma_window
            cur_E_T = fast_ewma(np.array(T_arr), window=np.int64(cur_T_ewma_window))[-1]
            cur_abs_E_b = np.abs(fast_ewma(b_arr[:i], window=np.int64(b_ewma_window))[-1])
            print(f'  Expect: E_T={cur_E_T},abs_E_b={cur_abs_E_b},threshold={cur_E_T*cur_abs_E_b}')

    if cur_bar_start < num_ticks:
        bar_end_indices.append(num_ticks - 1)
        T_arr.append(num_ticks - cur_bar_start)

    return bar_end_indices, T_arr, abs_theta_arr, thresholds


def aggregate_imblance_bars(
        ticks,
        bar_unit=BarUnit.TICK,
        min_bar_size=0,
        max_bar_size=np.inf,
        b0=1,
        E_T_init=1000,
        abs_E_b_init=None,
        T_ewma_window=None,
        b_ewma_window=3000
):
    """
    :param ticks: DataFrame of TickCols
    :param bar_unit: unit of bar size
    :param min_bar_size: min bar size in bar_unit
    :param max_bar_size: max bar size in bar_unit
    :param b0: tick direction for the first tick
    :param E_T_init: expected number of ticks in the first bar
    :param abs_E_b_init: expected absolute imbalance in the first bar.
        If None, the mean of tick directions or signed flow of volume/dollars of first E_T_init ticks is used
    :param T_ewma_window: window size for EWMA of bar sizes. If None or 0, all bars are used
    :param b_ewma_window: window size for EWMA of tick direction or volume/dollars signed flows
    :return: DataFrame of BarCol columns
    """
    b_arr = _compute_tick_rule(ticks, b0=b0)
    if bar_unit == BarUnit.VOLUME:
        b_arr *= ticks[TickCol.VOLUME]
    elif bar_unit == BarUnit.DOLLARS:
        b_arr *= ticks[TickCol.PRICE]*ticks[TickCol.VOLUME]
    elif bar_unit == BarUnit.TICK:
        pass
    else:
        raise NotImplementedError(bar_unit)

    if abs_E_b_init is None:
        abs_E_b_init = np.abs(b_arr[:E_T_init].mean())
    bar_end_indices, _, _, _ = _compute_imbalance_bar_sizes(
        b_arr, min_bar_size, max_bar_size, E_T_init, abs_E_b_init, T_ewma_window, b_ewma_window
    )

    num_ticks = ticks.shape[0]
    bar_id_arr = np.zeros(num_ticks)
    cur_bar_id = 0
    for i in range(num_ticks):
        if i > bar_end_indices[cur_bar_id]:
            cur_bar_id += 1
        bar_id_arr[i] = cur_bar_id

    df = ticks.reset_index(drop=True)
    df['bar_id'] = bar_id_arr
    return _group_ticks_to_bars(df)

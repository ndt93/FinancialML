import numpy as np
import pandas as pd

from financial_ml.data_structures.constants import TickCol, BarCol, BarUnit
from financial_ml.features.microstructure import tick_rule
from financial_ml.utils.time_series import fast_ewma


def _compute_vwap(ticks):
    """
    :param ticks: DataFrame of TickCols
    :return: volume weighted average price over all ticks
    """
    p = ticks[TickCol.PRICE]
    v = ticks[TickCol.VOLUME]
    vwap = np.sum(p * v) / np.sum(v)
    return vwap


def _compute_bar_stats(ticks, timestamp='first'):
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
    df_grp = df.groupby(TickCol.BAR_ID)
    df_res = df_grp.apply(_compute_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res


def aggregate_time_bars(ticks: pd.DataFrame, frequency):
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


def aggregate_tick_bars(ticks: pd.DataFrame, freq: float):
    """
    :param ticks: DataFrame of TickCol columns
    :param freq: number of trades per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    df[TickCol.BAR_ID] = df.index // freq
    return _group_ticks_to_bars(df)


def aggregate_volume_bars(ticks: pd.DataFrame, freq: float, return_bar_id=False):
    """
    :param ticks: DataFrame of TickCol columns
    :param freq: volume traded per bar
    :param return_bar_id: return ticks with TickCol.BAR_ID
    :return: DataFrame of BarCol columns, Optional[ticks with TickCol.BAR_ID]
    """
    df = ticks.reset_index(drop=True)
    cm_vol = df[TickCol.VOLUME].cumsum()
    df[TickCol.BAR_ID] = cm_vol // freq
    res = _group_ticks_to_bars(df)
    if return_bar_id:
        return res, df
    return res


def aggregate_dollar_bars(ticks: pd.DataFrame, freq: float):
    """
    :param ticks: DataFrame of TickCol columns
    :param freq: dollars amount traded per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    cm_dollars = (df[TickCol.VOLUME]*df[TickCol.PRICE]).cumsum()
    df[TickCol.BAR_ID] = cm_dollars // freq
    return _group_ticks_to_bars(df)


def _compute_tick_bar_id(ticks, bar_end_indices):
    num_ticks = ticks.shape[0]
    bar_id_arr = np.zeros(num_ticks)
    cur_bar_id = 0
    for i in range(num_ticks):
        if i > bar_end_indices[cur_bar_id]:
            cur_bar_id += 1
        bar_id_arr[i] = cur_bar_id
    return bar_id_arr


def _compute_imbalance_bar_sizes(
        b_arr, min_bar_size, max_bar_size,
        E_T_init, abs_E_b_init,
        T_ewma_span, b_ewma_span
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
    :param T_ewma_span: number of bars in EWMA span of number of ticks in bar. If None or 0, all bars are used
    :param b_ewma_span: number of ticks in EWMA span of tick direction or volume/dollars signed flows.
    :return: end tick index for each bar, bar sizes, theta array, imbalance threshold array
    """

    def _update_threshold():
        return cur_E_T * cur_abs_E_b

    num_ticks = b_arr.shape[0]
    bar_end_indices = []
    T_arr = []

    cur_bar_start = 0
    cur_bar_size = abs(b_arr[0])
    cur_E_T = E_T_init
    cur_abs_E_b = abs_E_b_init
    cur_theta = b_arr[0]
    cur_threshold = _update_threshold()
    is_padding_bar = False

    abs_theta_arr = np.zeros(num_ticks)
    abs_theta_arr[0] = np.abs(b_arr[0])
    thresholds = np.zeros(num_ticks)
    thresholds[0] = cur_threshold

    # print(f'Init: E_T={cur_E_T},abs_E_b={cur_abs_E_b},threshold={cur_threshold}')

    for i in range(1, num_ticks):
        cur_bar_size += abs(b_arr[i])
        cur_theta += b_arr[i]
        abs_theta = np.abs(cur_theta)

        abs_theta_arr[i] = abs_theta
        thresholds[i] = cur_threshold

        if abs_theta >= cur_threshold or cur_bar_size >= max_bar_size or is_padding_bar:
            if cur_bar_size < min_bar_size:
                is_padding_bar = True
                continue

            T_arr.append(np.float64(i - cur_bar_start + 1))
            bar_end_indices.append(i)
            # print(f'Bar: size={cur_bar_size},num_ticks={T_arr[-1]},end_tick={bar_end_indices[-1]}')

            is_padding_bar = False
            cur_bar_start = i + 1
            cur_bar_size = 0
            cur_T_ewma_span = len(T_arr) if not T_ewma_span else T_ewma_span
            cur_E_T = fast_ewma(np.array(T_arr), span=np.int64(cur_T_ewma_span), adjust=True)[-1]
            cur_b_ewma_span = min(b_ewma_span, i + 1)
            cur_abs_E_b = np.abs(fast_ewma(b_arr[:i], span=np.int64(cur_b_ewma_span), adjust=True)[-1])
            cur_theta = 0
            cur_threshold = cur_E_T * cur_abs_E_b
            # print(f'  Expect: E_T={cur_E_T},abs_E_b={cur_abs_E_b},threshold={cur_threshold}')

    if cur_bar_start < num_ticks:
        bar_end_indices.append(num_ticks - 1)
        T_arr.append(num_ticks - cur_bar_start)

    return bar_end_indices, T_arr, abs_theta_arr, thresholds


def aggregate_imblance_bars(
        ticks: pd.DataFrame,
        bar_unit=BarUnit.TICK,
        min_bar_size=0,
        max_bar_size=np.inf,
        b0=1,
        E_T_init=1000,
        abs_E_b_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
        debug=False
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
    :param T_ewma_span: number of bars in EWMA span of number of ticks in bar.
        If None or 0, all bars are used
    :param b_ewma_span: number of ticks in EWMA span of tick direction or volume/dollars signed flows.
        If None or 0, 3*E_T_init is used.
    :param debug: returns ticks df with addition info
    :return: DataFrame of BarCol columns
    """
    b_arr = tick_rule(ticks, b0=b0)
    if bar_unit == BarUnit.VOLUME:
        b_arr *= ticks[TickCol.VOLUME].values
    elif bar_unit == BarUnit.DOLLARS:
        b_arr *= (ticks[TickCol.PRICE]*ticks[TickCol.VOLUME]).values
    elif bar_unit == BarUnit.TICK:
        pass
    else:
        raise NotImplementedError(bar_unit)

    if abs_E_b_init is None:
        abs_E_b_init = np.abs(b_arr[:E_T_init].mean())
    if b_ewma_span is None:
        b_ewma_span = 3 * E_T_init
    bar_end_indices, bar_sizes, thetas, thresholds = _compute_imbalance_bar_sizes(
        b_arr, min_bar_size, max_bar_size, E_T_init, abs_E_b_init, T_ewma_span, b_ewma_span
    )

    bar_id_arr = _compute_tick_bar_id(ticks, bar_end_indices)
    df = ticks.reset_index(drop=True)
    df[TickCol.BAR_ID] = bar_id_arr
    res = _group_ticks_to_bars(df)
    if debug:
        df = df.assign(theta=thetas, threshold=thresholds)
        return res, df
    return res


def _compute_runs_bar_sizes(
        b_arr, v_arr, min_bar_size, max_bar_size,
        E_T_init, P_b_buy_init, E_v_buy_init, E_v_sell_init,
        T_ewma_span, b_ewma_span
):
    """
    b: tick direction. see _compute_tick_rule
    v: quantity traded in volume or dollars
    T: number of ticks in a bar
    E_T: expected value for T
    P_b_buy: Probability of buy tick
    E_v_buy: Expected value of buy volume/dollars
    E_v_sell: Expected value of sell volume/dollars
    theta: runs measure. sum of b in a bar

    For tick runs bars:
    T* = argmin_T {theta >= E_T*max{P[b=1], 1-P[b=1]}}
    For dollars or volume runs bars:
    T* = argmin_T {theta >= E_T*max{P[b=1]E[v|b=1], (1-P[b=1])E[v|b=-1]}}
    E_T, P[b=1], E[v|b=1] and E[v|b=-1] are estimated as EWMA of previous bars' T and b

    :param b_arr: array of tick directions. see _compute_tick_rule
    :param v_arr: array of quantity trade in volume/dollars
    :param min_bar_size: measured in same unit as b_arr
    :param max_bar_size: measured in same unit as b_arr
    :param E_T_init: initial value for E_T
    :param P_b_buy_init: initial value for P_b_buy
    :param E_v_buy_init: initial value for E_v_buy
    :param E_v_sell_init: initial value for E_v_sell
    :param T_ewma_span: number of bars in EWMA span of number of ticks in bar. If None or 0, all bars are used
    :param b_ewma_span: number of ticks in EWMA span of volume/dollars quantity traded.
    :return: end tick index for each bar, bar sizes, theta array, runs threshold array
    """

    def _update_threshold():
        return cur_E_T * max(cur_P_b_buy*cur_E_v_buy, (1 - cur_P_b_buy)*cur_E_v_sell)

    num_ticks = b_arr.shape[0]
    bar_end_indices = []
    T_arr = []
    b_buy_ratio_arr = []
    v_buy_arr = []
    v_sell_arr = []

    cur_E_T = E_T_init
    cur_P_b_buy = P_b_buy_init
    cur_E_v_buy = E_v_buy_init
    cur_E_v_sell = E_v_sell_init

    cur_buy_runs = b_arr[0] if b_arr[0] >= 0 else 0
    cur_v_buy_runs = v_arr[0] if b_arr[0] >= 0 else 0
    cur_v_sell_runs = v_arr[0] if b_arr[0] < 0 else 0
    cur_theta = max(cur_v_buy_runs, cur_v_sell_runs)
    cur_threshold = _update_threshold()
    cur_bar_start = 0
    cur_bar_size = v_arr[0]
    is_padding_bar = False

    theta_arr = np.zeros(num_ticks)
    theta_arr[0] = cur_theta
    thresholds = np.zeros(num_ticks)
    thresholds[0] = cur_threshold

    # print(f'Init: E_T={cur_E_T},P_b_buy={cur_P_b_buy},E_v_buy={cur_E_v_buy},' +
    #       f'E_v_sell={cur_E_v_sell},threshold={cur_threshold}')

    for i in range(1, num_ticks):
        cur_bar_size += v_arr[i]
        if b_arr[i] >= 0:
            v_buy_arr.append(v_arr[i])
        if b_arr[i] < 0:
            v_sell_arr.append(v_arr[i])
        cur_buy_runs += 1 if b_arr[i] >= 0 else 0
        cur_v_buy_runs += v_arr[i] if b_arr[i] >= 0 else 0
        cur_v_sell_runs += v_arr[i] if b_arr[i] < 0 else 0
        cur_theta = max(cur_v_buy_runs, cur_v_sell_runs)

        theta_arr[i] = cur_theta
        thresholds[i] = cur_threshold

        if cur_theta >= cur_threshold or cur_bar_size >= max_bar_size or is_padding_bar:
            if cur_bar_size < min_bar_size:
                is_padding_bar = True
                continue

            ticks_in_bar = np.float64(i - cur_bar_start + 1)
            T_arr.append(ticks_in_bar)
            b_buy_ratio_arr.append(cur_buy_runs / ticks_in_bar)
            bar_end_indices.append(i)
            # print(f'Bar: size={cur_bar_size},num_ticks={T_arr[-1]},end_tick={bar_end_indices[-1]}')

            is_padding_bar = False
            cur_buy_runs = 0
            cur_v_buy_runs = 0
            cur_v_sell_runs = 0
            cur_bar_size = 0
            cur_bar_start = i + 1
            cur_T_ewma_span = len(T_arr) if not T_ewma_span else T_ewma_span
            cur_E_T = fast_ewma(np.array(T_arr), span=np.int64(cur_T_ewma_span), adjust=True)[-1]
            cur_P_b_buy = fast_ewma(np.array(b_buy_ratio_arr), span=np.int64(cur_T_ewma_span), adjust=True)[-1]
            cur_v_buy_ewma_span = min(len(v_buy_arr), b_ewma_span)
            cur_E_v_buy = fast_ewma(np.array(v_buy_arr), span=np.int64(cur_v_buy_ewma_span), adjust=True)[-1]
            cur_v_sell_ewma_span = min(len(v_sell_arr), b_ewma_span)
            cur_E_v_sell = fast_ewma(np.array(v_sell_arr), span=np.int64(cur_v_sell_ewma_span), adjust=True)[-1]
            cur_threshold = _update_threshold()
            # print(f'  Expect: E_T={cur_E_T},P_b_buy={cur_P_b_buy},E_v_buy={cur_E_v_buy},' +
            #       f'E_v_sell={cur_E_v_sell},threshold={cur_threshold}')

    if cur_bar_start < num_ticks:
        bar_end_indices.append(num_ticks - 1)
        T_arr.append(num_ticks - cur_bar_start)

    return bar_end_indices, T_arr, theta_arr, thresholds


def aggregate_runs_bars(
        ticks: pd.DataFrame,
        bar_unit=BarUnit.TICK,
        min_bar_size=0,
        max_bar_size=np.inf,
        b0=1,
        E_T_init=1000,
        P_b_buy_init=None,
        E_v_buy_init=None,
        E_v_sell_init=None,
        T_ewma_span=None,
        b_ewma_span=None,
        debug=False
):
    """
    :param ticks: DataFrame of TickCols
    :param bar_unit: unit of bar size
    :param min_bar_size: min bar size in bar_unit
    :param max_bar_size: max bar size in bar_unit
    :param b0: tick direction for the first tick
    :param E_T_init: expected number of ticks in the first bar
    :param P_b_buy_init: expected ration of buy tick direction in the first bar.
        If None, use first E_T_unit ticks to estimate
    :param E_v_buy_init: expected quantity in bar_unit traded over a buy tick direction in the first bar.
        If None, use first E_T_unit ticks to estimate
    :param E_v_sell_init: expected quantity in bar_unit traded over a sell tick direction in the first bar
        If None, use first E_T_unit_ticks_to_estimate
    :param T_ewma_span: number of bars in EWMA span of number of ticks in bar.
        If None or 0, all bars are used
    :param b_ewma_span: number of ticks in EWMA span of volume/dollars quantity traded.
        If None or 0, 3*E_T_init is used
    :param debug: return ticks DataFrame with additional colums for debugging
    :return: DataFrame of BarCol columns
    """
    b_arr = tick_rule(ticks, b0=b0)
    if bar_unit == BarUnit.VOLUME:
        v_arr = ticks[TickCol.VOLUME]
    elif bar_unit == BarUnit.DOLLARS:
        v_arr = ticks[TickCol.PRICE]*ticks[TickCol.VOLUME]
    elif bar_unit == BarUnit.TICK:
        v_arr = np.ones(len(b_arr), dtype=np.float64)
    else:
        raise NotImplementedError(bar_unit)

    if P_b_buy_init is None:
        P_b_buy_init = np.sum(b_arr[:E_T_init] > 0)/len(b_arr[:E_T_init])
    if E_v_buy_init is None:
        E_v_buy_init = np.mean(v_arr[:E_T_init][(b_arr[:E_T_init] > 0)])
    if E_v_sell_init is None:
        E_v_sell_init = np.mean(v_arr[:E_T_init][(b_arr[:E_T_init] < 0)])
    if b_ewma_span is None:
        b_ewma_span = 3 * E_T_init

    bar_end_indices, bar_size, thetas, thresholds = _compute_runs_bar_sizes(
        b_arr,
        v_arr,
        min_bar_size,
        max_bar_size,
        E_T_init,
        P_b_buy_init,
        E_v_buy_init,
        E_v_sell_init,
        T_ewma_span,
        b_ewma_span
    )

    bar_id_arr = _compute_tick_bar_id(ticks, bar_end_indices)
    df = ticks.reset_index(drop=True)
    df[TickCol.BAR_ID] = bar_id_arr
    res = _group_ticks_to_bars(df)
    if debug:
        df = df.assign(theta=thetas, threshold=thresholds)
        return res, df
    return res


def _resample_vwap(bars: pd.DataFrame):
    """
    :param bars: DataFrame of BarCols
    :return: volume weighted average price over resampled bars
    """
    p = bars[BarCol.VWAP]
    v = bars[BarCol.VOLUME]
    vwap = np.sum(p * v) / np.sum(v)
    return vwap


def _resample_bar_stats(bars: pd.DataFrame, timestamp='first'):
    """
    :param bars: DataFrame of BarCols
    :param timestamp: set to None to exclude timestamp column in the result
        'first' or 'last' to use the timestamp of the first or last tick in the bar
    :return: DataFrame of BarCol columns
    """
    bar_cols = [BarCol.OPEN, BarCol.HIGH, BarCol.LOW, BarCol.CLOSE, BarCol.VOLUME, BarCol.VWAP]
    bar = pd.Series([
        bars[BarCol.OPEN].iloc[0],
        np.max(bars[BarCol.HIGH]),
        np.min(bars[BarCol.LOW]),
        bars[BarCol.CLOSE].iloc[-1],
        np.sum(bars[BarCol.VOLUME]),
        _resample_vwap(bars)
    ], index=bar_cols)
    if timestamp == 'first':
        bar[BarCol.TIMESTAMP] = bars[BarCol.TIMESTAMP].iloc[0]
    elif timestamp == 'last':
        bar[BarCol.TIMESTAMP] = bars[BarCol.TIMESTAMP].iloc[-1]
    return bar


def _group_resampled_bars(df):
    df_grp = df.groupby(TickCol.BAR_ID)
    df_res = df_grp.apply(_resample_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res


def resample_time_to_volume_bars(bars: pd.DataFrame, freq: float, return_bar_id=False):
    df = bars.reset_index()
    cm_vol = df[BarCol.VOLUME].cumsum()
    df[TickCol.BAR_ID] = cm_vol // freq
    res = _group_resampled_bars(df)
    if return_bar_id:
        return res, df
    return res

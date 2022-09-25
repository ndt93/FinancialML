import numpy as np
import pandas as pd

from data_structures.constants import TickCol, BarCol


def compute_vwap(ticks):
    """
    :param ticks: DataFrame of TickCols
    :return: volume weighted average price over all ticks
    """
    p = ticks[TickCol.PRICE]
    v = ticks[TickCol.VOLUME]
    vwap = np.sum(p * v) / np.sum(v)
    return vwap


def compute_bar_stats(ticks, timestamp='last'):
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
        compute_vwap(ticks)
    ], index=bar_cols)
    if timestamp == 'first':
        bar[BarCol.TIMESTAMP] = ticks[TickCol.TIMESTAMP].iloc[0]
    elif timestamp == 'last':
        bar[BarCol.TIMESTAMP] = ticks[TickCol.TIMESTAMP].iloc[-1]
    return bar


def aggregate_time_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: time interval per bar.
    see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :return: DataFrame of BarCol columns
    """
    df = ticks.set_index(TickCol.TIMESTAMP)
    df_grp = df.groupby(pd.Grouper(freq=frequency))
    df_res = df_grp.apply(compute_bar_stats, timestamp=None)
    return df_res


def aggregate_tick_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: number of trades per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    df['bar_id'] = df.index // frequency
    df_grp = df.groupby('bar_id')
    df_res = df_grp.apply(compute_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res


def aggregate_volume_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: volume traded per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    cm_vol = df[TickCol.VOLUME].cumsum()
    df['bar_id'] = cm_vol // frequency
    df_grp = df.groupby('bar_id')
    df_res = df_grp.apply(compute_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res


def aggregate_dollar_bars(ticks, frequency):
    """
    :param ticks: DataFrame of TickCol columns
    :param frequency: dollars amount traded per bar
    :return: DataFrame of BarCol columns
    """
    df = ticks.reset_index(drop=True)
    cm_dollars = (df[TickCol.VOLUME]*df[TickCol.PRICE]).cumsum()
    df['bar_id'] = cm_dollars // frequency
    df_grp = df.groupby('bar_id')
    df_res = df_grp.apply(compute_bar_stats)
    df_res.set_index(BarCol.TIMESTAMP, inplace=True)
    return df_res

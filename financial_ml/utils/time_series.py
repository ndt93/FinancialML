import numpy as np
import pandas as pd
from numba import jit, float64, int64, boolean


@jit((float64[:], int64, boolean), nopython=True, nogil=True)
def fast_ewma(series, span, adjust):
    """
    Faster runtime implementation of pandas's ewm with adjust=True

    :param series: array of numerical values
    :param span: specify decay in terms of span. alpha = 2/(span + 1)
    :return: ewm series
    """
    n = series.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    alpha = 2 / float(span + 1)
    if adjust:
        w = 1
        ewma_old = series[0]
        ewma[0] = ewma_old
        for i in range(1, n):
            w += (1 - alpha)**i
            ewma_old = ewma_old * (1 - alpha) + series[i]
            ewma[i] = ewma_old / w
        return ewma
    else:
        ewma[0] = series[0]
        for i in range(1, n):
            ewma[i] = series[i] * alpha + ewma[i - 1] * (1 - alpha)
        return ewma


def get_daily_volatility(close: pd.Series, ewm_span: int, is_intraday=True):
    """
    :param close: Series of close prices
    :param ewm_span: ewm days span for estimating the volatility
    :param is_intraday: True if series contain intraday data. False if only daily data is available
    :return:
    """
    prev_day_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prev_day_idx = prev_day_idx[prev_day_idx > 0]
    if is_intraday:
        prev_day_idx -= 1
    day_pairs = pd.Series(close.index[prev_day_idx], index=close.index[close.shape[0] - prev_day_idx.shape[0]:])
    daily_returns = pd.Series(close.loc[day_pairs.index].values / close.loc[day_pairs.values].values) - 1
    return daily_returns.ewm(span=ewm_span).std()


def get_vertical_barriers(series: pd.Series, events: pd.Series, holding_period: int):
    """
    :param series: a series with a DateTimeIndex
    :param events: a series of events' timestamps
    :param holding_period: number of days after each event where there's a vertical barrier
    :return: a series of timestamps of the vertical barriers
    """
    barrier_indices = series.index.searchsorted(events + pd.Timedelta(days=holding_period))
    barrier_indices = barrier_indices[barrier_indices < series.shape[0]]
    barriers = pd.Series(series.index[barrier_indices], index=events[:barrier_indices.shape[0]])

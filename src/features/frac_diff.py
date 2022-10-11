import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from data_structures.constants import StatsCol


def _get_weights(order: float, size: int) -> np.ndarray:
    """
    :param order: order of differencing
    :param size: size of the weights sequence
    :return: a column vector of weights for the oldest to the newest data point
    """
    weights = [1.0]
    for k in range(1, size):
        next_w = -weights[-1] * (order - k + 1) / k
        weights.append(next_w)
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


def frac_diff_expanding(series: pd.DataFrame, order: float, threshold=1.):
    """
    Apply a standard fractional differencing using expanding window

    :param series: a time series of features
    :param order: order of differencing
    :param threshold: Initial data points with weights loss ratio larger than this will be skipped.
        Value is between 0 and 1. Set to 1 to not skip any data point.
    :return: a fractionally differenced version of the input series
    """
    assert 0 <= threshold <= 1
    weights = _get_weights(order, series.shape[0])
    weight_losses = np.cumsum(abs(weights))
    weight_losses /= weight_losses[-1]
    num_skips = (weight_losses[:-1] > threshold).sum()
    res = {}

    for col in series.columns:
        col_series = series[[col]].fillna(method='ffill').dropna()
        col_res = pd.Series()
        for iloc in range(num_skips, col_series.shape[0]):
            loc = col_series.index[iloc]
            if not np.isfinite(col_series.loc[loc, col]):
                continue
            col_res[loc] = np.dot(weights[-(iloc + 1):, :].T, col_series.loc[:loc])[0, 0]
        res[col] = col_res.copy(deep=True)

    res = pd.concat(res, axis=1)
    return res


def _get_weights_fixed(order: float, threshold: float, max_width: int) -> np.ndarray:
    """
    :param order: order of differencing
    :param threshold: discard weights with abs value below this threshold
    :return: a column vector of weights for the oldest to the newest data point
    """
    weights = [1.0]
    k = 1
    while k < max_width:
        next_w = -weights[-1] * (order - k + 1) / k
        if abs(next_w) < threshold:
            break
        weights.append(next_w)
        k += 1
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


def frac_diff_fixed(series: pd.DataFrame, order: float, threshold=1e-5):
    """
    Apply fractional differencing using fixed window

    :param series: a time series of features
    :param order: order of differencing
    :param threshold: weights with abs value less than this will be dropped
    :return: a fractionally differenced version of the input series 
    """
    weights = _get_weights_fixed(order, threshold, series.shape[0])
    width = len(weights) - 1
    res = {}

    for col in series.columns:
        col_series = series[[col]].fillna(method='ffill').dropna()
        col_res = pd.Series(dtype=float)
        for iloc in range(width, col_series.shape[0]):
            start_loc = col_series.index[iloc - width]
            end_loc = col_series.index[iloc]
            if not np.isfinite(series.loc[end_loc, col]):
                continue
            col_res[end_loc] = np.dot(weights.T, col_series.loc[start_loc:end_loc])[0, 0]
        res[col] = col_res.copy(deep=True)

    res = pd.concat(res, axis=1)
    return res


def run_adf_tests(bars: pd.DataFrame, price_col: str, orders=np.linspace(0, 1, 11)):
    """
    Perform the Augmented Dickey-Fuller test on each of the order of fractionally differenced series

    :param bars: a DataFrame of price series
    :param price_col: column of the price series in bars
    :param orders: list of order for apply the frac_diff
    :return: a DataFrame with ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'] columns
    """
    adf_out = pd.DataFrame(columns=[
        StatsCol.STAT, StatsCol.P_VAL, StatsCol.LAG, StatsCol.NUM_OBS, StatsCol.CONF_95, StatsCol.CORR
    ])
    for d in orders:
        log_prices = np.log(bars[[price_col]])
        frac_diff_prices = frac_diff_fixed(log_prices, d, threshold=0.01)
        corr = np.corrcoef(
            log_prices.loc[frac_diff_prices.index, price_col],
            frac_diff_prices[price_col]
        )[0, 1]
        adf_res = adfuller(frac_diff_prices, maxlag=1, regression='c', autolag=None)
        adf_out.loc[d] = list(adf_res[:4]) + [adf_res[4]['5%']] + [corr]
    return adf_out

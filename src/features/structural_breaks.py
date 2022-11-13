import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.recursive_ls import RecursiveLS


# --- CUSUM Tests ---

def recursive_residuals_cusum_stats(X: np.ndarray, y: np.ndarray):
    """
    Compute the Brown-Durbin-Evans CUSUM test statistic on the least square recursive residuals.
    Under the null hypothesis of stable coefficient beta(t) = beta, distribution of CUSUM statistic:
        S(t) ~ N[0, t - k - 1]. t = 1,..,T

    :param X: Txk matrix of time series of k features
    :param y: timeseries of T labels
    :return: timeseries of CUSUM statistics, p-values, statsmodels RecursiveLSResults object
    """
    rls = RecursiveLS(y, X)
    res = rls.fit()
    cusum = res.cusum
    null_dist_std = np.sqrt(X.shape[0] - X.shape[1] - 1)
    pval = np.minimum(
        norm.cdf(cusum, loc=0, scale=null_dist_std),
        1. - norm.cdf(cusum, loc=0, scale=null_dist_std)
    )
    return cusum, pval, res


def levels_cusum_stats(y: np.ndarray, n: int):
    """
    Compute the Chu-Stinchcombe-White CUSUM test statistic on levels S(n, t).
    Under the null hypothesis of no level change i.e. beta = 0: S(n, t) ~ N(0, 1). t = 1,...,T

    :param y: timeseries of (T > 2) prices or returns
    :param n: index of reference level
    :return: timeseries of (T - n - 1) CUSUM statistics
    """
    if len(y) < 2:
        return None
    diff = np.diff(y)
    t = np.arange(2, len(y) + 1)
    sigma = np.sqrt(1/(t - 1)*np.cumsum(diff**2))
    res = (y[n+1:] - y[n])*(sigma[n:]*np.sqrt(t[n:] - (n + 1)))**-1
    pval = np.minimum(norm.cdf(res), 1. - norm.cdf(res))
    return res, pval


# --- Explosiveness Test: Supremum Augmented Dickey-Fuller ---

def _get_lag_df(df: pd.DataFrame, lags: int | list[int]):
    """
    Get DataFrame with columns containing lagged values of input DataFrame
    :param df: input DataFrame
    :param lags: a number for max lags or list of lags
    :return: DataFrame columns for each lag value
    """
    df_out = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    for lag in lags:
        df_lag = df.shift(lag).copy(deep=True)
        df_lag.columns = [f'{c}_{lag}' for c in df_lag.columns]
        df_out = df_out.join(df_lag, how='outer')
    return df_out


def _get_yx(series: pd.DataFrame, constant: str, lags: int | list[int]):
    """
    Prepare the X and y matrices for the autoregressive models in ADF test

    :param series: A timeseries DataFrame
    :param constant: 'nc' if no constant, 'ct' for linear time trend, 'ctt' for quadratic time trend
    :param lags: a number for max lags or list of lags
    :return: X and y matrices
    """
    diff = series.diff().dropna()
    x = _get_lag_df(diff, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0]-1:-1, 0]
    y = diff.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
            if constant == 'ctt':
                x = np.append(x, trend**2, axis=1)
    return y, x


def _fit_adf(y: np.ndarray, x: np.ndarray):
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xx_inv = np.linalg.inv(xx)
    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(x, b_mean)
    b_var = np.dot(err.T, err)/(x.shape[0] - x.shape[1])*xx_inv
    return b_mean, b_var


def sadf_stat(series: pd.DataFrame, min_sample_len, constant, lags):
    """
    Calculate the Supremum Augmented Dickey-Fuller test statistic.

    :param series: a timeseries of prices or returns
    :param min_sample_len: min number of samples for the ADF regression
    :param constant: 'nc' if no constant, 'ct' for linear time trend, 'ctt' for quadratic time trend
    :param lags: a number for max lags or list of lags
    :return: the SADF test statistic
    """
    y, x = _get_yx(series, constant, lags)
    start_points = range(0, y.shape[0] + lags - min_sample_len + 1)
    all_adf = []
    max_adf = -np.inf
    for start in start_points:
        y_sub, x_sub = y[start:], x[start:]
        b_mean, b_var = _fit_adf(y_sub, x_sub)
        b_mean, b_std = b_mean[0, 0], np.sqrt(b_var[0, 0])
        all_adf.append(b_mean/b_std)
        max_adf = max(max_adf, all_adf[-1])
    return max_adf


def sadf_stat_series(series: pd.DataFrame, min_sample_len, constant, lags):
    """
    Calculate the Supremum Augmented Dickey-Fuller test statistic for each point in a time series

    :param series: a timeseries of prices or returns. Log-prices are generally better to keep return variance
        constants relative to the raw price level
    :param min_sample_len: min number of samples for the ADF regression
    :param constant: 'nc' if no constant, 'ct' for linear time trend, 'ctt' for quadratic time trend
    :param lags: a number for max lags or list of lags
    :return: time series of SADF test statistics
    """
    res = [
        sadf_stat(series[:t+1], min_sample_len, constant, lags)
        for t in range(min_sample_len, series.shape[0])
    ]
    res = pd.Series(res, index=series.index[min_sample_len:])
    return res

# TODO: Explosiveness tests: Quantile ADF, Conditional ADF, Sub- & Super- Martingale Tests

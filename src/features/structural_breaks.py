import numpy as np
from statsmodels.regression.recursive_ls import RecursiveLS


def recursive_residuals_cusum_stats(X: np.ndarray, y: np.ndarray):
    """
    Compute the Brown-Durbin-Evans CUSUM test statistic on the least square recursive residuals.
    Under the null hypothesis of stable coefficient beta(t) = beta, distribution of CUSUM statistic:
        S(t) ~ N[0, t - k - 1]. t = 1,..,T

    :param X: Txk matrix of time series of features
    :param y: timeseries of T labels
    :return: timeseries of CUSUM statistics, statsmodels RecursiveLSResults object
    """
    rls = RecursiveLS(y, X)
    res = rls.fit()
    return res.cusum, res


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
    sigma = 1/(t - 1)*np.cumsum(diff**2)
    res = (y[n+1:] - y[n])*(sigma[n:]*np.sqrt(t[:n] - (n + 1)))**-1
    return res

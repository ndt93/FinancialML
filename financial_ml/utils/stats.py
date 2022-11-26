import pandas as pd
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro, rv_continuous, kendalltau, weightedtau


# Auto-correlation tests
def pearson_autocorr(series, lag=1):
    return pd.Series(series).autocorr(lag=lag)


def durbin_watson_stat(series):
    return durbin_watson(series)


# Ordinal/rank correlation tests
def kendall_tau_pval(x, y):
    return kendalltau(x, y)[0]


def weighted_tau_pval(x, y):
    return weightedtau(x, y)[0]


# Normality tests
def jarque_bera_pval(series):
    return jarque_bera(series)[1]


def shapiro_wilk_pval(series):
    return shapiro(series)[1]


# Stationarity test
def augmented_dickey_fuller_pval(series, **kwargs):
    return adfuller(series, **kwargs)[1]


# Scipy's rv_continuous class from a KDE estimation
class KDERv(rv_continuous):

    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde

    def _pdf(self, x, *args):
        return self._kde.pdf(x)

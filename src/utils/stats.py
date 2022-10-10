import pandas as pd
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro


# Auto-correlation tests
def pearson_autocorr(series, lag=1):
    return pd.Series(series).autocorr(lag=lag)


def durbin_watson_stat(series):
    return durbin_watson(series)


# Normality tests
def jarque_bera_pval(series):
    return jarque_bera(series)[1]


def shapiro_wilk_pval(series):
    return shapiro(series)[1]


# Stationarity test
def augmented_dickey_fuller_pval(series, **kwargs):
    return adfuller(series, **kwargs)[1]

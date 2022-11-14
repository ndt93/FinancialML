import string

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


# --- Entropy estimators ---

def _stringify(msg):
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    return msg


def _word_pmf(msg: str | list, w: int):
    """
    Find probability mass function of all words of length w in msg
    """
    lib = {}
    msg = _stringify(msg)
    for i in range(w, len(msg)):
        msg_sub = msg[i-w:i]
        if msg_sub not in lib:
            lib[msg_sub] = [i-w]
        else:
            lib[msg_sub] = lib[msg_sub] + [i-w]
    pmf = float(len(msg) - w)
    pmf = {i: len(lib[i])/pmf for i in lib}
    return pmf


def plugin_entropy(msg: str | list, w: int):
    """
    Estimate entropy of a data message from words of length w using the plugin method

    :param msg: string or list of string serializable data
    :param w: length of subsequence for entropy estimation
    :return: the entropy rate (entropy per bit) and the PMF
    """
    pmf = _word_pmf(msg, w)
    res = -sum(pmf[i]*np.log2(pmf[i]) for i in pmf)/w
    return res, pmf


def lempelziv_lib(msg):
    """
    Get the non-redundant strings dictionary based on the Lempel-Ziv (LZ) compression algorithm.
    A larger dictionary implies a more complex message which contains more information and
    hence higher entropy than a more regular (predictable) sequence.

    :param msg: input message
    :return: the LZ dictionary
    """
    msg = _stringify(msg)
    i, lib = 1, set(msg[0])
    while i < len(msg):
        for j in range(i, len(msg)):
            sub_str = msg[i:j+1]
            if sub_str not in lib:
                lib.add(sub_str)
                break
        i = j + 1
    return lib


def _longest_match(msg, i, n):
    """
    Find the longest string in the n-size window before i that matches a string in the n-size window from i.
    Set n = i for expanding window.
    """
    match = ''
    for length in range(n):
        msg_right = msg[i:i+length+1]
        for j in range(i-n, i):
            msg_left = msg[j:j+length+1]
            if msg_right == msg_left:
                match = msg_right
                break
    return len(match) + 1, match


def konto_entropy(msg: str | list[str], window=None):
    """
    Kontoyiannis's LZ entropy estimate

    :param msg: the data message
    :param window: set to None for expanding window
    :return: dict of entropy estimate and related statistics
    """
    out = {'n_points': 0, 'sum_entropy': 0, 'nr_str': []}
    msg = _stringify(msg)
    if window is None:
        points = range(1, len(msg)//2 + 1)
    else:
        window = min(window, len(msg)//2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, sub_msg = _longest_match(msg, i, i)
            out['sum_entropy'] += np.log2(i+1)/l
        else:
            l, sub_msg = _longest_match(msg, i, window)
            out['sum_entropy'] += np.log2(window+1)/l
        out['nr_str'].append(sub_msg)

    out['n_points'] = len(points)
    out['entropy'] = out['sum_entropy']/out['n_points']
    out['redundancy'] = 1 - out['entropy']/np.log2(len(msg))
    return out


# --- Encoding ---

def binary_encode(series, threshold):
    """
    Encode a numeric series as a string of 0 and 1
    :param: series: input series
    :param threshold: binary threshold
    :return: encoded string
    """
    res = np.zeros_like(series, dtype=int)
    res[series > threshold] = 1
    return ''.join(map(str, res))


class QuantileEncoder(TransformerMixin, BaseEstimator):
    """
    Encode numeric series based on their quantiles
    """

    FULL_ALPHABET = string.digits + string.ascii_letters

    def __init__(self, n_quantiles=10):
        assert n_quantiles <= len(self.FULL_ALPHABET)
        self.n_quantiles_ = n_quantiles
        self.alphabet_ = np.array(self.FULL_ALPHABET[:n_quantiles] + '?')
        self.quantile_probs_ = np.linspace(0, 1, n_quantiles)
        self.quantiles_ = None  # type: pd.Series

    def fit(self, series):
        quantiles = np.quantile(series, self.quantile_probs_)
        self.quantiles_ = pd.Series(quantiles)
        return self

    def transform(self, series):
        if self.quantiles_ is None:
            raise RuntimeError('Fit must be called first before transform')
        abc_indices = self.quantiles_.searchsorted(series)
        res = self.alphabet_[abc_indices]
        return ''.join(res)


def sigma_encode(series, sigma):
    """
    Encode a numeric series using equally space intervals
    :param series: input series
    :param sigma: spacing between codes
    :return: encoded string
    """
    alphabet = string.digits + string.ascii_letters + '?'
    lo = np.min(series)
    abc_indices = (series - lo) // sigma
    abc_indices = np.minimum(abc_indices, len(alphabet) - 1)
    res = series[abc_indices]
    return ''.join(res)

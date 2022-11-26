from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from data_structures.constants import OptionCol, StatsCol
from utils.stats import KDERv


def implied_risk_free_rate(calls: pd.DataFrame, puts: pd.DataFrame, s0: float, t: float, d: float = 0.):
    """
    Get implied risk-free interest rate from call-put parity

    :param calls: DataFrame of call options Strike and Price
    :param puts: DataFrame of put options Strike and Price
    :param s0: current underlying asset price
    :param t: time of expiry of the options in years
    :param d: dividend amount from now until expiry
    :return: median risk-free rate, series of implied risk-free rate from all call-put pairs
    """
    calls = calls[[OptionCol.STRIKE, OptionCol.PRICE]]
    puts = puts[[OptionCol.STRIKE, OptionCol.PRICE]]
    callput = calls.merge(puts, on=OptionCol.STRIKE, how='outer', suffixes=['_c', '_p']).dropna()
    rf_rates = (callput[f'{OptionCol.PRICE}_p'] + s0 - callput[f'{OptionCol.PRICE}_c'] - d)
    rf_rates = -np.log(rf_rates/callput[OptionCol.STRIKE])/t
    rf_rates.index = callput[OptionCol.STRIKE]
    return rf_rates.median(), rf_rates


def implied_underlying_discrete_pdf(
        options: pd.DataFrame, t: float, r: float, n_interpolate: int = None, smooth_width: float = None
):
    """
    Get the implied discrete pdf of the underlying asset price from its option prices

    :param options: DataFrame of Strike and Price
    :param t: time to expiry in years
    :param r: risk-free interest rate
    :param n_interpolate: number of strikes to be interpolated in the pdf
    :param smooth_width: Gaussian filter width for smoothing
    :return: DataFrame estimated PDF for each interpolated strike price
    """
    price_curve = CubicSpline(x=options[OptionCol.STRIKE], y=options[OptionCol.PRICE])
    deriv2 = price_curve.derivative(2)

    if n_interpolate is None:
        n_interpolate = options.shape[0]*3
    k = np.linspace(options[OptionCol.STRIKE].min(), options[OptionCol.STRIKE].max(), num=n_interpolate)
    pdf = pd.DataFrame({OptionCol.STRIKE: k, StatsCol.PDF: np.exp(r*t)*deriv2(k)})
    pdf = pdf[pdf[StatsCol.PDF] >= 0]

    if smooth_width is not None:
        smooth_width = smooth_width
        pdf[StatsCol.PDF] = gaussian_filter1d(pdf[StatsCol.PDF], smooth_width)

    return pdf


def implied_underlying_distribution(
        calls: pd.DataFrame, puts: pd.DataFrame, t: float, r: float = None,
        s0: float = None, d: float = 0., n_interpolate: int = None, smooth_width: float = None,
        kde_samples=10000, random_state=None
):
    """
    Get the implied distribution of the underlying asset price from its option prices

    :param calls: DataFrame of call option Strike and Price
    :param puts: DataFrame of call option Strike and Price
    :param t: time to expiry in years
    :param r: risk-free interest rate. If None, estimate from call-put parity
    :param s0: current underlying asset price. Required if risk-free rate is None
    :param d: dividend from underlying asset from now til options expiry
    :param n_interpolate: number of interpolated strike prices for estimating discrete pdf.
        If None, default to 3*number_of_strikes
    :param smooth_width: Gaussian filter width for smoothing the estimated pdf
    :param kde_samples: number of generated samples to KDE estimation of the discrete pdf
    :param random_state: numpy's random seed for generatin the KDE samples
    :return: scipy's rv_continuous instance, pdf function
    """
    if r is None:
        if s0 is None:
            raise ValueError('At least r or s0 must not be None')
        r, _ = implied_risk_free_rate(calls, puts, s0, t, d=d)

    pdf_calls = implied_underlying_discrete_pdf(calls, t, r, n_interpolate=n_interpolate, smooth_width=smooth_width)
    pdf_puts = implied_underlying_discrete_pdf(puts, t, r, n_interpolate=n_interpolate, smooth_width=smooth_width)
    pdf = pd.concat([pdf_calls, pdf_puts], axis=0)

    if random_state is not None:
        np.random.seed(random_state)
    price_samples = np.random.choice(
        pdf[OptionCol.STRIKE],
        p=pdf[StatsCol.PDF]/pdf[StatsCol.PDF].sum(),
        size=kde_samples
    )
    pdf_fn = gaussian_kde(price_samples)
    pdf_rv = KDERv(pdf_fn)
    return pdf_rv, pdf_fn

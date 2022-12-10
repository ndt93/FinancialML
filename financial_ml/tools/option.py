import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from financial_ml.data_structures.constants import OptionCol, StatsCol, OptionType
from financial_ml.utils.stats import KDERv


def implied_risk_free_rate(calls: pd.DataFrame, puts: pd.DataFrame, s0: float, t: float, d: float = 0.):
    """
    Get thee implied continuously-compounded risk-free interest rate from call-put parity

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
    if callput.shape == 0:
        return None, None
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
    :param r: continuously-compounded risk-free interest rate
    :param n_interpolate: number of strikes to be interpolated in the pdf
    :param smooth_width: Gaussian filter width for smoothing
    :return: DataFrame estimated PDF for each interpolated strike price
    """
    if options.shape[0] == 0:
        return pd.DataFrame(columns=[OptionCol.STRIKE, StatsCol.PDF], dtype=float)
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
    :param r: continuously-compounded risk-free interest rate. If None, estimate from call-put parity
    :param s0: current underlying asset price.
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
            raise ValueError('At least s0 is required to estimate r')
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
    pdf_rv = KDERv(pdf_fn, a=0, xtol=1e-6)
    return {'rv': pdf_rv, 'pdf': pdf_fn, 'r': r}


# Black-Scholes-Merton EUR Option Pricing
def _bsm_d1(s0, k, r, sigma, t, div_yield=0):
    return (np.log(s0/k) + (r - div_yield + sigma**2/2)*t)/(sigma * np.sqrt(t))


def _bsm_d2(s0, k, r, sigma, t, div_yield=0):
    return _bsm_d1(s0, k, r, sigma, t, div_yield=div_yield) - sigma * np.sqrt(t)


def bsm_option_price(option_type: OptionType, s0, k, r, sigma, T, divs=None, div_yield=0, is_futures=False):
    """
    European option pricing using Black-Scholes-Merton model

    :param option_type:
    :param s0: current underlying instrument price
    :param k: strike price
    :param r: risk-free interest rate
    :param sigma: volatility
    :param T: time to maturity in years
    :param divs: list of (dividend amount, time to ex-dividend date in years)
    :param div_yield: continuous dividend yield
    :param is_futures: if pricing a futures option
    """
    assert divs is None or div_yield == 0
    if divs is not None:
        divs_present = sum([v*np.exp(-r*e) for v, e in divs])
        s0 -= divs_present
    if is_futures:
        div_yield = r

    d1 = _bsm_d1(s0, k, r, sigma, T, div_yield=div_yield)
    d2 = _bsm_d2(s0, k, r, sigma, T, div_yield=div_yield)
    if option_type == OptionType.CALL:
        call = s0 * np.exp(-div_yield*T) * norm.cdf(d1) - k * np.exp(-r*T) * norm.cdf(d2)
        return call
    elif option_type == OptionType.PUT:
        put = -s0 * np.exp(-div_yield*T) * norm.cdf(-d1) + k * np.exp(-r * T) * norm.cdf(-d2)
        return put
    else:
        raise NotImplementedError('Option type', option_type.value)


def implied_volatility(
        observed_price, pricing_fn=bsm_option_price,
        lower_bound=0.01, upper_bound=1.0, maxiter=100,
        **kwargs
):
    """
    Get implied volatility for an observed option price
    :param observed_price:
    :param pricing_fn: the option pricing function
    :param lower_bound: low bracket of interval for root finding
    :param upper_bound: high bracket of interval for root finding
    :param maxiter: number of iterations for root finding
    :param kwargs: arguments to the option pricing function
    """
    f = lambda sigma: pricing_fn(sigma=sigma, **kwargs) - observed_price
    x0 = brentq(f, a=lower_bound, b=upper_bound, maxiter=maxiter)
    return x0

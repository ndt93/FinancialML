import numpy as np
from scipy.stats import norm

from financial_ml.data_structures.constants import OptionType
from financial_ml.tools.option import implied_asset_price


def _d1(v0, k, r, sigma, t, alpha, beta, mkt_ex_ret):
    return (np.log(v0/k) + (r + alpha + sigma**2/2) * t + beta * mkt_ex_ret) / (sigma * np.sqrt(t))


def _d2(v0, k, r, sigma, t, alpha, beta, mkt_ex_ret):
    return _d1(v0, k, r, sigma, t, alpha, beta, mkt_ex_ret) - sigma*np.sqrt(t)


def _cond_equity_value(v0, k, r, sigma, T, alpha, beta, mkt_ex_ret):
    d1 = _d1(v0, k, r, sigma, T, alpha, beta, mkt_ex_ret)
    d2 = _d2(v0, k, r, sigma, T, alpha, beta, mkt_ex_ret)
    call = v0 * norm.cdf(d1) - k * np.exp(-r*T) * norm.cdf(d2)
    return call


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


def compute_structural_asset_values(
        equity_values: list,
        debt_values: list,
        debt_maturities: list,
        data_interval: float,
        risk_free_rates: list,
        init_asset_volatility: float = None,
        tolerance=1e-6,
        max_iters=100
):
    """
    Compute a series of a company's asset values using Merton's structural model
    where the company's the equity value is treated as call option with the book value of debt
    as the strike price and expires at the debt's maturity.

    :param equity_values:
    :param debt_values:
    :param debt_maturities:
    :param data_interval:
    :param risk_free_rates:
    :param init_asset_volatility:
    :param tolerance:
    :param max_iters:
    """
    if max_iters <= 0:
        raise Exception('Unable to converge after max_iters')

    if init_asset_volatility is None:
        init_asset_volatility = np.std(np.diff(np.log(equity_values))) / np.sqrt(data_interval)

    asset_values = [
        implied_asset_price(
            option_price=equity,
            x0=(equity + debt),
            option_type=OptionType.CALL,
            sigma=init_asset_volatility,
            k=max(1, debt),
            r=r,
            T=T
        )
        for equity, debt, T, r in zip(equity_values, debt_values, debt_maturities, risk_free_rates)
    ]
    asset_volatility = np.std(np.diff(np.log(asset_values))) / np.sqrt(data_interval)
    if abs(asset_volatility - init_asset_volatility) < tolerance:
        return asset_values, asset_volatility

    return compute_structural_asset_values(
        equity_values=equity_values,
        debt_values=debt_values,
        debt_maturities=debt_maturities,
        data_interval=data_interval,
        risk_free_rates=risk_free_rates,
        init_asset_volatility=asset_volatility,
        tolerance=tolerance,
        max_iters=max_iters - 1
    )

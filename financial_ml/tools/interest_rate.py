from scipy.interpolate import CubicSpline
import numpy as np


def interpolate_yield_curve(term_structure: [(float, float)]):
    """
    Interpolate a continuous interest rate yield curve from discrete term structure data
    """
    x = [i[0] for i in term_structure]
    y = [i[1] for i in term_structure]
    curve = CubicSpline(x, y)
    return curve


def get_implied_forward_rate(s0: float, f: float, t: float):
    """
    Get implied forward rate from spot and forward/futures price of an instrument
    e.g (r - dividend_yield) for equity/stock index or (r - rf) for currency
    :param s0: spot price
    :param f: forward or futures price
    :param t: time to expiry of the forward/futures contract in years
    :return: annualized continuously compounded implied forward rate
    """
    return np.log(f/s0)/t

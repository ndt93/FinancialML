import numpy as np


def get_strategy_sharpe_ratio(p, freq, profit, loss):
    """
    Get the expected Sharpe ratio for a strategy with a binomial probability of outcomes

    :param p: probability/precision of the strategy
    :param freq: number of positions produced by the strategy a year
    :param profit: profit threshold for each position
    :param loss: loss threshold for each position. should be a negative number
    :return: the expected Sharpe ratio
    """
    numer = (profit - loss)*p + loss
    denom = (profit - loss)*(p*(1 - p))**.5
    return numer/denom*freq**.5


def get_strategy_required_precision(target_sr, freq, profit, loss):
    """
    Get the required precision for a target Sharpe ratio

    :param target_sr: the target Sharpe ratio
    :param freq: number of positions produced by the strategy a year
    :param profit: profit threshold for each position
    :param loss: loss threshold for each position. should be a negative number
    :return: the expected Sharpe ratio
    """
    a = (freq + target_sr**2)*(profit - loss)**2
    b = (2*freq*loss - target_sr**2*(profit - loss))*(profit - loss)
    c = freq*loss**2
    p = (-b + (b**2 - 4*a*c)**.5)/(2.*a)
    return p


def get_strategy_required_freq(target_sr, p, profit, loss):
    """
    Get the required bets frequency for a target Sharpe ratio

    :param target_sr: the target Sharpe ratio
    :param p: probability/precision of the strategy
    :param profit: profit threshold for each position
    :param loss: loss threshold for each position. should be a negative number
    :return: the expected Sharpe ratio
    """
    freq = (target_sr*(profit-loss))**2*p*(1-p)/((profit-loss)*p + loss)**2
    if not np.isclose(get_strategy_sharpe_ratio(p, freq, profit, loss), target_sr):
        return None
    return freq

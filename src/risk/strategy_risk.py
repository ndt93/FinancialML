import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.simulation import generate_mixed_gaussians


def strategy_sharpe_ratio(p, freq, profit, loss):
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


def strategy_required_precision(target_sr, freq, profit, loss):
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


def strategy_required_freq(target_sr, p, profit, loss):
    """
    Get the required bets frequency for a target Sharpe ratio

    :param target_sr: the target Sharpe ratio
    :param p: probability/precision of the strategy
    :param profit: profit threshold for each position
    :param loss: loss threshold for each position. should be a negative number
    :return: the expected Sharpe ratio
    """
    freq = (target_sr*(profit-loss))**2*p*(1-p)/((profit-loss)*p + loss)**2
    if not np.isclose(strategy_sharpe_ratio(p, freq, profit, loss), target_sr):
        return None
    return freq


def strategy_probability_of_failure(returns: pd.Series | np.ndarray, freq: int, target_sr: float):
    """
    Calculate the probability that a strategy fail to achieve the target Sharpe ratio in the long run

    :param returns:
    :param freq:
    :param target_sr:
    :return:
    """
    pos_ret = returns[returns > 0]
    mean_pos_ret = pos_ret.mean()
    mean_neg_ret = returns[returns <= 0].mean()
    p = pos_ret.shape[0]/float(returns.shape[0])
    print(mean_pos_ret, mean_neg_ret, p)
    p_thresh = strategy_required_precision(target_sr, freq, mean_pos_ret, mean_neg_ret)
    risk = norm.cdf(p_thresh, p, p*(1-p))  # Use bootstrap if number of samples is small
    return risk


def simulate_strategy_probability_of_failure(mu1, mu2, sigma1, sigma2, prob1, n_samples, freq, target_sr):
    """
    A reference implementation for estimating a strategy probability of failure by drawing returns
    from a gaussian mixture distribution

    :param mu1: mean of the 1st Gaussian in the return distribution
    :param mu2: mean of the 2nd Gaussian in the return distribution
    :param sigma1: std of the 1st Gaussian in the return distribution
    :param sigma2: std of the 2nd Gaussian in the return distribution
    :param prob1: probability of drawing from the 1st gaussian
    :param n_samples: number of samples
    :param freq: number of positions taken by the strategy a year
    :param target_sr: the target Sharpe ratio
    :return: strategy risk/probability of failure
    """
    returns = generate_mixed_gaussians(mu1, mu2, sigma1, sigma2, prob1, n_samples)
    prob_failure = strategy_probability_of_failure(returns, freq, target_sr)
    return prob_failure

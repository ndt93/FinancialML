import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm

from data_structures.constants import PortfolioCol


class Metrics:
    NEG_LOG_LOSS = 'neg_log_loss'
    ACCURACY = 'accuracy'


def get_position_timings(positions: pd.Series):
    """
    Get timings of position flattening or flipping that indicate a new bet

    :param positions: series of position sizes with DateTimeIndex
    :return: series of timestamps of bet timings
    """
    no_pos_times = positions[positions == 0].index
    prev_pos = positions.shift(1)
    prev_pos = prev_pos[prev_pos != 0].index
    res = no_pos_times.intersection(prev_pos)
    flips = positions.iloc[1:]*positions.iloc[:-1].values
    res = res.union(flips[flips < 0].index).sort_values()
    return res


def compute_holding_period(positions: pd.Series):
    """
    Estimate the average holding period of a position in a strategy

    :param positions: series of position sizes with DateTimeIndex
    :return: average holding period in days
    """
    hp = pd.DataFrame(columns=['dT', 'w'])
    t_entry = 0.
    p_diff = positions.diff()
    t_diff = (positions.index - positions.index[0])/pd.Timedelta(1, 'D')
    for i in range(1, positions.shape[0]):
        if p_diff.iloc[i] * positions.iloc[i-1] >= 0:  # Position increased or unchanged
            if positions.iloc[i] != 0:
                t_entry = (t_entry*positions.iloc[i-1] + t_diff[i]*p_diff.iloc[i])/positions.iloc[i]
        else:  # Position decreased
            if positions.iloc[i]*positions.iloc[i-1] < 0:  # Position flip
                hp.loc[positions.index[i], ['dT', 'w']] = (t_diff[i] - t_entry, abs(positions.iloc[i-1]))
                t_entry = t_diff[i]  # Reset entry time
            else:
                hp.loc[positions.index[i], ['dT', 'w']] = (t_diff[i] - t_entry, abs(p_diff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT']*hp['w']).sum()/hp['w'].sum()
        return hp
    return None


def compute_time_weighted_return(porfolio_values: pd.DataFrame):
    """
    Compute the time weighted rate of return as defined by GIPS.

    :param porfolio_values: a DataFrame of [PortfolioCol.BEGIN_VALUE, PortfolioCol.CASHFLOW]
        at the time of each external cashflows. External cashflow are cash or assets that enter or exit the portfolio.
        Dividends and interests are not considered external cash flows.
    :return: time weighted rate of return
    """
    begin_balance = (porfolio_values[PortfolioCol.BEGIN_VALUE] + porfolio_values[PortfolioCol.CASHFLOW]).shift(1)
    pnl = porfolio_values[PortfolioCol.BEGIN_VALUE] - begin_balance
    rets = pnl / begin_balance
    return (1 + rets).product() - 1


def compute_hhi_returns_concentration(returns):
    """
    Based on the Herfindahl-Hirschman index, find if returns are concentrated over on a few positions or
    spread over many positions. The index is a number from 0 to 1 for low to high concentration.
    Must provide at least 3 returns.

    :param returns: a series of return from each position
    :return: (positive returns concentration, negative returns concentration)
    """
    def _get_hhi(rets):
        weights = rets/rets.sum()
        hhi = (weights**2).sum()
        con_hhi = (hhi - 1/rets.shape[0])/(1 - 1/rets.shape[0])
        return con_hhi

    if returns.shape[0] <= 2:
        return np.nan, np.nan
    pos_rets = returns[returns >= 0]
    neg_rets = returns[returns < 0]
    pos_hhi = np.nan if pos_rets.shape[0] == 0 else _get_hhi(pos_rets)
    neg_hhi = np.nan if neg_rets.shape[0] == 0 else _get_hhi(neg_rets)
    return pos_hhi, neg_hhi


def compute_dd_and_tuw(returns: pd.Series, dollars=False):
    """
    Compute the draw down and time underwater metrics from a series of returns or dollar values.
    The last data point will be ignored.

    :param returns: series of returns or dollar values
    :param dollars: set to True if returns are in dollar values
    :return: drawdown, time underwater in days
    """
    df0 = returns.to_frame('pnl')
    df0['hwm'] = returns.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index
    df1 = df1[df1['hwm'] > df1['min']]
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1. - df1['min']/df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1])/pd.Timedelta(1, 'D')).values
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


def sharpe_ratio(returns: np.ndarray, axis=None):
    """
    Calculate the classic Sharpe ratio = mean(returns)/std(returns)

    :param returns: array of returns or other performance metrics e.g. PnL
    :param axis: calculate sharpe ration along this axis
    :return: Sharpe ratio
    """
    return np.mean(returns, axis=axis) / np.std(returns, axis=axis)


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark=0, axis=None):
    """
    Calculate the probabilistic Sharpe ratio i.e. the probability that the estimated Sharpe ratio
    is higher than the benchmark Sharpe ratio, considering the shape of the returns distribution.

    :param returns: a series of returns
    :param benchmark: the benchmark Sharpe ratio.
    :param axis: calculate along this axis
    :return: the probabilistic Sharpe ratio
    """
    obs_sharpe = sharpe_ratio(returns, axis=axis)
    num_obs = returns.shape[axis] if axis else returns.size
    s = skew(returns, axis=axis)
    k = kurtosis(returns, axis=axis)

    numer = (obs_sharpe - benchmark) * np.sqrt(num_obs - 1)
    denom = np.sqrt(1 - s*obs_sharpe + (k - 1)/4*obs_sharpe**2)
    psr = norm.cdf(numer/denom)
    return psr


def deflated_sharpe_ratio(returns: np.ndarray, axis=0):
    """
    Calculate the deflated Sharpe ratio, which is the probabilistic Sharpe ratio where the benchmark Sharpe
    ratio is the expected maximum of the estimated Sharpe ratio

    :param returns: series of returns for multiple trials
    :param axis: 0 if each trial's returns occupies 1 column. 1 if it's 1 row per trial
    :return: (deflated Sharpe ratio, benchmark Sharpe ratio)
    """
    sharpe_ratios = sharpe_ratio(returns, axis=axis)
    sr_var = np.var(sharpe_ratios)
    n_trials = sharpe_ratios.shape[0]
    a = (1 - np.euler_gamma)*norm.ppf(1 - 1/n_trials)
    b = np.euler_gamma*norm.ppf(1 - 1/n_trials*np.exp(-1))
    benchmark_sr = np.sqrt(sr_var) * (a + b)
    return probabilistic_sharpe_ratio(returns, benchmark_sr, axis=axis), benchmark_sr


def annualized_sharpe_ratio(sharpe_ratio: np.ndarray, returns_per_year):
    """
    Annualize the given Sharpe ratio assuming returns are IID
    :param sharpe_ratio: observed Sharpe ratio
    :param returns_per_year: average returns observed per year
    :return:
    """
    return sharpe_ratio * np.sqrt(returns_per_year)


def information_ratio(returns, benchmark_returns, axis=None):
    """
    Calculate the information ratio, an equivalent of Sharpe ratio on a portfolio that measure its
    performance relative to a benchmark.
    info ratio = mean(excess_return)/tracking_error

    :param returns: returns series
    :param benchmark_returns: benchmark returns series
    :param axis:
    :return: the information ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_errors = np.std(excess_returns, axis=axis)
    return np.mean(excess_returns, axis=axis)/tracking_errors

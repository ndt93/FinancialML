import pandas as pd
import numpy as np

from data_structures.constants import PortfolioCol


class Metrics:
    NEG_LOG_LOSS = 'neg_log_loss'
    ACCURACY = 'accuracy'


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
    return _get_hhi(pos_rets), _get_hhi(neg_rets)


def compute_dd_and_tuw(returns, dollars=False):
    """
    Compute the draw down and time underwater metrics from a series of return or PnL dollar performance

    :param returns: series of returns or dollar PnL
    :param dollars: set to True if returns are in dollar PnL
    :return: drawdown i.e. maximum loss between 2 consecutive high-watermarks
    """
    df0 = returns.to_frame('pnl')
    df0['hwm'] = returns.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(kee='first').index
    df1 = df1[df1['hwm'] > df1['min']]
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1. - df1['min']/df1['hwm']
    tuw = (df1.index[1:] - df1.index[:-1])/pd.Timedelta(1, 'Y').values
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


def sharpe_ratio(returns_series: np.ndarray, axis=None):
    """
    :param returns_series: array of returns or other performance metrics e.g. PnL
    :param axis: calculate sharpe ration along this axis
    :return: Sharpe ratio
    """
    return np.mean(returns_series, axis=axis)/np.std(returns_series, axis=axis)

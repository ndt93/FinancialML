import numpy as np
import pandas as pd

from data_structures.constants import TickCol, BarCol, QuoteCol


def tick_rule(ticks: pd.DataFrame, b0: float):
    """
    Transform a sequence of ticks into a series of tick rule features b(t).
    b(t) determines a trade's aggressor side, where 1 and -1 represents a buy and sell initiated trade respectively.
    :param ticks: Dataframe of ticks with TickCol.PRICE
    :param b0: initial tick rule value
    :return: array of tick rule values
    """
    price_change = ticks[TickCol.PRICE].diff().values
    if len(price_change) == 0:
        return np.array([])
    tick_directions = np.zeros(len(price_change))
    tick_directions[0] = b0

    for i in range(1, len(price_change)):
        if price_change[i] == 0:
            tick_directions[i] = tick_directions[i - 1]
        else:
            tick_directions[i] = abs(price_change[i])/price_change[i]

    return tick_directions.astype(np.float64)


def roll_effective_spread(ticks: pd.DataFrame):
    """
    Compute the effective spread and true price noises (excluding microstructure noises from crossing bid-ask spread)
    using Roll model. Assuming mid-price series follow a Random Walk with no drift:
        m(t) = m(t-1) + u(t) => dm(t) ~ N[0, var(u(t))]
    The observed prices from sequential trading against the bid-ask spread: p(t) = m(t) + b(t)*c, where
    c is half of the spread and b(t) is the trade's aggressor side (see tick_rule).
    var(dm) and c can then be determined from observed p(t)

    :param ticks: Dataframe of ticks with TickCol.PRICE
    :return: c, var(dm)
    """
    dp = ticks[TickCol.PRICE].diff().values
    dp_var = np.var(dp)
    dp_corr = np.corrcoef(dp[:-1], dp[1:])[0, 1]
    c = np.sqrt(max(0, -dp_corr))
    dm_var = dp_var + 2*dp_corr
    return c, dm_var


def high_low_volatility(bars: pd.DataFrame):
    """
    Volatility estimate based on high-low prices instead of closing prices. See Beckers [1983] and Parkinson [1980].
    :param bars: Dataframe of price bars
    :return: the high-low volatility estimate
    """
    hl_log_rets = np.log(bars[BarCol.HIGH]) - np.log(bars[BarCol.LOW])
    return np.mean(hl_log_rets**2)/(4*np.log(2))


def _get_beta(bars: pd.DataFrame, window: int):
    hl = np.log(bars[BarCol.HIGH]/bars[BarCol.LOW])**2
    betas = hl.rolling(2).sum()
    betas = betas.rolling(window).mean()
    return betas.dropna()


def _get_gamma(bars):
    h = bars[BarCol.HIGH].rolling(2).max()
    l = bars[BarCol.LOW].rolling(2).min()
    gamma = np.log(h/l)**2
    return gamma.dropna()


def _get_alpha(beta, gamma):
    denom = 3 - 2*2**.5
    alpha = ((2*beta)**.5 - beta**.5)/denom - (gamma/denom)**.5
    alpha[alpha < 0] = 0
    return alpha.dropna()


def _corwin_schultz_volatility(beta, gamma):
    """
    The Becker-Parkinson High-Low volatility estimate from Corwin Schultz beta and gamma
    :param beta: beta in Corwin Schultz equation
    :param gamma:
    :return:
    """
    k2 = (8/np.pi)**.5
    denom = 3 - 2*2**.5
    sigma = (2**-.5 - 1)*beta**.5/(k2*denom)
    sigma += (gamma/(k2**2*denom))**.5
    sigma[sigma < 0] = 0
    return sigma


def corwin_schultz_spread(bars, window=1):
    """
    The Corwin and Schultz bid-ask spread and volatility estimates from high and low prices
    :param bars: DataFrame of price bars
    :param window: rolling window size for estimating beta
    :return: (spread, volatility) estimates
    """
    beta = _get_beta(bars, window)
    gamma = _get_gamma(bars)
    alpha = _get_alpha(beta, gamma)
    spread = 2*(np.exp(alpha) - 1)/(1 + np.exp(alpha))
    start_times = pd.Series(bars.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_times], axis=1)
    spread.columns = [QuoteCol.SPREAD, QuoteCol.START_TIME]
    volatility = _corwin_schultz_volatility(beta, gamma)
    return spread, volatility

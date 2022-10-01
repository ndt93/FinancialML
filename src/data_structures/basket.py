import pandas as pd
import numpy as np

from data_structures.constants import BarCol

"""
This module implements several methods to transform price series of complex basket of securities with dynamic
weights re-balancing, irregular dividends, coupons, corporate action, roll over, margin costs, funding costs etc.
as if it's a single series of cash-like product.
"""


def basket_to_etf(
        weights: pd.DataFrame,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        usd_ex_rate: pd.DataFrame,
        volumes: pd.DataFrame,
        dividends: pd.DataFrame,
        reblancing_bars: list
):
    """
    Get value over time of a $1 invested in a basket of securities.
    The result behaves as if it's a single ETF

    :param weights: DataFrame of weights series of each instrument in the portfolio
    :param open_prices: DataFrame of raw open price series. 1 column per instrument
    :param close_prices: DataFrame of raw close price series. 1 column per instrument
    :param usd_ex_rate: DataFrame USD value of 1 point of each instrument
    :param volumes: DataFrame of volume of each instrument. 1 column per instrument
    :param dividends: DataFrame of dividend series. Can be used for coupons or carrying/funding/margin costs.
        1 column per instrument
    :param reblancing_bars: list of bar indices where the basket weights are rebalanced.
        Avoid having the first or last bar in the rebalancing period
    :return: DataFrame of basket close value series starting at $1 and volume of units traded
    """
    num_bars = weights.shape[0]
    num_instr = weights.shape[1]
    sum_abs_weights = weights.abs().sum(axis=1)
    bar_indices = np.arange(0, num_bars)
    is_rebalancing = np.isin(bar_indices, reblancing_bars)

    basket_vals = np.zeros(num_bars)
    basket_vals[0] = 1.0
    value_changes = np.zeros((num_bars, num_instr))
    value_changes[0, :] = 0.0
    holdings = np.zeroes((num_bars, num_instr))
    holdings[0, :] = weights.iloc[0, :]*basket_vals[0] / sum_abs_weights.iloc[0]

    intra_bar_change = close_prices - open_prices
    bar_to_bar_change = np.insert(np.diff(close_prices), 0, 0)
    value_changes = np.where(is_rebalancing, intra_bar_change, bar_to_bar_change)
    net_val_changes = value_changes + dividends.values

    for t in range(1, num_bars):
        basket_val_change = (holdings[t - 1, :] * net_val_changes[t, :] * usd_ex_rate.iloc[t, :].values).sum()
        basket_vals[t] = basket_vals[t-1] + basket_val_change

        if t + 1 < num_bars and t in reblancing_bars:
            instr_val = weights.iloc[t, :]*basket_vals[t] / sum_abs_weights.iloc[t]
            holdings[t, :] = instr_val / (open_prices.iloc[t + 1, :] * usd_ex_rate.iloc[t, :])
        else:
            holdings[t, :] = holdings[t - 1, :]

    basket_volume = np.min(volumes.values / np.abs(holdings), axis=1)
    return pd.DataFrame({BarCol.CLOSE: basket_vals, BarCol.VOLUME: basket_volume}, index=weights.index)

import pandas as pd
import numpy as np

from financial_ml.data_structures.constants import BarCol

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
        rebalancing_bars: list
):
    """
    Get value over time of a $1 invested in a basket of securities. The result behaves as if it's a single ETF.
    Assume rebalancing is done by selling the basket to cash at the close of the rebalancing bars and bought in
    new proportions at the open of the next bar.

    :param weights: DataFrame of weights series of each instrument in the portfolio
    :param open_prices: DataFrame of raw open price series. 1 column per instrument
    :param close_prices: DataFrame of raw close price series. 1 column per instrument
    :param usd_ex_rate: DataFrame USD value of 1 point of each instrument
    :param volumes: DataFrame of volume of each instrument. 1 column per instrument
    :param dividends: DataFrame of dividend series. Can be used for coupons or carrying/funding/margin costs.
        1 column per instrument
    :param rebalancing_bars: list of bar indices where the basket weights are rebalanced.
        Avoid having the first or last bar in the rebalancing period
    :return: DataFrame of basket close value series starting at $1 and volume of units traded
    """
    num_bars = weights.shape[0]
    num_instr = weights.shape[1]
    sum_abs_weights = weights.abs().sum(axis=1)
    bar_indices = np.arange(0, num_bars)
    is_after_rebalancing = np.tile(np.isin(bar_indices, np.array(rebalancing_bars) + 1).reshape(-1, 1), num_instr)

    basket_vals = np.zeros(num_bars)
    basket_vals[0] = 1.0
    holdings = np.zeros((num_bars, num_instr))
    init_intr_vals = (weights.iloc[0, :]*basket_vals[0] / sum_abs_weights.iloc[0])
    holdings[0, :] = init_intr_vals / (close_prices.iloc[0, :] * usd_ex_rate.iloc[0, :])

    intra_bar_change = (close_prices - open_prices).values
    bar_to_bar_change = close_prices.diff().fillna(0.0).values
    value_changes = np.where(is_after_rebalancing, intra_bar_change, bar_to_bar_change)
    net_val_changes = value_changes + dividends.values

    for t in range(1, num_bars):
        instr_val_changes = holdings[t - 1, :] * net_val_changes[t, :] * usd_ex_rate.iloc[t, :].values
        basket_vals[t] = basket_vals[t - 1] + instr_val_changes.sum()

        if t + 1 < num_bars and t in rebalancing_bars:
            instr_vals = weights.iloc[t, :]*basket_vals[t] / sum_abs_weights.iloc[t]
            holdings[t, :] = instr_vals / (open_prices.iloc[t + 1, :] * usd_ex_rate.iloc[t, :])
        else:
            holdings[t, :] = holdings[t - 1, :]

    basket_volume = np.min(volumes.values / np.abs(holdings), axis=1)
    return pd.DataFrame({BarCol.CLOSE: basket_vals, BarCol.VOLUME: basket_volume}, index=weights.index)


def _compute_roll_gaps(bars: pd.DataFrame, contracts: pd.Series, match_end):
    """
    :param bars: DataFrame with [BarCol.CLOSE, BarCol.OPEN] and DateTimeIndex
    :param match_end: Roll backward (end rolled series = end raw series) if True
        or forward (start rolled series = start raw series) if False
    :return: Series of adjustment gaps with same index as bars
    """
    roll_dates = contracts.drop_duplicates(keep='first').index[1:]
    gaps = bars[BarCol.CLOSE]*0
    bar_iloc = list(bars.index)
    prior_roll_idx = [bar_iloc.index(rd) - 1 for rd in roll_dates]
    gaps.loc[roll_dates] = bars[BarCol.OPEN].loc[roll_dates] - bars[BarCol.CLOSE].iloc[prior_roll_idx].values
    gaps = gaps.cumsum()
    if match_end:
        gaps -= gaps.iloc[-2]
    return gaps


def adjust_rolled_series(
        bars: pd.DataFrame, contracts: pd.Series, adjust_columns: list, match_end=True, non_negative=False
):
    """
    Adjust the prices of multiple sequential contracts e.g. futures that roll over each other so
    that they behave like a single homogenous cash-like product. Used to calculate PnL and portfolio MtM values

    :param bars: DataFrame with [BarCol.CLOSE, BarCol.OPEN] and DateTimeIndex
    :param contracts: Series of contract name associated with each bar and same index as bars
        Its value should change at every roll date
    :param adjust_columns: price columns to be adjusted
    :param match_end: Roll backward (end rolled series = end raw series) if True
        or forward (start rolled series = start raw series) if False
    :param non_negative: return a non-negative rolled price as an additional columns
    :return: same as bars with adjusted BarCol.CLOSE and BarCol.VWAP
    """
    gaps = _compute_roll_gaps(bars, contracts, match_end)
    res = bars.copy()
    for col in adjust_columns:
        res[col] -= gaps
    if non_negative:
        returns = res[BarCol.CLOSE].diff() / bars[BarCol.CLOSE].shift(1)
        returns.iloc[0] = 0
        res[BarCol.RET_PRICES] = (1 + returns).cumprod()
    return res

import pandas as pd
import pytest
import numpy as np

from data_structures.constants import BarCol, ContractCol
from data_structures import basket

futures_df = pd.DataFrame({
    BarCol.TIMESTAMP: pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 11)]),
    BarCol.OPEN: [8.3, 11.71, 14.68, 8.18, 9.05, 8.62, 7.6, 6.44, 9.42, 10.7],
    BarCol.CLOSE: [7.56, 10.33, 7.14, 5.09, 11.3, 9.95, 5.53, 9.67, 12., 11.0],
    ContractCol.CONTRACT: ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C']
}).set_index(BarCol.TIMESTAMP)


@pytest.fixture
def futures_sample():
    return futures_df.copy()


def test_forward_roll_adj(futures_sample):
    rolled_df = basket.adjust_rolled_series(
        futures_sample,
        futures_sample[ContractCol.CONTRACT],
        [BarCol.CLOSE],
        match_end=False
    )
    adj_1 = 8.18 - 7.14
    adj_2 = adj_1 + (6.44 - 5.53)
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[:3].values,
        futures_sample[BarCol.CLOSE].iloc[:3].values,
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[3:7].values,
        futures_sample[BarCol.CLOSE].iloc[3:7].values - adj_1,
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[7:].values,
        futures_sample[BarCol.CLOSE].iloc[7:].values - adj_2,
        decimal=2
    )


def test_backward_roll_adj(futures_sample):
    rolled_df = basket.adjust_rolled_series(
        futures_sample,
        futures_sample[ContractCol.CONTRACT],
        [BarCol.CLOSE],
        match_end=True
    )
    adj_1 = 6.44 - 5.53
    adj_2 = adj_1 + (8.18 - 7.14)
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[7:].values,
        futures_sample[BarCol.CLOSE].iloc[7:].values,
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[3:7].values,
        futures_sample[BarCol.CLOSE].iloc[3:7].values + adj_1,
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.CLOSE].iloc[:3].values,
        futures_sample[BarCol.CLOSE].iloc[:3].values + adj_2,
        decimal=2
    )


def test_non_negative_adj(futures_sample):
    rolled_df = basket.adjust_rolled_series(
        futures_sample,
        futures_sample[ContractCol.CONTRACT],
        [BarCol.CLOSE],
        match_end=True,
        non_negative=True
    )
    np.testing.assert_array_almost_equal(
        rolled_df[BarCol.RET_PRICES].values,
        [1., 1.37, 0.94, 0.54, 1.19, 1.05, 0.58, 0.92, 1.14, 1.05],
        decimal=2
    )


time_index = pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 11)])
instr_A_df = pd.DataFrame({
    BarCol.TIMESTAMP: time_index,
    BarCol.OPEN: [8.3, 11.71, 14.68, 8.18, 9.05, 8.62, 7.6, 6.44, 9.42, 10.7],
    BarCol.CLOSE: [7.56, 10.33, 7.14, 5.09, 11.3, 9.95, 5.53, 9.67, 12., 11.0],
    BarCol.VOLUME: [10]*10,
    BarCol.DIVIDEND: [0.0]*10
}, index=time_index)
instr_B_df = pd.DataFrame({
    BarCol.OPEN: [101.83, 101.08, 92.36, 98.98, 100.27, 100.95, 98.73, 98.22, 94.4, 100.68],
    BarCol.CLOSE: [113.42, 101.74, 100.31, 99.23, 97.8, 103.01, 99.18, 98.53, 97.57, 105.93],
    BarCol.VOLUME: [15]*10,
    BarCol.DIVIDEND: [0.0]*10
}, index=time_index)
portfolio_target_weights = pd.DataFrame({
    'A': [0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4],
    'B': [0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6]
}, index=time_index)


@pytest.fixture
def basket_sample():
    num_bars = time_index.shape[0]
    weights = portfolio_target_weights.copy()
    open_prices = pd.DataFrame({'A': instr_A_df[BarCol.OPEN], 'B': instr_B_df[BarCol.OPEN]}, index=time_index)
    close_prices = pd.DataFrame({'A': instr_A_df[BarCol.CLOSE], 'B': instr_B_df[BarCol.CLOSE]}, index=time_index)
    usd_ex_rate = pd.DataFrame({'A': [1.0]*num_bars, 'B': [1.0]*num_bars}, index=time_index)
    volumes = pd.DataFrame({'A': instr_A_df[BarCol.VOLUME], 'B': instr_B_df[BarCol.VOLUME]}, index=time_index)
    dividends = pd.DataFrame({'A': instr_A_df[BarCol.DIVIDEND], 'B': instr_B_df[BarCol.DIVIDEND]}, index=time_index)
    return weights, open_prices, close_prices, usd_ex_rate, volumes, dividends


def test_basket_to_etf(basket_sample):
    rebalancing_bars = np.array([3, 7])
    res = basket.basket_to_etf(*basket_sample, rebalancing_bars)
    weights, opens, closes, usd_ex, vol, div = basket_sample

    a_rets = np.insert(closes['A'].values[1:]/closes['A'].values[:-1], 0, 1.0)
    b_rets = np.insert(closes['B'].values[1:]/closes['B'].values[:-1], 0, 1.0)
    for i in rebalancing_bars + 1:
        a_rets[i] = closes['A'].iloc[i]/opens['A'].iloc[i]
        b_rets[i] = closes['B'].iloc[i]/opens['B'].iloc[i]

    a_cum_rets_1 = a_rets[:4].cumprod()
    b_cum_rets_1 = b_rets[:4].cumprod()
    a_vals_1 = 0.9 * a_cum_rets_1
    b_vals_1 = 0.1 * b_cum_rets_1
    basket_vals_1 = a_vals_1 + b_vals_1

    rebalanced_a_val = basket_vals_1[-1]*0.2
    rebalanced_b_val = basket_vals_1[-1]*0.8
    a_cum_rets_2 = a_rets[4:8].cumprod()
    b_cum_rets_2 = b_rets[4:8].cumprod()
    a_vals_2 = rebalanced_a_val * a_cum_rets_2
    b_vals_2 = rebalanced_b_val * b_cum_rets_2
    basket_vals_2 = a_vals_2 + b_vals_2

    rebalanced_a_val = basket_vals_2[-1]*0.4
    rebalanced_b_val = basket_vals_2[-1]*0.6
    a_cum_rets_3 = a_rets[8:].cumprod()
    b_cum_rets_3 = b_rets[8:].cumprod()
    a_vals_3 = rebalanced_a_val * a_cum_rets_3
    b_vals_3 = rebalanced_b_val * b_cum_rets_3
    basket_vals_3 = a_vals_3 + b_vals_3

    basket_vals = np.concatenate([basket_vals_1, basket_vals_2, basket_vals_3])
    np.testing.assert_array_almost_equal(basket_vals, res[BarCol.CLOSE].values)

    np.testing.assert_array_almost_equal(res[BarCol.VOLUME].iloc[:3], [84.0]*3, decimal=2)
    np.testing.assert_array_almost_equal(res[BarCol.VOLUME].iloc[3:7], [652.54]*4, decimal=2)
    np.testing.assert_array_almost_equal(res[BarCol.VOLUME].iloc[7:], [339.672]*3, decimal=2)

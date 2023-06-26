import numpy as np

from financial_ml.data_structures.constants import OptionType
from financial_ml.tools.option import implied_asset_price


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
            option_type=OptionType.CALL,
            sigma=init_asset_volatility,
            k=debt,
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

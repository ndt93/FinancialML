from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

from financial_ml.tools.option import implied_asset_price


def _d1(s0, k, r, sigma, t, alpha, beta, mkt_ex_ret):
    return (np.log(s0/k) + (r + alpha + sigma**2/2)*t + beta*mkt_ex_ret) / (sigma*np.sqrt(t))


def _d2(s0, k, r, sigma, t, alpha, beta, mkt_ex_ret):
    return _d1(s0, k, r, sigma, t, alpha, beta, mkt_ex_ret) - sigma*np.sqrt(t)


def _cond_equity_value(s0, k, r, sigma, T, alpha, beta, mkt_ex_ret):
    d1 = _d1(s0, k, r, sigma, T, alpha, beta, mkt_ex_ret)
    d2 = _d2(s0, k, r, sigma, T, alpha, beta, mkt_ex_ret)
    call = s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return call


@dataclass
class FirmParams:

    alpha: float
    beta: float
    sigma: float

    def approx_equal(self, other, tol=1e-6):
        return (
            abs(self.alpha - other.alpha) < tol and
            abs(self.beta - other.beta) < tol and
            abs(self.sigma - other.sigma) < tol
        )


def _regress_firm_params(
        asset_rets: np.array,
        market_rets: np.array,
        risk_free_rates: np.array,
        interval: float
):
    asset_rets = asset_rets - risk_free_rates * interval
    market_rets = market_rets - risk_free_rates * interval
    market_rets = sm.add_constant(market_rets)

    ols = sm.OLS(asset_rets, market_rets)
    res = ols.fit()
    beta = res.params[1]
    const = res.params[0]
    sigma = np.std(res.resid)/np.sqrt(interval)
    alpha = const/interval + sigma**2/2
    return FirmParams(alpha=alpha, beta=beta, sigma=sigma)


class FirmStructuralCreditRisk:

    def __init__(self):
        self.firm_params = None

    def fit(
            self,
            expected_market_ret: float,
            market_rets: np.array,
            equity_values: np.array,
            debt_values: np.array,
            debt_maturities: np.array,
            interval: float,
            risk_free_rates: np.array,
            tol=1e-6,
            maxiter=100
    ):
        """
        Estimate firm's parameters by estimating a series of a firm's asset values and returns
        using Merton's structural model where the company's the equity value is treated as call option
        with the book value of debt as the strike price and expires at the debt's maturity.

        :param expected_market_ret:
        :param market_rets:
        :param equity_values:
        :param debt_values:
        :param debt_maturities:
        :param interval:
        :param risk_free_rates:
        :param tol:
        :param maxiter:
        """
        equity_rets = np.diff(np.log(equity_values))
        exp_mkt_ex_rets = expected_market_ret - risk_free_rates * interval
        self.firm_params = _regress_firm_params(
            asset_rets=equity_rets,
            market_rets=market_rets,
            risk_free_rates=risk_free_rates,
            interval=interval
        )

        iteration = 0
        while iteration < maxiter:
            asset_values = np.array([
                implied_asset_price(
                    option_price=equity,
                    x0=(equity + debt),
                    pricing_fn=_cond_equity_value,
                    k=max(1, debt),
                    r=r,
                    sigma=self.firm_params.sigma,
                    T=T,
                    alpha=self.firm_params.alpha,
                    beta=self.firm_params.beta,
                    mkt_ex_ret=mkt_ex_ret * T
                )
                for equity, debt, T, r, mkt_ex_ret in
                zip(equity_values, debt_values, debt_maturities, risk_free_rates, exp_mkt_ex_rets)
            ])

            deltas = norm.cdf(_d1(
                s0=asset_values,
                k=np.maximum(1, debt_values[:-1]),
                r=risk_free_rates,
                sigma=self.firm_params.sigma,
                t=debt_maturities[:-1],
                alpha=self.firm_params.alpha,
                beta=self.firm_params.beta,
                mkt_ex_ret=exp_mkt_ex_rets * debt_maturities[:-1]
            ))
            asset_rets = equity_rets / (deltas * asset_values / equity_values[:-1])

            new_firm_params = _regress_firm_params(
                asset_rets=asset_rets,
                market_rets=market_rets,
                risk_free_rates=risk_free_rates,
                interval=interval
            )
            if self.firm_params.approx_equal(new_firm_params, tol=tol):
                self.firm_params = new_firm_params
                return asset_rets, asset_values

            self.firm_params = new_firm_params
            iteration += 1

        raise Exception(f'Failed to converge after {maxiter} iterations')

    def predict_default(self, equity_value, debt_value, debt_maturity, risk_free_rate, market_ret):
        debt_value = max(1, debt_value)
        mkt_ex_ret = market_ret - risk_free_rate*debt_maturity
        asset_value = implied_asset_price(
            option_price=equity_value,
            x0=(equity_value + debt_value),
            pricing_fn=_cond_equity_value,
            k=debt_value,
            r=risk_free_rate,
            sigma=self.firm_params.sigma,
            T=debt_maturity,
            alpha=self.firm_params.alpha,
            beta=self.firm_params.beta,
            mkt_ex_ret=mkt_ex_ret
        )
        return 1 - norm.cdf(_d2(
            s0=asset_value,
            k=debt_value,
            r=risk_free_rate,
            sigma=self.firm_params.sigma,
            t=debt_maturity,
            alpha=self.firm_params.alpha,
            beta=self.firm_params.beta,
            mkt_ex_ret=mkt_ex_ret
        ))

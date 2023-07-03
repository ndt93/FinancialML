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

    def __init__(self, firm_params=None):
        self.firm_params = firm_params

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
        Compute firm's parameters by estimating a series of a firm's asset values and returns
        using Merton's structural model where the company's the equity value is treated as call option
        with the book value of debt as the strike price and expires at the debt's maturity.

        :param expected_market_ret: Annualized expected market log-return for equity valuation
        :param market_rets: Realized market log-returns over each data interval.
            Should include both broad equity and corporate bonds return.
        :param equity_values: Firm's market equity value i.e. market cap at the start of each interval
        :param debt_values: Firm's last known debt value at the start of each interval.
            E.g. DLC + 1/2*DLTT from firm's balance sheet
        :param debt_maturities: Maturity of firm's debts at start of each interval
        :param interval: Time duration in years for the returns and values series
        :param risk_free_rates: continuously compounded risk-free interest rate e.g. 1Y Treasury yield
        :param tol: error margin for convergence criteria
        :param maxiter: max number of iterations for parameters estimation
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
        """
        Predict firm's default probability at deb maturity conditioned on the realized market return

        :param equity_value: firm's market equity value
        :param debt_value: firm's debts value on balance sheet
        :param debt_maturity: firm's debts maturity
        :param risk_free_rate: continuously compounded risk-free interest rate
        :param market_ret: annualized market return
        """

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


def create_systematic_default_dataset(firm_data, treasury_yields, bonds_index, equities_index):
    dataset = firm_data.merge(treasury_yields, on='date')
    dataset = dataset.rename(columns={'yield': 'treasury_yield'})
    dataset = dataset.merge(bonds_index, on='date').rename(columns={'price': 'bonds_index'})
    dataset = dataset.merge(equities_index, on='date').rename(columns={'price': 'equities_index'})
    dataset = dataset.dropna(subset='treasury_yield')
    return dataset


def create_firm_model(dataset, interval, expected_market_ret, equity_ratio=0.81, return_assets=False):
    equity_mkt_rets = np.diff(np.log(dataset['equities_index']))
    bnd_mkt_rets = np.diff(np.log(dataset['bonds_index']))
    market_rets = equity_ratio * equity_mkt_rets + (1 - equity_ratio) * bnd_mkt_rets

    debt_values = (dataset['shortTermDebt'] + dataset['longTermDebt'] / 2).values
    risk_free_rates = dataset['treasury_yield'].iloc[:-1].values

    model = FirmStructuralCreditRisk()
    res = model.fit(
        expected_market_ret=expected_market_ret,
        market_rets=market_rets,
        equity_values=dataset['marketCap'].values,
        debt_values=debt_values,
        debt_maturities=[1] * len(dataset),
        interval=interval,
        risk_free_rates=risk_free_rates,
    )
    if return_assets:
        return model, res
    else:
        return model


def get_default_count_probs(default_vectors):
    default_vectors = [v[:, np.newaxis] for v in default_vectors]

    while len(default_vectors) > 1:
        next_vectors = []
        for i in range(0, len(default_vectors) - 1, 2):
            default_mat = default_vectors[i] @ default_vectors[i + 1].T
            max_num_defaults = default_mat.shape[0] + default_mat.shape[1] - 2
            default_vec = np.zeros(max_num_defaults + 1)

            for j in range(default_mat.shape[0]):
                for k in range(default_mat.shape[1]):
                    default_vec[j + k] += default_mat[j, k]

            default_vec = default_vec[:, np.newaxis]
            next_vectors.append(default_vec)

        if len(default_vectors) % 2 != 0:
            next_vectors.append(default_vectors[-1])
        default_vectors = next_vectors

    return default_vectors[0].squeeze()


def get_systematic_default_probs(datasets, interval, firms_data, market_ret):
    firm_models = {
        ticker: create_firm_model(
            dataset,
            interval=interval,
            expected_market_ret=market_ret,
        )
        for ticker, dataset in datasets.items()
    }

    default_probs = {
        ticker: model.predict_default(
            equity_value=firms_data[ticker]['equity_value'],
            debt_value=firms_data[ticker]['debt_value'],
            debt_maturity=firms_data[ticker]['debt_maturity'],
            risk_free_rate=firms_data[ticker]['risk_free_rate'],
            market_ret=market_ret
        )
        for ticker, model in firm_models.items()
    }
    default_vectors = [np.array([1 - p, p]) for p in default_probs.values()]
    sys_default_probs = get_default_count_probs(default_vectors)
    return sys_default_probs, default_probs, firm_models

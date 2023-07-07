import pytest
import pandas as pd
import numpy as np

from financial_ml.features.systematic_default import FirmStructuralCreditRisk, predict_systematic_default


def test_firm_model():
    dataset = pd.read_csv('../../data/systematic_default/aapl_data.csv')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset_monthly = dataset.resample('MS', on='date').first()

    equity_ratio = 0.81
    equity_mkt_rets = np.diff(np.log(dataset_monthly['equities_index']))
    bnd_mkt_rets = np.diff(np.log(dataset_monthly['bonds_index']))
    market_rets = equity_ratio * equity_mkt_rets + (1 - equity_ratio) * bnd_mkt_rets
    debt_values = (dataset_monthly['shortTermDebt'] + dataset_monthly['longTermDebt']/2).values
    risk_free_rates = dataset_monthly['treasury_yield'].iloc[:-1].values

    model = FirmStructuralCreditRisk()
    asset_rets, asset_values = model.fit(
        expected_market_ret=0.12,
        market_rets=market_rets,
        equity_values=dataset_monthly['marketCap'].values,
        debt_values=debt_values,
        debt_maturities=[1] * len(dataset_monthly),
        interval=1/12,
        risk_free_rates=risk_free_rates,
    )
    assert len(asset_rets) == len(dataset_monthly) - 1
    assert len(asset_values) == len(dataset_monthly) - 1
    assert model.firm_params.alpha == pytest.approx(0.0623, abs=1e-3)
    assert model.firm_params.beta == pytest.approx(1.5044, abs=1e-3)
    assert model.firm_params.sigma == pytest.approx(0.1942, abs=1e-3)

    last_rec = dataset.iloc[-1, :]
    default_prob = model.predict_default(
        equity_value=last_rec['marketCap'],
        debt_value=(last_rec['shortTermDebt'] + last_rec['longTermDebt']/2),
        debt_maturity=1,
        risk_free_rate=last_rec['treasury_yield'],
        market_ret=-3
    )
    assert default_prob == pytest.approx(0.5271, abs=1e-3)


def test_systematic_default():
    tickers = ['COST', 'CSCO', 'CRM', 'ACN', 'ADBE', 'LIN', 'TXN', 'DHR']
    datasets = {
        ticker: pd.read_csv(f'../../data/systematic_default/{ticker.lower()}_data.csv', parse_dates=['date'])
        for ticker in tickers
    }
    datasets_monthly = {
        ticker: dataset.resample('MS', on='date').first()
        for ticker, dataset in datasets.items()
    }
    firms_data = {
        ticker: {
            'equity_value': dataset['marketCap'].iloc[-1],
            'debt_value': (dataset['shortTermDebt'].iloc[-1] + dataset['longTermDebt'].iloc[-1] / 2),
            'debt_maturity': 1,
            'risk_free_rate': dataset['treasury_yield'].iloc[-1],
        }
        for ticker, dataset in datasets.items()
    }
    sys_probs, _, _ = predict_systematic_default(datasets_monthly, 1 / 12, firms_data, 0.12)
    np.testing.assert_array_almost_equal(sys_probs, np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.]))

import pandas as pd
import numpy as np

from financial_ml.features.systematic_default import FirmStructuralCreditRisk


def test_model_fit():
    dataset = pd.read_csv('../../data/aapl_structural_data.csv')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset_monthly = dataset.resample('MS', on='date').first()

    equity_ratio = 0.81
    equity_mkt_rets = np.diff(np.log(dataset_monthly['mkt_price']))
    bnd_mkt_rets = np.diff(np.log(dataset_monthly['bnd_price']))
    market_rets = equity_ratio * equity_mkt_rets + (1 - equity_ratio) * bnd_mkt_rets
    debt_values = (dataset_monthly['shortTermDebt'] + dataset_monthly['longTermDebt'] / 2).values
    risk_free_rates = dataset_monthly['treasury_yield'].iloc[:-1].values

    model = FirmStructuralCreditRisk()
    res = model.fit(
        expected_market_ret=0.12,
        market_rets=market_rets,
        equity_values=dataset_monthly['marketCap'].values,
        debt_values=debt_values,
        debt_maturities=[1] * len(dataset_monthly),
        interval=1/12,
        risk_free_rates=risk_free_rates,
    )
    print(res)
    print(model.firm_params)

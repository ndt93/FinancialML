import pytest
import numpy as np

from risk.strategy_risk import (
    strategy_sharpe_ratio,
    strategy_required_precision,
    strategy_required_freq,
    simulate_strategy_probability_of_failure
)


def test_strategy_sharpe_ratio():
    p = 0.7
    freq = 260
    profit = 0.005
    loss = -0.01
    assert strategy_sharpe_ratio(p, freq, profit, loss) == pytest.approx(1.173, rel=1e-3)


def test_strategy_required_precision():
    target_sr = 1.173
    freq = 260
    profit = 0.005
    loss = -0.01
    assert strategy_required_precision(target_sr, freq, profit, loss) == pytest.approx(0.7, rel=1e-3)


def test_strategy_required_freq():
    target_sr = 1.173
    p = 0.7
    profit = 0.005
    loss = -0.01
    assert strategy_required_freq(target_sr, p, profit, loss) == pytest.approx(260, rel=1e-3)


def test_strategy_probability_of_failure():
    np.random.seed(42)
    mu1 = 0.01
    mu2 = -0.01
    sigma1 = 0.01
    sigma2 = 0.01
    prob1 = 0.5
    freq = 260
    n_samples = 260*10
    target_sr = 1

    fail_prob = simulate_strategy_probability_of_failure(mu1, mu2, sigma1, sigma2, prob1, n_samples, freq, target_sr)
    assert 0.45 < fail_prob < 0.55

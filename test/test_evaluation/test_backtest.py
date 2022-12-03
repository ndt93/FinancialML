import pytest
import numpy as np
from scipy.stats import norm

from financial_ml.evaluation.backtest import compute_pbo


@pytest.fixture
def perm_matrix_norm_rets():
    np.random.seed(42)
    n_strats = 10
    n_obs = 300
    mean_ret = 0.08
    std_ret = 0.15
    return norm.rvs(size=(n_obs, n_strats), loc=mean_ret, scale=std_ret, random_state=42)


@pytest.fixture
def perm_matrix_optimal():
    np.random.seed(42)
    n_strats = 10
    n_obs = 300
    mean_ret = 0.08
    std_ret = 0.15
    optimal_strat_idx = 7
    optimal_strat_mean = 0.1
    optimal_strat_std = 0.12
    out = norm.rvs(size=(n_obs, n_strats), loc=mean_ret, scale=std_ret, random_state=42)
    out[:, optimal_strat_idx] = norm.rvs(size=n_obs, loc=optimal_strat_mean, scale=optimal_strat_std, random_state=42)
    return out


def test_compute_pbo_norm_rets(perm_matrix_norm_rets):
    pbo, rank_logits, train_optimal_perm, test_assoc_perm = compute_pbo(perm_matrix_norm_rets, n_partitions=16)
    assert len(rank_logits) == 12870
    assert len(train_optimal_perm) == 12870
    assert len(test_assoc_perm) == 12870
    assert pbo >= 0.5


def test_compute_pbo_optimal(perm_matrix_optimal):
    pbo, _, _, _ = compute_pbo(perm_matrix_optimal, n_partitions=16)
    assert pbo < 0.1

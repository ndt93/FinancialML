import random

import numpy as np
import pandas as pd
import pytest

from financial_ml.assets_allocation import HRP


def generate_data(n_obs, n_factors, n_instruments, sigma):
    np.random.seed(12345)
    random.seed(12345)
    x = np.random.normal(0, 0.1, size=(n_obs, n_factors))
    cols = [random.randint(0, n_factors - 1) for _ in range(n_instruments)]
    y = x[:, cols] + np.random.normal(0, sigma, size=(n_obs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols


@pytest.fixture
def returns_data():
    n_obs, n_factors, n_instr, sigma = 500, 5, 5, .1
    data, cols = generate_data(n_obs, n_factors, n_instr, sigma)
    return data, cols


def test_corr_to_dist():
    corr_mat = np.array([
        [1, .7, .2],
        [.7, 1, -.2],
        [.2, -.2, 1]
    ])
    dist_mat = HRP._corr_to_dist(corr_mat)
    expected = np.array([
        [0, .3873, .6325],
        [.3873, 0, .7746],
        [.6325, .7746, 0]
    ])
    np.testing.assert_array_almost_equal(dist_mat, expected, decimal=3)


def test_hrp(returns_data):
    returns, factors = returns_data
    w = HRP.hrp_allocations(returns.values)
    assert w.sum() == pytest.approx(1)
    assert w.index[w.argmax()] == 4

    cov = np.cov(returns, rowvar=False)
    vars = np.var(returns, axis=0).values
    cl1_var = w[0]**2*vars[0] + w[6]**2*vars[6] + 2*w[0]*w[6]*cov[0, 6]
    cl2_var = w[3]**2*vars[3] + w[5]**2*vars[5] + 2*w[3]*w[5]*cov[3, 5]
    cl3_var = w[1]**2*vars[1] + w[9]**2*vars[9] + 2*w[1]*w[9]*cov[1, 9]
    cl4_var = w[2]**2*vars[2] + w[7]**2*vars[7] + w[8]**2*vars[8] + 2*(w[2]*w[7]*cov[2, 7] + w[2]*w[8]*cov[2, 8])
    cl5_var = w[4]**2*vars[4]
    cl_vars = [cl1_var, cl2_var, cl3_var, cl4_var, cl5_var]
    for i in range(len(cl_vars)):
        for j in range(i + 1, len(cl_vars)):
            assert abs(cl_vars[i] - cl_vars[j]) < 0.001

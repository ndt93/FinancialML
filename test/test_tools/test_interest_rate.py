import pytest
import numpy as np

from financial_ml.tools.interest_rate import interpolate_yield_curve, get_implied_forward_rate


def test_interpolate_yield_curve():
    t = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    yields = np.array([3.81, 4.13, 4.31, 4.72, 4.72, 4.33, 4.07, 3.75, 3.69, 3.57, 3.82, 3.56])/100
    curve = interpolate_yield_curve(list(zip(t, yields)))
    np.testing.assert_array_almost_equal(curve(t), yields)


def test_implied_forward_rate():
    ir = get_implied_forward_rate(1, 1.1, 0.5)
    assert ir == pytest.approx(0.1906, abs=1e-3)

import pytest
import numpy as np
import pandas as pd

from features.entropy import (
    binary_encode,
    QuantileEncoder,
    sigma_encode,
    konto_entropy,
    plugin_entropy,
    lempelziv_lib
)


def test_binary_encode():
    series = pd.Series([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])
    out = binary_encode(series, 0.5)
    assert out == '000011'


def test_quantile_encode():
    series = pd.Series(np.linspace(0, 100, 21))
    encoder = QuantileEncoder(21)
    out = encoder.fit_transform(series)
    assert out == '0123456789abcdefghijk'


def test_sigma_encode():
    series = pd.Series(np.linspace(0, 100, 21))
    out = sigma_encode(series, 5)
    assert out == '0123456789abcdefghijk'


def test_lz():
    msg = 'YRCADDOHDOSU'
    out = lempelziv_lib(msg)
    assert len(out.difference({'Y', 'R', 'C', 'A', 'D', 'DO', 'H', 'DOS', 'U'})) == 0


def test_koto_entropy():
    assert konto_entropy('10000111') == konto_entropy('10000110')
    assert konto_entropy('11100001') == pytest.approx(0.9682, abs=1e-3)
    assert konto_entropy('01100001') == pytest.approx(0.8432, abs=1e-3)


def test_plugin_entropy():
    assert plugin_entropy('10000111', 1) == pytest.approx(1.0, abs=1e-3)
    assert plugin_entropy('10000110', 1) == pytest.approx(0.9544, abs=1e-3)
    assert plugin_entropy('11100001', 1) == pytest.approx(1.0, abs=1e-3)
    assert plugin_entropy('01100001', 1) == pytest.approx(0.9544, abs=1e-3)

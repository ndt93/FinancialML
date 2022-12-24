import pandas as pd
import numpy as np

from financial_ml.features.filters import filter_symmetric_cusum, get_mmi
from financial_ml.utils.simulation import gen_geometric_brownian


def test_symmetric_cumsum_filter():
    time_index = pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 16)])
    series = pd.Series(
        0 + np.array([1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1]).cumsum(),
        index=time_index
    )
    events = filter_symmetric_cusum(series, 3)
    np.testing.assert_array_equal(events.values, time_index[[4, 10]].values)


def test_mmi():
    series = np.array(list(gen_geometric_brownian(100, 0.0, 0.15, 1000, random_state=42)))
    series = np.diff(np.log(series))
    mmi = get_mmi(series)
    assert 0.73 < mmi < 0.77

import pandas as pd
import numpy as np

from financial_ml.events.filters import filter_symmetric_cusum


def test_symmetric_cumsum_filter():
    time_index = pd.to_datetime([f'2022-01-{i:02d}' for i in range(1, 16)])
    series = pd.Series(
        0 + np.array([1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1]).cumsum(),
        index=time_index
    )
    events = filter_symmetric_cusum(series, 3)
    np.testing.assert_array_equal(events.values, time_index[[4, 10]].values)

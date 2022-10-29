import pandas as pd

from data_structures.constants import PortfolioCol
from evaluation.metrics import compute_time_weighted_return


def test_twrr():
    out = compute_time_weighted_return(
        pd.DataFrame({
            PortfolioCol.BEGIN_VALUE: [1, 2, 4, 8, 16],
            PortfolioCol.CASHFLOW: [0]*5
        })
    )
    print(out)
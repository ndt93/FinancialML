import pandas as pd


def fill_index(indices, series, fill_val=0.):
    if not isinstance(fill_val, (float, int)):
        raise NotImplementedError('fill_val type not supported', type(fill_val))
    return (pd.Series(fill_val, index=indices) + series).fillna(0)

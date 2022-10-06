import numpy as np
from scipy.stats import norm


def generate_geometric_brownian(s0: float, size: int, interval=1/365, mu=0.08, sigma=0.15):
    series = [s0]
    rets = mu*interval + sigma*np.sqrt(interval)*norm.rvs(size=size - 1)
    for t in range(size - 1):
        prev_s = series[-1]
        s_t = prev_s + prev_s * rets[t]
        series.append(s_t)
    return series

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from labeling.weighting import count_events_per_bar


def compute_gaussian_mixture_position_size(bar_times, event_times, sides, **kwargs):
    """
    Size position using the cumulative distribution of number of concurrent positions, defined as
    c_t = c_t_long - c_t_short. The distribution is modeled using a mixture of 2 Gaussians.
    position size = (F(c_t) - F(0))/(1 - F(0)) if c_t >= 0 and (F(c_t) - F0)/F(0) if c_t < 0.

    :param bar_times: a Series of bar Timestamps
    :param event_times: a Series of event start times (in index) and end times (in values).
    Each event is assumed to signal 1 potential position
    :param sides: a Series of 1 (long position) and -1 (short position) signal for each event
    :param kwargs: arguments to pass to sklearn GaussianMixture model.
        Exclude n_components (fixed to 2) and covariance_type (fixed to 'spherical')
    :return: a Series of position sizes from -1 (full short) to 1 (full long) for each bar time
    """
    long_concurrency = count_events_per_bar(bar_times, event_times[sides == 1])
    short_concurrency = count_events_per_bar(bar_times, event_times[sides == -1])
    concurrency = pd.Series(0, index=bar_times) + long_concurrency - short_concurrency

    mixture_model = GaussianMixture(n_components=2, covariance_type='spherical', **kwargs)
    mixture_model.fit(concurrency.to_frame('num_pos'))
    rvs = [norm(loc=m, scale=np.sqrt(v)) for m, v in zip(mixture_model.means_, mixture_model.covariances_)]
    cdfs = np.array([dist.cdf(concurrency.values) for dist in rvs]).T
    mixed_cdfs = (cdfs * mixture_model.weights_).sum(axis=1)
    no_pos_cdf = (np.array([dist.cdf(0) for dist in rvs]) * mixture_model.weights_).sum()
    res = (mixed_cdfs - no_pos_cdf) / np.where(concurrency.values >= 0, (1 - no_pos_cdf), no_pos_cdf)
    return pd.Series(res, index=concurrency.index)

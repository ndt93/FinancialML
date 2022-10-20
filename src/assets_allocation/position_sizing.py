import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from data_structures.constants import EventCol, PositionCol
from labeling.weighting import count_events_per_bar


def compute_gaussian_mixture_position_size(bar_times, event_times, sides, **kwargs):
    """
    Size position using the cumulative distribution of number of concurrent positions, defined as
    c_t = c_t_long - c_t_short. The distribution is modeled using a mixture of 2 Gaussians.
    position_size_t = (F(c_t) - F(0))/(1 - F(0)) if c_t >= 0 and (F(c_t) - F0)/F(0) if c_t < 0.

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
    concurrency = ((pd.Series(0, index=bar_times) + long_concurrency).fillna(0) - short_concurrency).fillna(0)

    mixture_model = GaussianMixture(n_components=2, covariance_type='spherical', **kwargs)
    mixture_model.fit(concurrency.to_frame('num_pos'))
    rvs = [norm(loc=m, scale=np.sqrt(v)) for m, v in zip(mixture_model.means_, mixture_model.covariances_)]
    cdfs = np.array([dist.cdf(concurrency.values) for dist in rvs]).T
    mixed_cdfs = (cdfs * mixture_model.weights_).sum(axis=1)
    no_pos_cdf = (np.array([dist.cdf(0)[0] for dist in rvs]) * mixture_model.weights_).sum()
    res = (mixed_cdfs - no_pos_cdf) / np.where(concurrency.values >= 0, (1 - no_pos_cdf), no_pos_cdf)
    return pd.Series(res, index=concurrency.index)


def compute_budgeted_position_size(bar_times, event_times, sides, budget_fn=max):
    """
    Size position using a certain budget function e.g. max or some quantile of number of concurrent long/short
    positions. position_size_t = c_t_l/budget_fn([c_i_l]) - c_t_s/budget_fn([c_i_s])

    :param bar_times: a Series of bar Timestamps
    :param event_times: a Series of event start times (in index) and end times (in values).
        Each event is assumed to signal 1 potential position
    :param sides: a Series of 1 (long position) and -1 (short position) signal for each event
    :param budget_fn: function to apply over all concurrent long or short positions count and return a budget number
    :return: a Series of position sizes from -1 (full short) to 1 (full long) for each bar time
    """
    long_concurrency = count_events_per_bar(bar_times, event_times[sides == 1])
    short_concurrency = count_events_per_bar(bar_times, event_times[sides == -1])
    long_budget = budget_fn(long_concurrency.values)
    short_budget = budget_fn(short_concurrency.values)
    res = (pd.Series(0, index=bar_times) + long_concurrency/long_budget).fillna(0)
    res = (res - short_concurrency/short_budget).fillna(0)
    return res


def _avg_active_positions(positions: pd.DataFrame):
    position_times = positions.index.union(positions[PositionCol.END_TIME].values).dropna().drop_duplicates()
    res = pd.Series(dtype=float)
    for t in position_times:
        is_active = (positions.index.values <= t) & \
                    ((t < positions[PositionCol.END_TIME]) | pd.isnull(positions[PositionCol.END_TIME]))
        active_pos_starts = positions[is_active].index
        if len(active_pos_starts) > 0:
            res[t] = positions.loc[active_pos_starts, PositionCol.SIZE].mean()
        else:
            res[t] = 0
    return res


def _discretize_position_sizes(sizes, step_size):
    discrete_sizes = (sizes / step_size).round() * step_size
    discrete_sizes[discrete_sizes > 1] = 1
    discrete_sizes[discrete_sizes < -1] = -1
    return discrete_sizes


def compute_position_size_from_probabilities(
        events: pd.DataFrame, step_size, prob: pd.Series, pred: pd.Series, num_classes: int, **kwargs
):
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)

    test_stats = (prob - 1./num_classes)/(prob * (1. - prob))**0.5
    sizes = pred * (2*norm.cdf(test_stats) - 1)
    if EventCol.SIDE in events.columns:
        sizes *= events.loc[sizes.index, EventCol.SIDE]

    positions = sizes.to_frame(PositionCol.SIZE).join(events[[EventCol.END_TIME]], how='left')
    sizes = _avg_active_positions(positions)
    res = _discretize_position_sizes(sizes, step_size=step_size)
    return res

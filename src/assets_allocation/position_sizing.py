import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from data_structures.constants import EventCol, PositionCol
from labeling.weighting import count_events_per_bar
from utils.data import fill_index


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
    concurrency = fill_index(bar_times, long_concurrency) - fill_index(bar_times, short_concurrency)

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
    res = fill_index(bar_times, long_concurrency/long_budget) - fill_index(bar_times, short_concurrency/short_budget)
    return res


# Size positions using ML model's probabilities and prediction
def _avg_active_positions(positions: pd.DataFrame):
    position_times = positions.index.union(positions[PositionCol.END_TIME].values).dropna().drop_duplicates()
    res = pd.Series(dtype=float)
    for t in position_times:
        is_active = (positions.index.values <= t) & \
                    ((t <= positions[PositionCol.END_TIME]) | pd.isnull(positions[PositionCol.END_TIME]))
        active_pos_starts = positions[is_active].index.drop_duplicates()
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
        events: pd.DataFrame, step_size, prob: pd.Series, pred: pd.Series, num_classes: int
) -> pd.Series:
    """
    Size positions by the model's predicted probabilities p of each event.
    Using the test statistic: z = (p - 1/num_cls)/sqrt(p(1 - p)): size = pred_cls * (2 * F(z) - 1)
    Concurrent position sizes are averaged and discretized using step_size

    :param events: DataFrame with EventCol.END_TIME and optionally EventCol.SIDE
    :param step_size: step used to discretize position sizes
    :param prob: Series of predicted probabilities of each event. Should have same index as events
    :param pred: Series of class prediction of each event. Should have same index as events
    :param num_classes: Number of classification clasess
    :return: a Series of position sizes between -1 and 1 at all events start times and end_times
    """

    if prob.shape[0] == 0:
        return pd.Series(dtype=float)

    test_stats = (prob - 1./num_classes)/(prob * (1. - prob))**0.5
    sizes = pred * (2*norm.cdf(test_stats) - 1)
    if EventCol.SIDE in events.columns:
        sizes = sizes * events[EventCol.SIDE]

    positions = sizes.to_frame(PositionCol.SIZE)
    positions[EventCol.END_TIME] = events[EventCol.END_TIME]
    sizes = _avg_active_positions(positions)
    res = _discretize_position_sizes(sizes, step_size=step_size)
    return res


# Dynamic position sizing from divergence between forecasts and market prices
def _scaled_sigmoid(w, x):
    return x * (w + x ** 2) ** -0.5


def _inverse_price(f, w, m):
    return f - m*(w/(1 - m**2))**.5


def get_sigmoid_coeff(x, m):
    return x**2 * (m**-2 - 1)


def compute_position_size_from_divergence(forecast, market_price, max_size, w):
    """
    Size positions dynamically based on the divergence between forecasted price and market price.
    size_i_t = int[m(w, forecast_i - mkPrice_t) * max_size]
    m(w, x) = x^2/sqrt(w + x^2) is a sigmoid function with scaling coefficient w.
        Supports for other functions will be implemented in the future.

    Use compute_limit_price function to get the limit for the order to change from current position size
    to the new target position size

    :param forecast: the forecast price
    :param market_price: the market price
    :param max_size: maximum position size
    :param w: scaling coefficient for the sigmoid function
    :return: position sizes from -max_size to max_size
    """
    return int(_scaled_sigmoid(w, forecast - market_price) * max_size)


def compute_limit_price(cur_pos, target_pos, max_position, forecast, w):
    """
    Compute the break-even limit price for the order to change from the current position to
    target position (using the compute_position_size_from_divergence function).

    :param cur_pos: current position sizes
    :param target_pos: target position sizes
    :param max_position: max position size
    :param forecast: forecasted price
    :param w: regulating coefficient for the sigmoid function
    :return: the limit price for the order
    """
    sign = 1 if target_pos > cur_pos else -1
    limit_price = 0
    for j in range(abs(cur_pos + sign), abs(target_pos + 1)):
        limit_price += _inverse_price(forecast, w, j/float(max_position))
    limit_price /= target_pos - cur_pos
    return limit_price

import numpy as np
from scipy.stats import norm


def gen_geometric_brownian(s0: float, drift, volatility, num_steps: int, interval=0.001, random_state=None):
    """
    Generate a geometric brownian stochastic process: dS = mu*S*dt + sigma*S*dz

    :param s0: initial value
    :param drift: mu
    :param volatility: sigma
    :param num_steps: number of simulation steps
    :param interval: dt
    :param random_state: random state
    :return: realized times series of a geometric brownian stochastic process
    """
    s = s0
    for _ in range(num_steps):
        ret = drift*interval + volatility*np.sqrt(interval)*norm.rvs(random_state=random_state)
        s += s * ret
        yield s


def gen_discrete_ou_process(forecast, half_life, sigma, seed, num_steps, random_state=None):
    """
    Generate a discrete Ornstein-Uhlenbeck stochastic process:
        P(t) = (1 - phi)*E[P(T)] + phi*P(t-1) + sigma*z(t)
    This process converges to an expected value E[P(T)] at a rate determined by phi.

    :param forecast: E[P(T)]
    :param half_life: phi = 2^(-1/half_life)
    :param sigma: coefficient for the random shocks
    :param seed: P(0)
    :param num_steps: number of simulation steps
    :return: times series of a geometric brownian stochastic process
    """
    p = seed
    phi = 2**(-1/half_life)
    for _ in range(num_steps):
        p = (1 - phi)*forecast + phi*p + sigma*norm.rvs(random_state=random_state)
        yield p


def gen_mixed_gaussians(mu1, mu2, sigma1, sigma2, prob1, n_samples, np_rand_seed=None):
    if np_rand_seed:
        np.random.seed(np_rand_seed)
    ret1 = np.random.normal(mu1, sigma1, size=int(n_samples*prob1))
    ret2 = np.random.normal(mu2, sigma2, size=int(n_samples) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret

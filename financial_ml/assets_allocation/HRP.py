"""
This module implements the Hierarchical Risk Parity (HRP) algorithm for portfolio assets allocation & optimization.
HRP is a modern ML & graph theory based alternative to Markowitz's Critical Line Algorithm (CLA) for mean-variance
efficient frontier optimization. It is more stable and outperforms CLA or other quadratic optimizers in general
on out-of-sample data.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


def _corr_to_dist(corr_mat: np.ndarray):
    """
    Encode the correlation matrix as a proper metrics space distance metrics.
    Higher correlation results in closer distance.

    :param corr_mat: correlation matrix
    :return: encoded distance matrix from the correlation matrix
    """
    return np.sqrt((1 - corr_mat)/2.)


def _quasi_diagonalize(linkage_mat: np.ndarray):
    """
    Reorganize the rows and columns of a covariance matrix so that the largest values lie along the diagonal,
    without requiring a change of basis. This preserves the original investments, placing similar investment
    closer and dissimilar investments further.

    :param linkage_mat: matrix in from similar to output of scipy hierarchical clustering linkage function
        applied on the encoded distance matrix (see corr_to_dist) of the target covariance matrix
    :return: a reordering list of indices of the original items
    """
    linkage_mat = linkage_mat.astype(int)
    sort_indices = pd.Series([linkage_mat[-1, 0], linkage_mat[-1, 1]])
    n_items = linkage_mat[-1, 3]

    while sort_indices.max() >= n_items:
        sort_indices.index = range(0, sort_indices.shape[0]*2, 2)
        df_cluster = sort_indices[sort_indices >= n_items]
        i = df_cluster.index
        j = df_cluster.values - n_items
        sort_indices[i] = linkage_mat[j, 0]
        df_cluster = pd.Series(linkage_mat[j, 1], index=i+1)
        sort_indices = sort_indices.append(df_cluster)
        sort_indices = sort_indices.sort_index()
        sort_indices.index = range(sort_indices.shape[0])

    return sort_indices.tolist()


def _get_inverse_cov_portfolio(cov: np.ndarray) -> np.ndarray:
    res = 1./np.diag(cov)
    res /= res.sum()
    return res


def _get_cluster_var(cov: np.ndarray, cluster_indices: list) -> float:
    cov = cov[np.ix_(cluster_indices, cluster_indices)]
    weights = _get_inverse_cov_portfolio(cov).reshape(-1, 1)
    var = np.dot(np.dot(weights.T, cov), weights)[0, 0]
    return var


def _recursive_bisection(cov: np.ndarray, sort_indices: list):
    """
    Recursively bisect and perform inverse-variance allocation

    :param cov: a covariance matrix
    :param sort_indices: sort indices for quasi-diagonalization of cov
    :return:
    """
    weights = pd.Series(1, index=sort_indices)
    partitions = [sort_indices]
    while len(partitions) > 0:
        partitions = [
            p[i:j]
            for p in partitions
            for i, j in [(0, len(p)//2), (len(p)//2, len(p))]
            if len(p) > 1
        ]
        for i in range(0, len(partitions), 2):
            part_1 = partitions[i]
            part_2 = partitions[i+1]
            var_1 = _get_cluster_var(cov, part_1)
            var_2 = _get_cluster_var(cov, part_2)
            alpha = 1 - var_1/(var_1 + var_2)
            weights[part_1] *= alpha
            weights[part_2] *= 1 - alpha
    return weights


def hrp_allocations(returns_mat: np.ndarray, **kwargs):
    """
    Get Hierarchical Risk Parity portfolio allocations in 4 steps:
    1. Encode the correlation matrix as a valid distance metrics in the metrics space
    2. Run hierarchical clustering on the encoded distance matrix
    3. Quasi-diagonalize the covariance matrix using the clusters in (2)
    4. Recursively bisect and compute the inverse variance allocation weights on diagonalized matrix in (3)

    :param returns_mat: TxN matrix of instrument returns. 1 column per instrument. 1 row per observation
    :param kwargs: arguments for scipy's linkage method for hierarchical clustering
    :return: a Series of allocation weights
    """
    cov = np.cov(returns_mat, rowvar=False)
    corr = np.corrcoef(returns_mat, rowvar=False)
    dist_mat = _corr_to_dist(corr)
    link_mat = linkage(dist_mat, **kwargs)
    sort_indices = _quasi_diagonalize(link_mat)
    alloc_weights = _recursive_bisection(cov, sort_indices)
    return alloc_weights

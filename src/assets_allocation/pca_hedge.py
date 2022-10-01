import numpy as np


def pca_weights(cov: np.ndarray, risk_dist=None, risk_target=1.0):
    eig_val, eig_vec = np.linalg.eigh(cov)
    sort_indices = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[sort_indices], eig_vec[:, sort_indices]
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0
    loads = risk_target * (risk_dist/eig_val)**0.5
    weights = np.dot(eig_vec, np.reshape(loads, (-1, 1)))
    return weights

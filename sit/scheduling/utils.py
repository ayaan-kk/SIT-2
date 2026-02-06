"""Utility functions for SIT scheduling.

Provides kernel construction, log-determinant objectives for diversity,
and predicted-risk computation from tomography matrices.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sit.simulator.workloads import Workload


def build_similarity_kernel(
    workloads: List[Workload],
    sigma: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """Build an RBF similarity kernel from workload resource vectors.

    K[i, j] = exp(-||x_i - x_j||^2 / sigma^2)

    Parameters
    ----------
    workloads : list of Workload
        Workload objects whose ``resource_vector`` attributes are used.
    sigma : float
        Bandwidth parameter for the RBF kernel.  Larger values yield a
        smoother (more uniform) similarity surface.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        Symmetric positive-semidefinite kernel matrix.
    workload_names : list of str
        Ordered list of workload names corresponding to rows/columns of *K*.
    """
    n = len(workloads)
    workload_names = [w.name for w in workloads]

    # Stack resource vectors into (n, d) matrix
    X = np.stack([w.resource_vector for w in workloads], axis=0)  # (n, d)

    # Pairwise squared Euclidean distances via expansion:
    #   ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    sq_norms = np.sum(X ** 2, axis=1)  # (n,)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T  # (n, n)

    # Clamp numerical noise that can make very small negatives
    dist_sq = np.maximum(dist_sq, 0.0)

    K = np.exp(-dist_sq / (sigma ** 2))
    return K, workload_names


def logdet_objective(
    K_sub: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute log det(K_sub + epsilon * I).

    This is the diversity objective used in DPP-style selection: a higher
    value indicates that the selected subset spans a larger volume in
    feature space.

    Parameters
    ----------
    K_sub : np.ndarray, shape (m, m)
        Sub-matrix of the kernel corresponding to the selected items.
    epsilon : float
        Regularisation constant added to the diagonal for numerical
        stability (ensures strict positive definiteness).

    Returns
    -------
    float
        log det(K_sub + epsilon * I).
    """
    m = K_sub.shape[0]
    if m == 0:
        return 0.0
    regularised = K_sub + epsilon * np.eye(m)
    # Use slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(regularised)
    if sign <= 0:
        # Should not happen with proper regularisation, but guard anyway
        return -np.inf
    return logdet


def marginal_gain(
    K: np.ndarray,
    current_set: List[int],
    candidate: int,
    epsilon: float = 1e-6,
) -> float:
    """Compute the marginal log-det gain of adding *candidate* to *current_set*.

    gain = logdet(K_{S + {c}}) - logdet(K_S)

    This equals the log of the Schur complement determinant, which
    measures how much new "volume" (diversity) the candidate adds.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Full kernel matrix.
    current_set : list of int
        Indices of already-selected items.
    candidate : int
        Index of the candidate item.
    epsilon : float
        Regularisation constant.

    Returns
    -------
    float
        Marginal gain in log-det diversity.
    """
    if len(current_set) == 0:
        # First item: gain = log(K[c,c] + epsilon)
        return np.log(K[candidate, candidate] + epsilon)

    S = list(current_set)
    # Current sub-matrix
    K_S = K[np.ix_(S, S)]
    # Extended sub-matrix with candidate
    S_new = S + [candidate]
    K_S_new = K[np.ix_(S_new, S_new)]

    return logdet_objective(K_S_new, epsilon) - logdet_objective(K_S, epsilon)


def compute_predicted_risk(
    target_name: str,
    cotenant_names: List[str],
    tomography_mean: pd.DataFrame,
    tomography_stderr: Optional[pd.DataFrame] = None,
    beta: float = 0.0,
) -> float:
    """Sum predicted interference from all co-tenants onto the target.

    For each co-tenant *s*, the predicted interference is looked up from
    the tomography mean matrix as ``tomography_mean.loc[target_name, s]``.
    If *beta* > 0 and *tomography_stderr* is provided, the upper
    confidence bound (UCB) is used instead:

        risk_s = mean_s + beta * stderr_s

    Parameters
    ----------
    target_name : str
        Name of the target workload (row in the tomography matrix).
    cotenant_names : list of str
        Names of spectator workloads co-located with the target.
    tomography_mean : pd.DataFrame
        DataFrame with target names as rows and spectator names as columns.
        Each entry is the mean predicted interference (e.g. latency uplift).
    tomography_stderr : pd.DataFrame or None
        Same shape as *tomography_mean* but holding standard errors.
    beta : float
        UCB exploration weight.  When 0, only the mean is used.

    Returns
    -------
    float
        Total predicted risk (sum of per-co-tenant interference estimates).
    """
    if len(cotenant_names) == 0:
        return 0.0

    total_risk = 0.0
    for s in cotenant_names:
        mean_val = tomography_mean.loc[target_name, s]
        if beta > 0.0 and tomography_stderr is not None:
            stderr_val = tomography_stderr.loc[target_name, s]
            total_risk += mean_val + beta * stderr_val
        else:
            total_risk += mean_val

    return float(total_risk)

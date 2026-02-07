"""Formalize SIT tomography as a linear inverse problem y = Ax + epsilon.

Provides identifiability diagnostics (condition number, rank, mutual
coherence, restricted isometry proxy) and L1-regularized reconstruction
algorithms (ISTA, group LASSO, nonneg-L1) using only NumPy.

Mathematical setup
------------------
Let *T* targets and *S* spectators define the tomography grid.  The
measurement matrix **A** has shape ``(T, S)`` where ``A[t, s]`` is the
observed interference effect of spectator *s* on target *t*.  The linear
inverse problem is:

.. math::

    \\mathbf{y} = \\mathbf{A}\\,\\mathbf{x} + \\boldsymbol{\\epsilon}

where **y** is a per-target observation vector, **x** is the vector of
spectator interference coefficients, and epsilon is noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Soft-thresholding (proximal operator for L1)
# ---------------------------------------------------------------------------

def _soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding: sign(z) * max(|z| - threshold, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


# ---------------------------------------------------------------------------
# TomographyModel class
# ---------------------------------------------------------------------------

class TomographyModel:
    """Linear inverse-problem model for SIT interference tomography.

    Parameters
    ----------
    tomo_mean_df : pd.DataFrame
        Mean tomography matrix with targets as rows and spectators as
        columns.  Each entry ``(t, s)`` is the mean interference effect
        of spectator *s* on target *t*.
    tomo_stderr_df : pd.DataFrame
        Standard-error matrix with the same shape and index structure as
        *tomo_mean_df*.
    """

    def __init__(
        self,
        tomo_mean_df: pd.DataFrame,
        tomo_stderr_df: pd.DataFrame,
    ) -> None:
        self.tomo_mean_df = tomo_mean_df
        self.tomo_stderr_df = tomo_stderr_df
        self.targets: List[str] = list(tomo_mean_df.index)
        self.spectators: List[str] = list(tomo_mean_df.columns)

    # -- measurement matrix -------------------------------------------------

    def measurement_matrix(self) -> np.ndarray:
        """Construct the measurement matrix **A**.

        Rows correspond to observations indexed by target--spectator
        pairs (one row per target) and columns correspond to spectator
        interference coefficients.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_targets, n_spectators)`` extracted from
            the mean tomography DataFrame.
        """
        A = np.asarray(self.tomo_mean_df.values, dtype=np.float64)
        return A

    # -- identifiability diagnostics ----------------------------------------

    def condition_number(self) -> float:
        """Condition number of **A** (ratio of largest to smallest singular value).

        A large condition number indicates the system is ill-conditioned
        and small perturbations in **y** or **A** may cause large
        changes in the reconstructed **x**.

        Returns
        -------
        float
            ``cond(A)`` or ``np.inf`` if **A** has a zero singular value.
        """
        A = self.measurement_matrix()
        if A.size == 0:
            return np.inf
        svd_vals = np.linalg.svd(A, compute_uv=False)
        if svd_vals[-1] < 1e-15:
            return np.inf
        return float(svd_vals[0] / svd_vals[-1])

    def rank(self) -> int:
        """Numerical rank of **A**.

        Uses NumPy's default tolerance for determining near-zero
        singular values.

        Returns
        -------
        int
            Numerical rank.
        """
        A = self.measurement_matrix()
        if A.size == 0:
            return 0
        return int(np.linalg.matrix_rank(A))

    def mutual_coherence(self) -> float:
        """Maximum absolute off-diagonal entry of the normalized Gram matrix.

        The Gram matrix is ``G = (A_n)^T A_n`` where ``A_n`` has
        unit-norm columns.  Mutual coherence ``mu`` is:

        .. math::

            \\mu = \\max_{i \\neq j} |G_{ij}|

        A low coherence indicates the columns of **A** are nearly
        orthogonal, which benefits sparse recovery.

        Returns
        -------
        float
            Mutual coherence in ``[0, 1]``, or ``0.0`` for empty/
            single-column matrices.
        """
        A = self.measurement_matrix()
        n_cols = A.shape[1] if A.ndim == 2 else 0
        if A.size == 0 or n_cols <= 1:
            return 0.0

        # Normalize columns to unit L2 norm
        col_norms = np.linalg.norm(A, axis=0)
        # Guard against zero-norm columns
        col_norms = np.where(col_norms < 1e-15, 1.0, col_norms)
        A_normalized = A / col_norms[np.newaxis, :]

        gram = A_normalized.T @ A_normalized  # (S, S)

        # Zero out diagonal, then take max absolute value
        np.fill_diagonal(gram, 0.0)
        return float(np.max(np.abs(gram)))

    def restricted_isometry_proxy(self, k: int) -> float:
        """Approximate restricted isometry property (RIP) constant for sparsity *k*.

        The exact RIP constant ``delta_k`` is NP-hard to compute.  This
        method approximates it by sampling random *k*-sparse support sets
        and measuring how much the corresponding sub-matrix deviates from
        an isometry:

        .. math::

            \\delta_k \\approx \\max_S \\bigl|\\sigma_{\\max}^2(A_S) - 1\\bigr|

        over sampled support sets *S* of size *k*, where ``A_S`` is the
        column-normalized sub-matrix restricted to columns in *S*.

        Parameters
        ----------
        k : int
            Sparsity level (number of nonzero components).

        Returns
        -------
        float
            Proxy RIP constant in ``[0, inf)``.  Values close to zero
            indicate good isometry for *k*-sparse signals.
        """
        A = self.measurement_matrix()
        if A.size == 0:
            return 1.0
        n_rows, n_cols = A.shape
        k = min(k, n_cols)
        if k <= 0:
            return 0.0

        # Normalize columns
        col_norms = np.linalg.norm(A, axis=0)
        col_norms = np.where(col_norms < 1e-15, 1.0, col_norms)
        A_norm = A / col_norms[np.newaxis, :]

        rng = np.random.default_rng(42)
        n_trials = min(500, max(50, int(np.round(10 * n_cols / max(k, 1)))))
        max_delta = 0.0

        for _ in range(n_trials):
            support = rng.choice(n_cols, size=k, replace=False)
            A_sub = A_norm[:, support]
            svd_vals = np.linalg.svd(A_sub, compute_uv=False)
            # RIP deviation: max(|sigma^2 - 1|) over singular values
            deviations = np.abs(svd_vals ** 2 - 1.0)
            max_delta = max(max_delta, float(np.max(deviations)))

        return max_delta

    def sensitivity_analysis(
        self,
        perturbation_scale: float = 0.1,
    ) -> dict:
        """Measure how perturbations in **A** affect the OLS solution.

        Adds Gaussian noise of scale ``perturbation_scale * ||A||_F``
        to **A** and solves the least-squares problem with a synthetic
        observation vector.  Repeats over multiple noise realizations.

        Parameters
        ----------
        perturbation_scale : float
            Relative perturbation magnitude (default 0.1, i.e. 10 %).

        Returns
        -------
        dict
            Keys:

            * ``mean_relative_change`` -- average ``||x_pert - x_clean|| / ||x_clean||``
            * ``max_relative_change``  -- worst-case relative change
            * ``perturbation_scale``   -- echo of input
            * ``n_trials``             -- number of noise realizations
        """
        A = self.measurement_matrix()
        if A.size == 0:
            return {
                "mean_relative_change": 0.0,
                "max_relative_change": 0.0,
                "perturbation_scale": perturbation_scale,
                "n_trials": 0,
            }

        n_rows, n_cols = A.shape
        rng = np.random.default_rng(42)

        # Synthetic y from a sparse ground-truth x
        x_true = np.zeros(n_cols)
        n_nonzero = max(1, n_cols // 3)
        x_true[:n_nonzero] = rng.standard_normal(n_nonzero)
        y = A @ x_true

        # Solve clean OLS: x_clean = A^+ y
        x_clean = _ols_solve(A, y)
        norm_clean = np.linalg.norm(x_clean)
        if norm_clean < 1e-15:
            return {
                "mean_relative_change": 0.0,
                "max_relative_change": 0.0,
                "perturbation_scale": perturbation_scale,
                "n_trials": 0,
            }

        frob_norm = np.linalg.norm(A, "fro")
        n_trials = 50
        relative_changes: List[float] = []

        for _ in range(n_trials):
            noise = rng.standard_normal(A.shape) * perturbation_scale * frob_norm
            A_pert = A + noise
            x_pert = _ols_solve(A_pert, y)
            rel_change = float(np.linalg.norm(x_pert - x_clean) / norm_clean)
            relative_changes.append(rel_change)

        return {
            "mean_relative_change": float(np.mean(relative_changes)),
            "max_relative_change": float(np.max(relative_changes)),
            "perturbation_scale": perturbation_scale,
            "n_trials": n_trials,
        }

    def identifiability_report(self) -> dict:
        """Comprehensive identifiability diagnostic for the tomography system.

        Returns
        -------
        dict
            Keys:

            * ``rank``              -- numerical rank of **A**
            * ``condition_number``  -- condition number of **A**
            * ``mutual_coherence``  -- mutual coherence of **A**
            * ``n_targets``         -- number of rows (targets)
            * ``n_spectators``      -- number of columns (spectators)
            * ``is_well_posed``     -- ``True`` if condition number < 100
            * ``recommendations``   -- list of human-readable advice strings
        """
        cond = self.condition_number()
        rnk = self.rank()
        mc = self.mutual_coherence()

        A = self.measurement_matrix()
        n_targets = A.shape[0] if A.ndim == 2 else 0
        n_spectators = A.shape[1] if A.ndim == 2 else 0

        is_well_posed = cond < 100.0

        recommendations: List[str] = []

        if cond == np.inf:
            recommendations.append(
                "Matrix is rank-deficient; some spectator effects are not "
                "identifiable.  Consider adding more target observations or "
                "removing collinear spectators."
            )
        elif cond >= 100.0:
            recommendations.append(
                f"High condition number ({cond:.1f}); the system is "
                "ill-conditioned.  Use regularized reconstruction (L1 or "
                "nonneg-L1) rather than OLS."
            )

        if rnk < min(n_targets, n_spectators):
            recommendations.append(
                f"Rank ({rnk}) is less than min(T, S) = "
                f"{min(n_targets, n_spectators)}.  The system has a "
                "non-trivial null space; not all spectator coefficients "
                "are uniquely recoverable."
            )

        if n_targets < n_spectators:
            recommendations.append(
                f"Under-determined system (T={n_targets} < S={n_spectators}). "
                "L1 regularization is recommended for sparse recovery."
            )

        if mc > 0.9:
            recommendations.append(
                f"High mutual coherence ({mc:.3f}); spectator columns are "
                "highly correlated.  Sparse recovery guarantees are weak."
            )
        elif mc > 0.5:
            recommendations.append(
                f"Moderate mutual coherence ({mc:.3f}); sparse recovery "
                "may require more observations for reliable results."
            )

        if not recommendations:
            recommendations.append(
                "System appears well-posed.  OLS and L1 reconstruction "
                "should both perform reliably."
            )

        return {
            "rank": rnk,
            "condition_number": cond,
            "mutual_coherence": mc,
            "n_targets": n_targets,
            "n_spectators": n_spectators,
            "is_well_posed": is_well_posed,
            "recommendations": recommendations,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_solve(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least-squares via the pseudo-inverse (robust to rank deficiency).

    Returns x_hat = A^+ y.
    """
    if A.size == 0 or y.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)
    # lstsq handles rank-deficient A gracefully
    x_hat, _residuals, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    return x_hat


def _step_size(A: np.ndarray) -> float:
    """Compute the ISTA step size: 1 / ||A^T A||_2 = 1 / sigma_max(A)^2.

    Returns a safe step size that guarantees convergence.
    """
    if A.size == 0:
        return 1.0
    svd_vals = np.linalg.svd(A, compute_uv=False)
    L = float(svd_vals[0]) ** 2  # Lipschitz constant = largest eigenvalue of A^T A
    if L < 1e-15:
        return 1.0
    return 1.0 / L


# ---------------------------------------------------------------------------
# L1-regularized reconstruction (ISTA)
# ---------------------------------------------------------------------------

def l1_reconstruct(
    y: np.ndarray,
    A: np.ndarray,
    lambda_l1: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """L1-regularized (LASSO) reconstruction via ISTA.

    Solves the optimization problem:

    .. math::

        \\min_{\\mathbf{x}} \\;
        \\tfrac{1}{2}\\|\\mathbf{y} - \\mathbf{A}\\mathbf{x}\\|_2^2
        + \\lambda \\|\\mathbf{x}\\|_1

    using the Iterative Shrinkage-Thresholding Algorithm (ISTA):

    .. math::

        \\mathbf{x}^{(k+1)} = S_{\\lambda t}\\!
        \\bigl(\\mathbf{x}^{(k)} + t\\,\\mathbf{A}^T
        (\\mathbf{y} - \\mathbf{A}\\mathbf{x}^{(k)})\\bigr)

    where ``t`` is the step size (inverse of the Lipschitz constant of
    the gradient) and ``S_tau`` is soft-thresholding.

    Parameters
    ----------
    y : np.ndarray
        Observation vector of length *m*.
    A : np.ndarray
        Measurement matrix of shape ``(m, n)``.
    lambda_l1 : float
        L1 regularization strength.
    max_iter : int
        Maximum number of ISTA iterations (default 1000).
    tol : float
        Convergence tolerance on ``||x^{k+1} - x^k||`` (default 1e-6).

    Returns
    -------
    np.ndarray
        Reconstructed coefficient vector of length *n*.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    if A.size == 0 or y.size == 0:
        n = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(n)

    m, n = A.shape
    step = _step_size(A)
    x = np.zeros(n)

    At = A.T  # precompute transpose
    threshold = lambda_l1 * step

    for _ in range(max_iter):
        residual = y - A @ x
        gradient_step = x + step * (At @ residual)
        x_new = _soft_threshold(gradient_step, threshold)

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x


# ---------------------------------------------------------------------------
# Group LASSO reconstruction
# ---------------------------------------------------------------------------

def group_lasso_reconstruct(
    y: np.ndarray,
    A: np.ndarray,
    groups: List[List[int]],
    lambda_gl: float,
    max_iter: int = 1000,
) -> np.ndarray:
    """Group LASSO reconstruction for per-target spectator groups.

    Solves:

    .. math::

        \\min_{\\mathbf{x}} \\;
        \\tfrac{1}{2}\\|\\mathbf{y} - \\mathbf{A}\\mathbf{x}\\|_2^2
        + \\lambda \\sum_{g \\in \\mathcal{G}} \\|\\mathbf{x}_g\\|_2

    via proximal gradient descent.  The proximal operator for the group
    L2 penalty is the block soft-thresholding:

    .. math::

        \\text{prox}_{\\lambda\\|\\cdot\\|_2}(\\mathbf{z}_g) =
        \\mathbf{z}_g \\cdot \\max\\!\\bigl(0,\\;
        1 - \\tfrac{\\lambda}{\\|\\mathbf{z}_g\\|_2}\\bigr)

    Parameters
    ----------
    y : np.ndarray
        Observation vector of length *m*.
    A : np.ndarray
        Measurement matrix of shape ``(m, n)``.
    groups : list of list of int
        Each sub-list contains column indices belonging to one group.
        Groups may be non-overlapping; indices not in any group are
        treated as ungrouped (individual L1 penalty).
    lambda_gl : float
        Group regularization strength.
    max_iter : int
        Maximum iterations (default 1000).

    Returns
    -------
    np.ndarray
        Reconstructed coefficient vector of length *n*.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    if A.size == 0 or y.size == 0:
        n = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(n)

    m, n = A.shape
    step = _step_size(A)
    x = np.zeros(n)

    At = A.T
    tol = 1e-6

    # Build set of grouped indices for fast lookup
    grouped_indices = set()
    for g in groups:
        grouped_indices.update(g)

    # Ungrouped indices get individual L1 penalty
    ungrouped = [i for i in range(n) if i not in grouped_indices]

    for _ in range(max_iter):
        residual = y - A @ x
        gradient_step = x + step * (At @ residual)

        x_new = gradient_step.copy()

        # Block soft-threshold for each group
        for g in groups:
            if len(g) == 0:
                continue
            g_idx = np.array(g)
            z_g = gradient_step[g_idx]
            norm_g = np.linalg.norm(z_g)
            if norm_g < 1e-15:
                x_new[g_idx] = 0.0
            else:
                shrink = max(0.0, 1.0 - (lambda_gl * step) / norm_g)
                x_new[g_idx] = z_g * shrink

        # Individual L1 for ungrouped coefficients
        if ungrouped:
            u_idx = np.array(ungrouped)
            x_new[u_idx] = _soft_threshold(
                gradient_step[u_idx], lambda_gl * step
            )

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x


# ---------------------------------------------------------------------------
# Non-negative L1 reconstruction
# ---------------------------------------------------------------------------

def nonneg_l1_reconstruct(
    y: np.ndarray,
    A: np.ndarray,
    lambda_l1: float,
    max_iter: int = 1000,
) -> np.ndarray:
    """L1-regularized reconstruction with non-negativity constraint.

    Same objective as :func:`l1_reconstruct` but with the additional
    constraint ``x >= 0``.  After each ISTA soft-thresholding step the
    solution is projected onto the non-negative orthant.

    Parameters
    ----------
    y : np.ndarray
        Observation vector.
    A : np.ndarray
        Measurement matrix.
    lambda_l1 : float
        L1 regularization strength.
    max_iter : int
        Maximum iterations (default 1000).

    Returns
    -------
    np.ndarray
        Non-negative reconstructed coefficient vector.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    if A.size == 0 or y.size == 0:
        n = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(n)

    m, n = A.shape
    step = _step_size(A)
    x = np.zeros(n)

    At = A.T
    threshold = lambda_l1 * step
    tol = 1e-6

    for _ in range(max_iter):
        residual = y - A @ x
        gradient_step = x + step * (At @ residual)
        x_new = _soft_threshold(gradient_step, threshold)
        # Project onto non-negative orthant
        np.maximum(x_new, 0.0, out=x_new)

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x


# ---------------------------------------------------------------------------
# Reconstruction diagnostics
# ---------------------------------------------------------------------------

def reconstruction_diagnostics(
    y: np.ndarray,
    A: np.ndarray,
    x_hat: np.ndarray,
) -> dict:
    """Diagnostic statistics for a reconstruction.

    Parameters
    ----------
    y : np.ndarray
        Observation vector.
    A : np.ndarray
        Measurement matrix.
    x_hat : np.ndarray
        Reconstructed coefficient vector.

    Returns
    -------
    dict
        Keys:

        * ``residual_norm``   -- ``||y - A x_hat||_2``
        * ``relative_error``  -- ``||y - A x_hat||_2 / ||y||_2``
        * ``sparsity_ratio``  -- fraction of entries with ``|x_hat| < 1e-6``
        * ``signal_to_noise`` -- ``||A x_hat||_2 / ||y - A x_hat||_2``
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()

    if A.size == 0 or y.size == 0 or x_hat.size == 0:
        return {
            "residual_norm": 0.0,
            "relative_error": 0.0,
            "sparsity_ratio": 1.0,
            "signal_to_noise": 0.0,
        }

    residual = y - A @ x_hat
    residual_norm = float(np.linalg.norm(residual))
    y_norm = float(np.linalg.norm(y))
    signal_norm = float(np.linalg.norm(A @ x_hat))

    relative_error = residual_norm / y_norm if y_norm > 1e-15 else 0.0
    snr = signal_norm / residual_norm if residual_norm > 1e-15 else np.inf

    n = len(x_hat)
    n_near_zero = int(np.sum(np.abs(x_hat) < 1e-6))
    sparsity_ratio = n_near_zero / n if n > 0 else 1.0

    return {
        "residual_norm": residual_norm,
        "relative_error": float(relative_error),
        "sparsity_ratio": float(sparsity_ratio),
        "signal_to_noise": float(snr),
    }


# ---------------------------------------------------------------------------
# Comparison of reconstructions
# ---------------------------------------------------------------------------

def compare_reconstructions(
    tomo_mean_df: pd.DataFrame,
    tomo_stderr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare OLS, L1, and nonneg-L1 reconstructions of the tomography.

    Constructs the measurement matrix from the mean/stderr DataFrames,
    generates a synthetic observation vector (row-wise sum of the mean
    matrix), and solves with three methods.

    Parameters
    ----------
    tomo_mean_df : pd.DataFrame
        Mean tomography matrix (targets x spectators).
    tomo_stderr_df : pd.DataFrame
        Standard-error matrix (same shape).

    Returns
    -------
    pd.DataFrame
        One row per method (``'OLS'``, ``'L1'``, ``'nonneg_L1'``) with
        columns: ``method``, ``mse``, ``sparsity_ratio``,
        ``condition_number``, ``residual_norm``, ``relative_error``.
    """
    model = TomographyModel(tomo_mean_df, tomo_stderr_df)
    A = model.measurement_matrix()

    if A.size == 0:
        return pd.DataFrame(
            columns=[
                "method", "mse", "sparsity_ratio",
                "condition_number", "residual_norm", "relative_error",
            ]
        )

    cond = model.condition_number()

    # Observation vector: per-target aggregate interference (row sums)
    y = np.sum(A, axis=1)

    # Choose a sensible lambda based on the data scale
    lambda_l1 = 0.1 * np.max(np.abs(A.T @ y)) if y.size > 0 else 0.01

    # --- OLS ---
    x_ols = _ols_solve(A, y)

    # --- L1 (ISTA) ---
    x_l1 = l1_reconstruct(y, A, lambda_l1)

    # --- Nonneg L1 ---
    x_nn = nonneg_l1_reconstruct(y, A, lambda_l1)

    rows = []
    for name, x_hat in [("OLS", x_ols), ("L1", x_l1), ("nonneg_L1", x_nn)]:
        diag = reconstruction_diagnostics(y, A, x_hat)
        mse = float(np.mean((y - A @ x_hat) ** 2))
        rows.append({
            "method": name,
            "mse": mse,
            "sparsity_ratio": diag["sparsity_ratio"],
            "condition_number": cond,
            "residual_norm": diag["residual_norm"],
            "relative_error": diag["relative_error"],
        })

    return pd.DataFrame(rows)

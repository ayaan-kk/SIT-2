"""Bootstrap confidence intervals for interference tomography.

Provides cell-level and full-matrix bootstrapping of treatment effects
from IRBS trial data.  Resamples control and treatment trial indices
independently with replacement, computes the per-resample metric
difference, and returns percentile-based confidence intervals.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def bootstrap_cell(
    trial_types,
    per_trial_summaries: Dict[str, np.ndarray],
    metric: str = "p99",
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, float]:
    """Bootstrap a single cell of the tomography matrix.

    Separates trials into control and treatment groups, then repeatedly
    resamples each group with replacement to build a distribution of
    the treatment effect (mean_treatment - mean_control).

    Args:
        trial_types: array-like of ``'control'`` / ``'treatment'`` strings,
            one per trial.
        per_trial_summaries: dict mapping metric names (e.g. ``'p99'``,
            ``'cvar'``, ``'mean'``) to arrays of per-trial values.
        metric: which metric key to use for effect computation.
        n_bootstrap: number of bootstrap resamples (default 2000).
        rng: numpy random generator; created with default seed if *None*.

    Returns:
        ``(mean_effect, ci_lo, ci_hi, std_err)`` where *ci_lo* / *ci_hi*
        are the 2.5-th and 97.5-th percentiles of the bootstrap
        distribution, and *std_err* is the standard deviation of the
        bootstrap effect estimates.
    """
    if rng is None:
        rng = np.random.default_rng()

    trial_types = np.asarray(trial_types)
    metric_values = np.asarray(per_trial_summaries[metric], dtype=np.float64)

    ctrl_idx = np.where(trial_types == "control")[0]
    treat_idx = np.where(trial_types == "treatment")[0]

    if len(ctrl_idx) == 0 or len(treat_idx) == 0:
        return 0.0, 0.0, 0.0, 0.0

    ctrl_values = metric_values[ctrl_idx]
    treat_values = metric_values[treat_idx]

    n_ctrl = len(ctrl_values)
    n_treat = len(treat_values)

    # Vectorised bootstrap: resample indices with replacement and compute
    # per-resample means in bulk.
    boot_ctrl_idx = rng.integers(0, n_ctrl, size=(n_bootstrap, n_ctrl))
    boot_treat_idx = rng.integers(0, n_treat, size=(n_bootstrap, n_treat))

    boot_ctrl_means = ctrl_values[boot_ctrl_idx].mean(axis=1)
    boot_treat_means = treat_values[boot_treat_idx].mean(axis=1)

    effects = boot_treat_means - boot_ctrl_means

    mean_effect = float(np.mean(effects))
    ci_lo = float(np.percentile(effects, 2.5))
    ci_hi = float(np.percentile(effects, 97.5))
    std_err = float(np.std(effects, ddof=1))

    return mean_effect, ci_lo, ci_hi, std_err


def bootstrap_all_cells(
    irbs_results_dict: dict,
    metric: str = "p99",
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
    """Bootstrap every cell in an IRBS results dictionary.

    Iterates over all ``(target_name, spectator_name)`` entries and calls
    :func:`bootstrap_cell` for each.

    Args:
        irbs_results_dict: dict keyed by ``(target_name, spectator_name)``.
            Each value must be a dict with keys ``'trial_types'`` (array of
            ``'control'``/``'treatment'``) and ``'per_trial_summaries'``
            (dict of metric-name -> array).
        metric: metric to bootstrap (default ``'p99'``).
        n_bootstrap: number of bootstrap resamples per cell.
        rng: numpy random generator; created if *None*.

    Returns:
        Dict mapping ``(target, spectator)`` to
        ``(mean, ci_lo, ci_hi, std_err)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    results: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}

    for (target, spectator), cell_data in irbs_results_dict.items():
        trial_types = cell_data["trial_types"]
        per_trial_summaries = cell_data["per_trial_summaries"]

        # Use a child RNG so that cell ordering does not affect results
        child_rng = np.random.default_rng(rng.integers(0, 2**63))

        results[(target, spectator)] = bootstrap_cell(
            trial_types,
            per_trial_summaries,
            metric=metric,
            n_bootstrap=n_bootstrap,
            rng=child_rng,
        )

    return results

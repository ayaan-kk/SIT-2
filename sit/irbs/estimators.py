"""IRBS estimators for treatment effect estimation.

Provides point estimates and bootstrap confidence intervals for the
average treatment effect (ATE) of interference on tail-latency metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


SUPPORTED_METRICS = ("mean", "p95", "p99", "cvar95", "cvar99", "slo_violation_rate")


def compute_effect(
    trial_types: np.ndarray,
    per_trial_summaries: pd.DataFrame,
    metric: str = "p99",
) -> float:
    """Compute the estimated treatment effect (tau_hat).

    tau_hat = mean(metric | Z=1) - mean(metric | Z=0)

    A positive tau_hat indicates that the treatment (spectator present)
    *increases* the metric value relative to control.

    Args:
        trial_types: int array of shape (n_trials,), 0=control 1=treatment.
        per_trial_summaries: DataFrame with at least the ``metric`` column.
        metric: one of the supported metric names.

    Returns:
        tau_hat as a float.

    Raises:
        ValueError: if metric is not supported or there are no control/treatment
            trials.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported: {SUPPORTED_METRICS}"
        )

    trial_types = np.asarray(trial_types)
    control_mask = trial_types == 0
    treatment_mask = trial_types == 1

    if not np.any(control_mask):
        raise ValueError("No control trials (Z=0) found.")
    if not np.any(treatment_mask):
        raise ValueError("No treatment trials (Z=1) found.")

    metric_values = per_trial_summaries[metric].values
    mean_treatment = float(np.mean(metric_values[treatment_mask]))
    mean_control = float(np.mean(metric_values[control_mask]))

    return mean_treatment - mean_control


def bootstrap_effect_ci(
    trial_types: np.ndarray,
    per_trial_summaries: pd.DataFrame,
    metric: str = "p99",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the treatment effect.

    Resamples control and treatment trials separately (preserving group sizes),
    computes tau_hat on each bootstrap replicate, and returns the percentile CI.

    Args:
        trial_types: int array of shape (n_trials,), 0=control 1=treatment.
        per_trial_summaries: DataFrame with at least the ``metric`` column.
        metric: one of the supported metric names.
        n_bootstrap: number of bootstrap resamples.
        alpha: significance level (default 0.05 for a 95 % CI).
        rng: numpy random Generator.  If None, a default is created.

    Returns:
        (tau_hat, ci_lower, ci_upper)
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported: {SUPPORTED_METRICS}"
        )

    if rng is None:
        rng = np.random.default_rng()

    trial_types = np.asarray(trial_types)
    metric_values = per_trial_summaries[metric].values

    control_idx = np.where(trial_types == 0)[0]
    treatment_idx = np.where(trial_types == 1)[0]

    if len(control_idx) == 0:
        raise ValueError("No control trials (Z=0) found.")
    if len(treatment_idx) == 0:
        raise ValueError("No treatment trials (Z=1) found.")

    control_values = metric_values[control_idx]
    treatment_values = metric_values[treatment_idx]

    # Point estimate
    tau_hat = float(np.mean(treatment_values) - np.mean(control_values))

    # Bootstrap resampling
    boot_taus = np.empty(n_bootstrap)
    n_ctrl = len(control_values)
    n_treat = len(treatment_values)

    for b in range(n_bootstrap):
        boot_ctrl = control_values[rng.integers(0, n_ctrl, size=n_ctrl)]
        boot_treat = treatment_values[rng.integers(0, n_treat, size=n_treat)]
        boot_taus[b] = np.mean(boot_treat) - np.mean(boot_ctrl)

    ci_lower = float(np.percentile(boot_taus, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_taus, 100 * (1.0 - alpha / 2)))

    return tau_hat, ci_lower, ci_upper

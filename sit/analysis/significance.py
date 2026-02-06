"""Statistical significance tests for SIT results."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(values[idx])

    mean_val = float(np.mean(values))
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean_val, ci_lower, ci_upper


def bootstrap_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for difference of means (A - B).

    Returns (diff, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    na = len(values_a)
    nb = len(values_b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0

    boot_diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx_a = rng.integers(0, na, size=na)
        idx_b = rng.integers(0, nb, size=nb)
        boot_diffs[b] = np.mean(values_a[idx_a]) - np.mean(values_b[idx_b])

    diff = float(np.mean(values_a) - np.mean(values_b))
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return diff, ci_lower, ci_upper


def compute_scheduler_significance(
    results_df: pd.DataFrame,
    sit_scheduler: str = "sit_dpp",
    baseline_scheduler: str = "random",
    metric: str = "p99",
    group_cols: Optional[list] = None,
    n_bootstrap: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Compute significance of SIT improvement over baseline.

    Returns DataFrame with group-level significance results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if group_cols is None:
        group_cols = ["regime"]

    sit_data = results_df[results_df["scheduler"] == sit_scheduler]
    base_data = results_df[results_df["scheduler"] == baseline_scheduler]

    rows = []
    for group_vals, sit_group in sit_data.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        # Find matching baseline group
        mask = pd.Series(True, index=base_data.index)
        for col, val in zip(group_cols, group_vals):
            mask &= base_data[col] == val
        base_group = base_data[mask]

        if len(sit_group) == 0 or len(base_group) == 0:
            continue

        diff, ci_lo, ci_hi = bootstrap_difference_ci(
            base_group[metric].values,
            sit_group[metric].values,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

        row = dict(zip(group_cols, group_vals))
        row.update({
            f"baseline_{metric}_mean": float(base_group[metric].mean()),
            f"sit_{metric}_mean": float(sit_group[metric].mean()),
            "improvement": diff,
            "improvement_ci_lower": ci_lo,
            "improvement_ci_upper": ci_hi,
            "significant": ci_lo > 0,  # significant if CI doesn't cross 0
            "improvement_pct": diff / float(base_group[metric].mean()) * 100
            if base_group[metric].mean() > 0 else 0.0,
        })
        rows.append(row)

    return pd.DataFrame(rows)

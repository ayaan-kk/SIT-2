"""Rigorous statistical testing: effect sizes, FDR correction, block bootstrap.

All routines are implemented with **numpy only** (no scipy dependency).

Effect-size measures
--------------------
- Cohen's d:  standardised mean difference, (M_a - M_b) / s_pooled
- Cliff's delta: non-parametric ordinal effect size in [-1, 1]

Bootstrap
---------
- Circular block bootstrap for autocorrelated / time-series data
- Paired permutation test for matched-seed comparisons

Multiple testing
----------------
- Benjamini-Hochberg FDR correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d (standardised mean difference).

    d = (mean(a) - mean(b)) / s_pooled

    where s_pooled = sqrt(((n_a-1)*var(a) + (n_b-1)*var(b)) / (n_a+n_b-2))

    Positive d means group_a has a larger mean than group_b.

    Parameters
    ----------
    group_a, group_b : array-like
        Two independent samples.

    Returns
    -------
    float
        Cohen's d.  Returns 0.0 when pooled std is zero.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)

    if na < 2 or nb < 2:
        return 0.0

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    s_pooled = np.sqrt(pooled_var)

    if s_pooled == 0.0:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / s_pooled)


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, str]:
    """Compute Cliff's delta (non-parametric ordinal effect size).

    delta = (#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)

    Magnitude thresholds (Romano et al. 2006):
        |delta| < 0.147  -> negligible
        |delta| < 0.33   -> small
        |delta| < 0.474  -> medium
        else              -> large

    Parameters
    ----------
    group_a, group_b : array-like
        Two independent samples.

    Returns
    -------
    (float, str)
        The effect size and its magnitude label.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)

    if na == 0 or nb == 0:
        return 0.0, "negligible"

    # Vectorised pairwise comparison using broadcasting
    # a[:, None] has shape (na, 1), b[None, :] has shape (1, nb)
    more = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    delta = float((more - less) / (na * nb))

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = 10,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Circular block bootstrap confidence interval for the mean.

    The circular block bootstrap (Politis & Romano 1992) handles
    autocorrelation by resampling contiguous blocks of observations.
    The series is treated as circular: a block starting near the end
    wraps around to the beginning.

    Parameters
    ----------
    values : array-like
        1-D array of observations (may be autocorrelated).
    block_size : int
        Length of each resampled block.
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (default 0.05 -> 95 % CI).
    rng : np.random.Generator or None
        Random number generator (uses default_rng(42) if None).

    Returns
    -------
    (mean, ci_lo, ci_hi)
        Point estimate and percentile-based confidence bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        v = float(values[0])
        return v, v, v

    # Effective block size capped at n
    bs = min(block_size, n)
    # Number of blocks needed to cover n observations
    n_blocks = int(np.ceil(n / bs))

    boot_means = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Draw n_blocks random starting positions (circular)
        starts = rng.integers(0, n, size=n_blocks)
        # Build the resampled series by concatenating circular blocks
        indices = np.concatenate([np.arange(s, s + bs) % n for s in starts])
        # Trim to exactly n observations
        indices = indices[:n]
        boot_means[b] = np.mean(values[indices])

    mean_val = float(np.mean(values))
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean_val, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Controls the *false discovery rate* at level *alpha*.

    Algorithm:
    1. Sort p-values in ascending order.
    2. For rank *k* (1-based), compute threshold = k / m * alpha.
    3. Find the largest k such that p_(k) <= threshold.
    4. Reject all hypotheses with rank <= k.

    Parameters
    ----------
    p_values : array-like
        Raw (uncorrected) p-values, one per hypothesis.
    alpha : float
        Desired FDR level (default 0.05).

    Returns
    -------
    np.ndarray
        Boolean array of same length as *p_values*; True = rejected.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if m == 0:
        return np.array([], dtype=bool)

    # Sort indices
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH thresholds: (rank / m) * alpha   (1-based rank)
    ranks = np.arange(1, m + 1)
    thresholds = ranks / m * alpha

    # Find largest k where p_(k) <= threshold_k
    below = sorted_p <= thresholds
    if not np.any(below):
        return np.zeros(m, dtype=bool)

    k_max = np.max(np.where(below)[0])  # 0-based index of largest qualifying rank

    # Reject all with rank <= k_max+1
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[:k_max + 1]] = True

    return rejected


# ---------------------------------------------------------------------------
# Effect size summary table
# ---------------------------------------------------------------------------

def compute_effect_size_table(
    sched_df: pd.DataFrame,
    sit_scheduler: str = "sit_dpp",
    n_bootstrap: int = 5000,
    block_size: int = 10,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Compute effect sizes of SIT vs. every other scheduler.

    For each baseline scheduler, computes Cohen's d, Cliff's delta,
    and block-bootstrap CI on both p99 and cvar99.

    Parameters
    ----------
    sched_df : pd.DataFrame
        Full scheduling results with ``scheduler``, ``p99``, ``cvar99``.
    sit_scheduler : str
        Name of the SIT scheduler to compare against baselines.
    n_bootstrap : int
        Bootstrap replicates for CIs.
    block_size : int
        Block size for circular block bootstrap.
    alpha : float
        CI significance level.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    pd.DataFrame
        One row per baseline scheduler with columns:
        baseline, cohens_d_p99, cliffs_delta_p99, cliffs_magnitude_p99,
        p99_ci_lo, p99_ci_hi, cohens_d_cvar99, cliffs_delta_cvar99,
        cliffs_magnitude_cvar99, cvar99_ci_lo, cvar99_ci_hi.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sit_data = sched_df[sched_df["scheduler"] == sit_scheduler]
    other_schedulers = [
        s for s in sched_df["scheduler"].unique() if s != sit_scheduler
    ]

    rows: List[Dict] = []
    for baseline in sorted(other_schedulers):
        base_data = sched_df[sched_df["scheduler"] == baseline]
        row: Dict = {"baseline": baseline}

        for metric in ("p99", "cvar99"):
            sit_vals = sit_data[metric].values
            base_vals = base_data[metric].values

            # Cohen's d: positive means baseline has larger (worse) metric
            cd = cohens_d(base_vals, sit_vals)
            row[f"cohens_d_{metric}"] = cd

            # Cliff's delta
            delta, mag = cliffs_delta(base_vals, sit_vals)
            row[f"cliffs_delta_{metric}"] = delta
            row[f"cliffs_magnitude_{metric}"] = mag

            # Bootstrap CI on the *difference* (baseline - sit)
            diff = base_vals.mean() - sit_vals.mean() if len(base_vals) > 0 and len(sit_vals) > 0 else 0.0
            # Block-bootstrap CI on baseline values to quantify uncertainty
            _, ci_lo, ci_hi = block_bootstrap_ci(
                base_vals - np.mean(sit_vals) if len(sit_vals) > 0 else base_vals,
                block_size=block_size,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                rng=rng,
            )
            row[f"{metric}_diff_mean"] = diff
            row[f"{metric}_ci_lo"] = ci_lo
            row[f"{metric}_ci_hi"] = ci_hi

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# p99 stability analysis
# ---------------------------------------------------------------------------

def compute_p99_stability(
    samples_per_condition: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Assess whether the p99 estimate is stable given the sample size.

    Uses bootstrap to compute a CI for the 99th percentile.  The
    estimate is deemed "stable" if the CI width is less than 20 % of
    the point estimate.

    Parameters
    ----------
    samples_per_condition : array-like
        1-D array of latency samples for one condition.
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level for the CI.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    dict
        Keys: n_samples, p99_estimate, ci_lo, ci_hi, ci_width,
        ci_width_pct, is_stable, min_samples_heuristic.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    samples = np.asarray(samples_per_condition, dtype=float)
    n = len(samples)

    if n == 0:
        return {
            "n_samples": 0,
            "p99_estimate": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "ci_width": 0.0,
            "ci_width_pct": np.inf,
            "is_stable": False,
            "min_samples_heuristic": _heuristic_min_samples_p99(),
        }

    p99_est = float(np.percentile(samples, 99))

    # Bootstrap the p99
    boot_p99 = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_p99[b] = np.percentile(samples[idx], 99)

    ci_lo = float(np.percentile(boot_p99, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_p99, 100 * (1 - alpha / 2)))
    ci_width = ci_hi - ci_lo

    ci_width_pct = ci_width / p99_est if p99_est > 0 else np.inf
    is_stable = ci_width_pct < 0.20

    return {
        "n_samples": n,
        "p99_estimate": p99_est,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_width": ci_width,
        "ci_width_pct": float(ci_width_pct),
        "is_stable": is_stable,
        "min_samples_heuristic": _heuristic_min_samples_p99(),
    }


def _heuristic_min_samples_p99() -> int:
    """Rule-of-thumb minimum samples for a stable p99.

    For the 99th percentile, at least ~1/(1-0.99) = 100 observations
    are needed to have *any* exceedance; practical stability typically
    requires ~10x that.  We return 1000 as a conservative heuristic.
    """
    return 1000


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    paired_diffs: np.ndarray,
    n_permutations: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Two-sided paired permutation test.

    Under H_0 the paired differences are exchangeable (equally likely
    to be positive or negative).  We randomly flip signs and compute
    the test statistic (mean of differences) for each permutation.

    Parameters
    ----------
    paired_diffs : array-like
        Observed paired differences (e.g. p99_baseline - p99_sit for
        the same seed / condition).
    n_permutations : int
        Number of random sign-flips.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    float
        Two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    diffs = np.asarray(paired_diffs, dtype=float)
    n = len(diffs)

    if n == 0:
        return 1.0

    observed_stat = abs(np.mean(diffs))

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        perm_stat = abs(np.mean(diffs * signs))
        if perm_stat >= observed_stat:
            count_extreme += 1

    return float((count_extreme + 1) / (n_permutations + 1))


# ---------------------------------------------------------------------------
# Minimum sample size for p99
# ---------------------------------------------------------------------------

def minimum_sample_size_p99(
    target_ci_width_pct: float = 0.2,
    alpha: float = 0.01,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Estimate the minimum sample size for a stable p99 via simulation.

    Draws increasingly large samples from a synthetic heavy-tailed
    distribution (log-normal, which mimics latency) and uses bootstrap
    to find the smallest *n* where the p99 CI width is less than
    *target_ci_width_pct* of the p99 estimate.

    The search uses an exponential grid: n in {100, 200, 400, 800, ...}
    up to 100 000, then refines with binary search.

    Parameters
    ----------
    target_ci_width_pct : float
        Maximum acceptable CI width as a fraction of the p99 estimate.
    alpha : float
        Significance level for the CI.
    n_bootstrap : int
        Bootstrap replicates per trial.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    int
        Estimated minimum sample size.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Synthetic heavy-tailed latency distribution: log-normal(mu=3, sigma=0.8)
    # This gives realistic tail behaviour for latency data.
    mu, sigma = 3.0, 0.8

    def _ci_width_pct_at_n(n: int) -> float:
        """Compute relative CI width for p99 at sample size n."""
        samples = rng.lognormal(mu, sigma, size=n)
        p99 = np.percentile(samples, 99)
        if p99 <= 0:
            return np.inf

        boot_p99 = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_p99[b] = np.percentile(samples[idx], 99)

        ci_lo = np.percentile(boot_p99, 100 * alpha / 2)
        ci_hi = np.percentile(boot_p99, 100 * (1 - alpha / 2))
        return float((ci_hi - ci_lo) / p99)

    # Phase 1: exponential grid search
    candidates = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000]
    lo, hi = candidates[0], candidates[-1]

    for n_cand in candidates:
        width_pct = _ci_width_pct_at_n(n_cand)
        if width_pct < target_ci_width_pct:
            hi = n_cand
            break
    else:
        # Even 100k wasn't enough; return that as the estimate.
        return candidates[-1]

    # Find the largest candidate below hi that did NOT satisfy the criterion
    for n_cand in candidates:
        if n_cand >= hi:
            break
        width_pct = _ci_width_pct_at_n(n_cand)
        if width_pct >= target_ci_width_pct:
            lo = n_cand

    # Phase 2: binary search between lo and hi
    for _ in range(8):  # 8 iterations gives precision ~(hi-lo)/256
        if hi - lo <= 50:
            break
        mid = (lo + hi) // 2
        width_pct = _ci_width_pct_at_n(mid)
        if width_pct < target_ci_width_pct:
            hi = mid
        else:
            lo = mid

    return hi

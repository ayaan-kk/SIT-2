"""Bootstrap confidence interval coverage validation.

Simulates known ground truth from a large experiment and checks whether
bootstrap CIs computed from smaller experiments achieve their nominal
coverage rate.  Two bootstrap methods are compared: standard IID bootstrap
and circular block bootstrap.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sit.simulator.workloads import get_targets, get_spectators
from sit.simulator.device_profiles import get_device_profiles
from sit.experiments.run_experiments import run_irbs_condition
from sit.analysis.significance import bootstrap_mean_ci


# ---------------------------------------------------------------------------
# Helper: effective sample size
# ---------------------------------------------------------------------------

def compute_effective_sample_size(values: np.ndarray) -> float:
    """Estimate effective sample size using autocorrelation.

    Uses the initial positive sequence estimator: autocorrelations are
    summed until the first non-positive lag, giving

        ESS = n / (1 + 2 * sum_of_positive_autocorrelations)

    Parameters
    ----------
    values : np.ndarray
        1-D array of observations.

    Returns
    -------
    float
        Estimated effective sample size, clamped to [1, n].
    """
    n = len(values)
    if n <= 1:
        return float(n)

    centered = values - np.mean(values)
    variance = np.var(centered, ddof=0)
    if variance < 1e-15:
        return float(n)

    # Autocorrelation via FFT (zero-padded to avoid circular artifacts)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_vals = np.fft.rfft(centered, n=fft_size)
    acf_full = np.fft.irfft(fft_vals * np.conj(fft_vals), n=fft_size)
    acf = acf_full[:n] / (variance * n)  # acf[0] == 1.0

    # Initial positive sequence estimator
    rho_sum = 0.0
    for lag in range(1, n):
        if acf[lag] <= 0.0:
            break
        rho_sum += acf[lag]

    tau = 1.0 + 2.0 * rho_sum
    ess = n / tau
    return float(max(1.0, min(ess, float(n))))


# ---------------------------------------------------------------------------
# Block bootstrap CI
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Circular block bootstrap confidence interval for the mean.

    Constructs each bootstrap replicate by sampling random starting
    indices and extracting contiguous blocks that wrap around circularly.
    Each replicate has the same length as the original array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observations.
    block_size : int
        Number of contiguous observations per block.
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (e.g. 0.05 for a 95 % CI).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of (mean, ci_lower, ci_upper)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    total_indices = n_blocks * block_size

    # Pre-compute the intra-block offset array [0, 1, ..., block_size-1]
    offsets = np.arange(block_size)

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        # Build index array: for each start, append start..start+block_size-1 (mod n)
        indices = np.empty(total_indices, dtype=np.intp)
        for i, s in enumerate(starts):
            base = i * block_size
            indices[base:base + block_size] = (s + offsets) % n
        # Trim to exactly n observations
        boot_means[b] = np.mean(values[indices[:n]])

    mean_val = float(np.mean(values))
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Main coverage experiment
# ---------------------------------------------------------------------------

def run_ci_coverage_experiment(
    n_trials: int = 16,
    n_samples: int = 300,
    n_outer_repeats: int = 200,
    n_bootstrap: int = 2000,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Validate bootstrap CI coverage by simulation.

    For each confidence level the procedure is:

    1. **Ground truth** -- run a large experiment (100 trials) with a fixed
       seed to obtain the "true" mean of treatment-trial means.
    2. **Replication** -- run *n_outer_repeats* smaller experiments (each
       with *n_trials* trials), compute a bootstrap CI at the requested
       confidence level, and record whether the CI contains the ground
       truth.
    3. **Assessment** -- empirical coverage = fraction of CIs that contain
       the ground truth.  Perfect calibration means empirical coverage
       matches the nominal confidence level.

    Two bootstrap methods are compared:

    * ``standard_bootstrap`` -- IID resampling via
      :func:`bootstrap_mean_ci`.
    * ``block_bootstrap`` -- circular block bootstrap with
      ``block_size=3`` via :func:`block_bootstrap_ci`.

    Parameters
    ----------
    n_trials : int
        Number of trials per small experiment.
    n_samples : int
        Number of latency samples per trial.
    n_outer_repeats : int
        Number of small experiments per (confidence_level, method) cell.
    n_bootstrap : int
        Number of bootstrap replicates for each CI.
    confidence_levels : list of float, optional
        Nominal confidence levels to evaluate.  Defaults to
        ``[0.50, 0.80, 0.90, 0.95, 0.99]``.

    Returns
    -------
    dict
        ``"coverage_df"`` : :class:`~pandas.DataFrame`
            One row per (confidence_level, method) combination with columns
            *confidence_level*, *method*, *nominal_coverage*,
            *empirical_coverage*, *ci_width_mean*, *ci_width_std*,
            *coverage_error*.

        ``"reliability_data"`` : dict
            Maps each *confidence_level* (float) to a list of
            ``(covered_or_not, ci_width)`` tuples aggregated across both
            methods.
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.80, 0.90, 0.95, 0.99]

    # ---- Fixed experimental condition -----------------------------------
    targets = get_targets()
    spectators = get_spectators()
    devices = get_device_profiles()

    target = targets["rpc_microservice"]
    spectator = spectators["cache_thrash"]
    device = devices["workstation_a"]
    load = 0.7
    distance = "same_llc"
    regime = "structured"

    methods = ["standard_bootstrap", "block_bootstrap"]
    base_seed = 20240101

    # ---- Ground truth (shared across confidence levels) -----------------
    rng_truth = np.random.default_rng(base_seed)
    truth_result = run_irbs_condition(
        target=target,
        spectator=spectator,
        device=device,
        load=load,
        distance=distance,
        regime=regime,
        n_trials=100,
        n_samples=n_samples,
        rng=rng_truth,
    )
    truth_df = truth_result["summary_df"]
    true_mean = float(truth_df.loc[truth_df["is_treatment"] == 1, "mean"].mean())

    # ---- Iterate over confidence levels and methods ---------------------
    coverage_rows: List[Dict[str, Any]] = []
    reliability_data: Dict[float, List[Tuple[int, float]]] = {
        cl: [] for cl in confidence_levels
    }

    for cl_idx, cl in enumerate(confidence_levels):
        alpha = 1.0 - cl

        for method in methods:
            covered_flags: List[int] = []
            widths: List[float] = []

            for rep in range(n_outer_repeats):
                # Deterministic seed that varies across (cl, method, rep)
                method_offset = 0 if method == "standard_bootstrap" else 1
                rep_seed = (
                    base_seed
                    + 100_000 * (cl_idx + 1)
                    + 10_000 * method_offset
                    + rep
                )

                rng_exp = np.random.default_rng(rep_seed)
                result = run_irbs_condition(
                    target=target,
                    spectator=spectator,
                    device=device,
                    load=load,
                    distance=distance,
                    regime=regime,
                    n_trials=n_trials,
                    n_samples=n_samples,
                    rng=rng_exp,
                )

                # Treatment-trial mean latencies
                sdf = result["summary_df"]
                treat_vals = sdf.loc[sdf["is_treatment"] == 1, "mean"].values

                # Bootstrap CI
                boot_seed = base_seed + 50_000_000 + rep_seed
                rng_boot = np.random.default_rng(boot_seed)

                if method == "standard_bootstrap":
                    _, ci_lo, ci_hi = bootstrap_mean_ci(
                        treat_vals,
                        n_bootstrap=n_bootstrap,
                        alpha=alpha,
                        rng=rng_boot,
                    )
                else:
                    _, ci_lo, ci_hi = block_bootstrap_ci(
                        treat_vals,
                        block_size=3,
                        n_bootstrap=n_bootstrap,
                        alpha=alpha,
                        rng=rng_boot,
                    )

                covered = int(ci_lo <= true_mean <= ci_hi)
                width = ci_hi - ci_lo

                covered_flags.append(covered)
                widths.append(width)
                reliability_data[cl].append((covered, width))

            empirical_coverage = float(np.mean(covered_flags))
            widths_arr = np.array(widths)

            coverage_rows.append({
                "confidence_level": cl,
                "method": method,
                "nominal_coverage": cl,
                "empirical_coverage": empirical_coverage,
                "ci_width_mean": float(np.mean(widths_arr)),
                "ci_width_std": float(
                    np.std(widths_arr, ddof=1) if len(widths_arr) > 1 else 0.0
                ),
                "coverage_error": empirical_coverage - cl,
            })

    coverage_df = pd.DataFrame(coverage_rows)

    return {
        "coverage_df": coverage_df,
        "reliability_data": reliability_data,
    }

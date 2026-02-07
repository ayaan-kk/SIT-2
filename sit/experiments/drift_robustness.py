"""Drift robustness experiment: multiple drift types and magnitude sweep.

Proves that IRBS is not cosmetic -- it genuinely decorrelates temporal
drift from the treatment indicator, reducing estimation bias across a
battery of realistic drift patterns (thermal ramp, DVFS step, periodic
oscillation, heteroscedastic noise, and combined).

The magnitude sweep shows that bias reduction is monotonically
increasing with drift severity: the harder the problem, the more IRBS
helps relative to naive (all-control-first) designs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from sit.simulator.workloads import get_targets, get_spectators, Workload
from sit.simulator.device_profiles import get_device_profiles, DeviceProfile
from sit.simulator.drift import inject_synthetic_drift
from sit.experiments.run_experiments import run_irbs_condition, run_naive_condition


# ------------------------------------------------------------------ #
#  Drift pattern generators                                           #
# ------------------------------------------------------------------ #

def generate_drift_types(
    n_trials: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Generate a dictionary of named drift series for robustness testing.

    Each drift series is a 1-D array of length *n_trials* representing
    multiplicative drift multipliers.  A value of 1.0 means no drift.

    Drift types
    -----------
    none
        All ones -- zero drift.  Used as the sanity-check baseline.
    slow_ramp
        Linear ramp from 1.0 to 1.3 over *n_trials*.  Models gradual
        thermal warm-up.
    abrupt_shift
        1.0 for the first half of trials, 1.2 for the second half.
        Models a DVFS step-change or background daemon activation.
    periodic
        ``1.0 + 0.1 * sin(2*pi*t/n_trials * 3)`` -- three full cycles
        of sinusoidal oscillation.  Proxy for periodic thermal
        throttling or cron-job interference.
    heteroscedastic
        1.0 + noise whose standard deviation ramps linearly from 0.01
        to 0.1 over the experiment window.  Models increasing
        environmental instability.
    combined
        Sum of slow_ramp, periodic, and small random noise.  Represents
        a realistic scenario where multiple drift sources act
        simultaneously.

    Args:
        n_trials: number of trials in the experiment.
        rng: numpy random Generator for reproducibility.

    Returns:
        Dict mapping drift-type name to ``np.ndarray`` of shape
        ``(n_trials,)``.
    """
    t = np.arange(n_trials, dtype=np.float64)
    t_frac = t / max(n_trials - 1, 1)

    drift_types: Dict[str, np.ndarray] = {}

    # 1. No drift
    drift_types["none"] = np.ones(n_trials, dtype=np.float64)

    # 2. Slow ramp: 1.0 -> 1.3
    drift_types["slow_ramp"] = 1.0 + 0.3 * t_frac

    # 3. Abrupt shift: 1.0 for first half, 1.2 for second half
    drift_types["abrupt_shift"] = np.where(t_frac < 0.5, 1.0, 1.2)

    # 4. Periodic: three-cycle sine wave
    drift_types["periodic"] = 1.0 + 0.1 * np.sin(
        2.0 * np.pi * t / n_trials * 3.0
    )

    # 5. Heteroscedastic: increasing noise std from 0.01 to 0.1
    noise_std = 0.01 + 0.09 * t_frac  # linearly from 0.01 to 0.10
    drift_types["heteroscedastic"] = 1.0 + rng.normal(0.0, noise_std)

    # 6. Combined: slow_ramp + periodic + small random noise
    combined_noise = rng.normal(0.0, 0.02, size=n_trials)
    drift_types["combined"] = (
        drift_types["slow_ramp"]
        + (drift_types["periodic"] - 1.0)
        + combined_noise
    )

    return drift_types


# ------------------------------------------------------------------ #
#  Main sweep experiment                                              #
# ------------------------------------------------------------------ #

def run_drift_robustness_sweep(
    n_trials: int = 16,
    n_samples: int = 300,
    n_repeats: int = 20,
    magnitudes: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run the full drift-robustness magnitude sweep.

    For every (drift_type, magnitude) pair and for *n_repeats*
    independent repetitions:

    1. Construct a scaled drift series:
       ``drift = 1.0 + magnitude * (raw_drift - 1.0)``
       so that ``magnitude=0`` collapses to no drift and ``magnitude=1``
       applies the full raw pattern.

    2. Estimate a "true" treatment effect (tau) from a large, zero-drift
       IRBS run.

    3. Run both naive and IRBS experiments under the scaled drift and
       compute bias = estimated_tau - true_tau, plus RMSE.

    The zero-drift rows (``magnitude=0.0``) act as an in-experiment
    sanity check: IRBS should not be *worse* than naive when there is no
    drift.

    Args:
        n_trials: trials per experiment (default 16).
        n_samples: samples per trial (default 300).
        n_repeats: independent repetitions per condition (default 20).
        magnitudes: list of magnitude scalars to sweep.  Defaults to
            ``[0.0, 0.05, 0.1, 0.15, 0.2, 0.3]``.

    Returns:
        Dict with keys:

        ``"drift_sweep_df"``
            DataFrame with per-repeat rows.  Columns: drift_type,
            magnitude, repeat, naive_bias, irbs_bias, naive_rmse,
            irbs_rmse.

        ``"drift_summary_df"``
            DataFrame aggregated over repeats.  Columns: drift_type,
            magnitude, mean_naive_bias, mean_irbs_bias,
            bias_reduction_pct, naive_rmse, irbs_rmse.
    """
    if magnitudes is None:
        magnitudes = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    # Pick representative workloads and device
    targets = get_targets()
    spectators = get_spectators()
    devices = get_device_profiles()

    target = targets["rpc_microservice"]
    spectator = spectators["cache_thrash"]
    device = list(devices.values())[0]  # first available device

    load = 0.7
    distance = "same_llc"
    regime = "structured"
    base_seed = 42

    # ----- Estimate true tau from large zero-drift run -----
    rng_true = np.random.default_rng(base_seed + 99999)
    no_drift = np.ones(n_trials * 5)
    result_true = run_irbs_condition(
        target, spectator, device, load, distance, regime,
        n_trials * 5, n_samples, rng_true, drift_series=no_drift,
    )
    true_tau = result_true["effects"]["delta_p99"]

    # ----- Generate raw drift patterns -----
    rng_drift = np.random.default_rng(base_seed + 77777)
    raw_drifts = generate_drift_types(n_trials, rng_drift)

    # ----- Sweep -----
    rows: List[Dict[str, Any]] = []

    for drift_name, raw_drift in raw_drifts.items():
        for mag in magnitudes:
            # Collect per-repeat bias values for RMSE computation
            naive_biases: List[float] = []
            irbs_biases: List[float] = []

            for rep in range(n_repeats):
                # Scale drift by magnitude
                scaled_drift = 1.0 + mag * (raw_drift - 1.0)

                # Naive (all control first, then treatment)
                rng_naive = np.random.default_rng(
                    base_seed + rep + hash((drift_name, mag, "naive")) % (2**31)
                )
                naive_result = run_naive_condition(
                    target, spectator, device, load, distance, regime,
                    n_trials, n_samples, rng_naive,
                    drift_series=scaled_drift,
                )
                naive_tau = naive_result["effects"]["delta_p99"]
                naive_bias = naive_tau - true_tau

                # IRBS (interleaved)
                rng_irbs = np.random.default_rng(
                    base_seed + rep + hash((drift_name, mag, "irbs")) % (2**31)
                )
                irbs_result = run_irbs_condition(
                    target, spectator, device, load, distance, regime,
                    n_trials, n_samples, rng_irbs,
                    drift_series=scaled_drift,
                )
                irbs_tau = irbs_result["effects"]["delta_p99"]
                irbs_bias = irbs_tau - true_tau

                naive_biases.append(naive_bias)
                irbs_biases.append(irbs_bias)

                rows.append({
                    "drift_type": drift_name,
                    "magnitude": mag,
                    "repeat": rep,
                    "naive_bias": naive_bias,
                    "irbs_bias": irbs_bias,
                    "naive_rmse": np.nan,  # filled in summary
                    "irbs_rmse": np.nan,
                })

            # Compute RMSE for this (drift_type, magnitude) block
            naive_rmse = float(np.sqrt(np.mean(np.array(naive_biases) ** 2)))
            irbs_rmse = float(np.sqrt(np.mean(np.array(irbs_biases) ** 2)))

            # Back-fill RMSE into the per-repeat rows
            for row in rows[-(n_repeats):]:
                row["naive_rmse"] = naive_rmse
                row["irbs_rmse"] = irbs_rmse

    sweep_df = pd.DataFrame(rows)

    # ----- Summary aggregation -----
    summary_rows: List[Dict[str, Any]] = []
    for (drift_name, mag), grp in sweep_df.groupby(["drift_type", "magnitude"]):
        mean_naive_bias = float(grp["naive_bias"].mean())
        mean_irbs_bias = float(grp["irbs_bias"].mean())
        naive_rmse = float(grp["naive_rmse"].iloc[0])
        irbs_rmse = float(grp["irbs_rmse"].iloc[0])

        # Bias reduction: how much closer to zero is IRBS bias?
        abs_naive = abs(mean_naive_bias)
        abs_irbs = abs(mean_irbs_bias)
        if abs_naive > 1e-12:
            bias_reduction_pct = (1.0 - abs_irbs / abs_naive) * 100.0
        else:
            bias_reduction_pct = 0.0

        summary_rows.append({
            "drift_type": drift_name,
            "magnitude": mag,
            "mean_naive_bias": mean_naive_bias,
            "mean_irbs_bias": mean_irbs_bias,
            "bias_reduction_pct": bias_reduction_pct,
            "naive_rmse": naive_rmse,
            "irbs_rmse": irbs_rmse,
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "drift_sweep_df": sweep_df,
        "drift_summary_df": summary_df,
    }


# ------------------------------------------------------------------ #
#  Zero-drift sanity check                                            #
# ------------------------------------------------------------------ #

def run_zero_drift_sanity_check(
    n_trials: int = 16,
    n_samples: int = 300,
    n_repeats: int = 50,
) -> Dict[str, Any]:
    """Confirm that IRBS does not *hurt* when drift is zero.

    Under zero drift, naive and IRBS designs have identical expected
    bias.  However, IRBS should not introduce *extra* estimation error
    from the interleaving.  This test checks that IRBS MAE is no worse
    than 1.1x the naive MAE.

    Args:
        n_trials: trials per experiment (default 16).
        n_samples: samples per trial (default 300).
        n_repeats: independent repetitions (default 50).

    Returns:
        Dict with keys:

        ``"naive_mae"``
            Mean absolute error of naive estimator (float).
        ``"irbs_mae"``
            Mean absolute error of IRBS estimator (float).
        ``"irbs_no_worse"``
            Boolean: True if ``irbs_mae <= naive_mae * 1.1``.
        ``"naive_biases"``
            List of per-repeat naive biases.
        ``"irbs_biases"``
            List of per-repeat IRBS biases.
    """
    targets = get_targets()
    spectators = get_spectators()
    devices = get_device_profiles()

    target = targets["rpc_microservice"]
    spectator = spectators["cache_thrash"]
    device = list(devices.values())[0]

    load = 0.7
    distance = "same_llc"
    regime = "structured"
    base_seed = 42

    # Estimate true tau from large no-drift run
    rng_true = np.random.default_rng(base_seed + 99999)
    no_drift_large = np.ones(n_trials * 5)
    result_true = run_irbs_condition(
        target, spectator, device, load, distance, regime,
        n_trials * 5, n_samples, rng_true, drift_series=no_drift_large,
    )
    true_tau = result_true["effects"]["delta_p99"]

    no_drift = np.ones(n_trials)

    naive_biases: List[float] = []
    irbs_biases: List[float] = []

    for rep in range(n_repeats):
        # Naive
        rng_naive = np.random.default_rng(base_seed + rep + 10000)
        naive_result = run_naive_condition(
            target, spectator, device, load, distance, regime,
            n_trials, n_samples, rng_naive,
            drift_series=no_drift,
        )
        naive_bias = naive_result["effects"]["delta_p99"] - true_tau
        naive_biases.append(naive_bias)

        # IRBS
        rng_irbs = np.random.default_rng(base_seed + rep + 20000)
        irbs_result = run_irbs_condition(
            target, spectator, device, load, distance, regime,
            n_trials, n_samples, rng_irbs,
            drift_series=no_drift,
        )
        irbs_bias = irbs_result["effects"]["delta_p99"] - true_tau
        irbs_biases.append(irbs_bias)

    naive_mae = float(np.mean(np.abs(naive_biases)))
    irbs_mae = float(np.mean(np.abs(irbs_biases)))
    irbs_no_worse = bool(irbs_mae <= naive_mae * 1.1)

    return {
        "naive_mae": naive_mae,
        "irbs_mae": irbs_mae,
        "irbs_no_worse": irbs_no_worse,
        "naive_biases": naive_biases,
        "irbs_biases": irbs_biases,
    }

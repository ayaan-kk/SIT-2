"""Drift bias demonstration for IRBS.

Shows that naive (sequential) experiment design is biased under temporal drift,
while IRBS (interleaved randomization) produces unbiased estimates.

The demonstration:
1. Creates a synthetic drift pattern (known, deterministic).
2. Runs many repetitions of both naive and IRBS experiments under that drift.
3. Compares estimated treatment effects to a drift-free reference (true_tau).
"""

import numpy as np
import pandas as pd
from typing import Optional

from sit.simulator.workloads import Workload
from sit.simulator.device_profiles import DeviceProfile
from sit.simulator.drift import inject_synthetic_drift

from .protocol import run_irbs_experiment, run_naive_experiment
from .estimators import compute_effect


def _estimate_true_tau(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples_per_trial: int,
    rng: np.random.Generator,
    metric: str = "p99",
) -> float:
    """Estimate the true treatment effect from a large drift-free run.

    Uses a no-drift series (all 1.0) so that the only difference between
    control and treatment is the presence of the spectator.
    """
    no_drift = np.ones(n_trials, dtype=np.float64)

    result = run_irbs_experiment(
        target=target,
        spectator=spectator,
        device=device,
        load=load,
        distance=distance,
        regime=regime,
        n_trials=n_trials,
        n_samples_per_trial=n_samples_per_trial,
        rng=rng,
        drift_series=no_drift,
    )

    return compute_effect(result["trial_types"], result["per_trial_summaries"], metric)


def run_drift_bias_demo(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples: int,
    rng: np.random.Generator,
    n_repeats: int = 50,
    metric: str = "p99",
) -> pd.DataFrame:
    """Run the drift-bias demonstration.

    For each of ``n_repeats`` repetitions the function:
    1. Creates a synthetic drift series via ``inject_synthetic_drift``.
    2. Runs a **naive** experiment (controls first, then treatments) with
       drift applied in sequential order.
    3. Runs an **IRBS** experiment (interleaved) with the same drift series.
    4. Records the estimated treatment effect (tau_hat) from each design.

    A drift-free reference run provides ``true_tau`` for comparison.

    Args:
        target: target workload.
        spectator: spectator workload.
        device: device profile.
        load: load level in [0, 1].
        distance: placement distance key.
        regime: interference regime.
        n_trials: total trials per experiment (should be even).
        n_samples: latency samples per trial.
        rng: numpy random Generator.
        n_repeats: number of repetitions.
        metric: metric to use for effect estimation.

    Returns:
        DataFrame with columns:
            repeat     - repetition index (0-based)
            naive_tau  - tau_hat from the naive design
            irbs_tau   - tau_hat from the IRBS design
            true_tau   - reference tau_hat from a drift-free run
    """
    # Estimate the true treatment effect from a large drift-free run
    ref_rng = np.random.default_rng(rng.integers(0, 2**63))
    # Use more trials for the reference to reduce noise
    n_ref_trials = max(n_trials, 100)
    true_tau = _estimate_true_tau(
        target=target,
        spectator=spectator,
        device=device,
        load=load,
        distance=distance,
        regime=regime,
        n_trials=n_ref_trials,
        n_samples_per_trial=n_samples,
        rng=ref_rng,
        metric=metric,
    )

    records = []

    for rep in range(n_repeats):
        # Create a fresh synthetic drift series for this repetition
        drift_rng = np.random.default_rng(rng.integers(0, 2**63))
        drift_series = inject_synthetic_drift(
            total_trials=n_trials,
            rng=drift_rng,
        )

        # --- Naive experiment (drift applied in sequential order) ---
        naive_rng = np.random.default_rng(rng.integers(0, 2**63))
        naive_result = run_naive_experiment(
            target=target,
            spectator=spectator,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_trials=n_trials,
            n_samples_per_trial=n_samples,
            rng=naive_rng,
            drift_series=drift_series,
        )
        naive_tau = compute_effect(
            naive_result["trial_types"],
            naive_result["per_trial_summaries"],
            metric=metric,
        )

        # --- IRBS experiment (interleaved, same drift series) ---
        irbs_rng = np.random.default_rng(rng.integers(0, 2**63))
        irbs_result = run_irbs_experiment(
            target=target,
            spectator=spectator,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_trials=n_trials,
            n_samples_per_trial=n_samples,
            rng=irbs_rng,
            drift_series=drift_series,
        )
        irbs_tau = compute_effect(
            irbs_result["trial_types"],
            irbs_result["per_trial_summaries"],
            metric=metric,
        )

        records.append({
            "repeat": rep,
            "naive_tau": naive_tau,
            "irbs_tau": irbs_tau,
            "true_tau": true_tau,
        })

    return pd.DataFrame(records)

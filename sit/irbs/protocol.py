"""IRBS protocol implementation.

Implements interleaved randomized block scheduling for interference experiments.
Control (Z=0) and treatment (Z=1) trials are randomly permuted to decorrelate
trial assignment from temporal drift, preventing confounding.

Also provides a naive (non-interleaved) baseline for comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sit.simulator.workloads import Workload
from sit.simulator.device_profiles import DeviceProfile
from sit.simulator.latency_generator import generate_trial_samples
from sit.simulator.drift import compute_drift_multiplier, DriftState


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = ("mean", "p95", "p99", "cvar95", "cvar99", "slo_violation_rate")


def compute_slo_threshold(target: Workload, device: DeviceProfile) -> float:
    """Compute the SLO latency threshold for a target workload on a device.

    The threshold is defined as 3x the scaled base latency.
    """
    return 3.0 * target.base_latency_us * device.baseline_latency_scale


def _compute_trial_summary(
    samples: np.ndarray,
    slo_threshold: float,
) -> Dict[str, float]:
    """Compute summary statistics for a single trial's samples.

    Returns a dict with keys: mean, p95, p99, cvar95, cvar99, slo_violation_rate.
    """
    mean_val = float(np.mean(samples))
    p95_val = float(np.percentile(samples, 95))
    p99_val = float(np.percentile(samples, 99))

    sorted_samples = np.sort(samples)
    n = len(sorted_samples)

    # CVaR95: mean of samples above the 95th percentile (top 5%)
    idx_95 = int(np.ceil(n * 0.95))
    tail_95 = sorted_samples[idx_95:]
    cvar95_val = float(np.mean(tail_95)) if len(tail_95) > 0 else p95_val

    # CVaR99: mean of samples above the 99th percentile (top 1%)
    idx_99 = int(np.ceil(n * 0.99))
    tail_99 = sorted_samples[idx_99:]
    cvar99_val = float(np.mean(tail_99)) if len(tail_99) > 0 else p99_val

    # SLO violation rate: fraction of samples above the threshold
    slo_violation_rate = float(np.mean(samples > slo_threshold))

    return {
        "mean": mean_val,
        "p95": p95_val,
        "p99": p99_val,
        "cvar95": cvar95_val,
        "cvar99": cvar99_val,
        "slo_violation_rate": slo_violation_rate,
    }


# ---------------------------------------------------------------------------
# IRBS experiment (interleaved randomization)
# ---------------------------------------------------------------------------

def run_irbs_experiment(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples_per_trial: int,
    rng: np.random.Generator,
    drift_series: Optional[np.ndarray] = None,
) -> Dict:
    """Run an IRBS experiment with interleaved randomized trial ordering.

    Creates *n_trials / 2* control trials (Z=0, no spectator) and
    *n_trials / 2* treatment trials (Z=1, with spectator), then randomly
    permutes the execution order so that drift is decorrelated from treatment
    assignment.

    Args:
        target: target workload under test.
        spectator: spectator (interferer) workload.
        device: device profile.
        load: load level in [0, 1].
        distance: placement distance key.
        regime: interference regime ("benign", "structured", "adversarial").
        n_trials: total number of trials (should be even).
        n_samples_per_trial: latency samples per trial.
        rng: numpy random Generator.
        drift_series: optional pre-computed array of drift multipliers,
            length >= n_trials.  If provided, drift_series[trial_position]
            is used as the drift multiplier for the trial executed at that
            position.  If None, drift is computed on-the-fly via
            ``compute_drift_multiplier``.

    Returns:
        Dict with keys:
            trial_types     - int array of shape (n_trials,), 0=control 1=treatment
            trial_order     - int array, the random permutation applied
            raw_samples     - list of n_trials ndarrays
            per_trial_summaries - DataFrame with summary metrics per trial
    """
    n_control = n_trials // 2
    n_treatment = n_trials - n_control

    # Build assignment vector: 0 = control, 1 = treatment
    trial_types = np.array([0] * n_control + [1] * n_treatment, dtype=int)

    # Random permutation (interleaving)
    trial_order = rng.permutation(n_trials)
    trial_types = trial_types[trial_order]

    slo_threshold = compute_slo_threshold(target, device)
    drift_state = DriftState()

    raw_samples: List[np.ndarray] = []
    summaries: List[Dict[str, float]] = []

    for position in range(n_trials):
        z = trial_types[position]

        # Determine drift multiplier for this position
        if drift_series is not None:
            dm = float(drift_series[position])
        else:
            dm = compute_drift_multiplier(
                trial_index=position,
                total_trials=n_trials,
                device=device,
                rng=rng,
                drift_state=drift_state,
            )

        # Generate samples
        trial_rng = np.random.default_rng(rng.integers(0, 2**63))
        samples = generate_trial_samples(
            target=target,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_samples=n_samples_per_trial,
            rng=trial_rng,
            spectator=spectator if z == 1 else None,
            drift_multiplier=dm,
        )

        raw_samples.append(samples)
        summaries.append(_compute_trial_summary(samples, slo_threshold))

    per_trial_summaries = pd.DataFrame(summaries)
    per_trial_summaries["trial_type"] = trial_types

    return {
        "trial_types": trial_types,
        "trial_order": trial_order,
        "raw_samples": raw_samples,
        "per_trial_summaries": per_trial_summaries,
    }


# ---------------------------------------------------------------------------
# Naive experiment (all control first, then all treatment -- no interleaving)
# ---------------------------------------------------------------------------

def run_naive_experiment(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples_per_trial: int,
    rng: np.random.Generator,
    drift_series: Optional[np.ndarray] = None,
) -> Dict:
    """Run a naive (non-interleaved) experiment.

    Executes all control trials first, then all treatment trials.  This
    ordering is susceptible to drift-induced bias because treatment trials
    systematically occur later in the experiment when drift may be higher.

    Args:
        Same as ``run_irbs_experiment``.

    Returns:
        Same structure as ``run_irbs_experiment``, but ``trial_order`` is
        simply ``[0, 1, 2, ...]`` (identity -- no permutation).
    """
    n_control = n_trials // 2
    n_treatment = n_trials - n_control

    # Fixed ordering: all controls first, then all treatments
    trial_types = np.array([0] * n_control + [1] * n_treatment, dtype=int)
    trial_order = np.arange(n_trials, dtype=int)  # identity permutation

    slo_threshold = compute_slo_threshold(target, device)
    drift_state = DriftState()

    raw_samples: List[np.ndarray] = []
    summaries: List[Dict[str, float]] = []

    for position in range(n_trials):
        z = trial_types[position]

        # Determine drift multiplier for this position
        if drift_series is not None:
            dm = float(drift_series[position])
        else:
            dm = compute_drift_multiplier(
                trial_index=position,
                total_trials=n_trials,
                device=device,
                rng=rng,
                drift_state=drift_state,
            )

        trial_rng = np.random.default_rng(rng.integers(0, 2**63))
        samples = generate_trial_samples(
            target=target,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_samples=n_samples_per_trial,
            rng=trial_rng,
            spectator=spectator if z == 1 else None,
            drift_multiplier=dm,
        )

        raw_samples.append(samples)
        summaries.append(_compute_trial_summary(samples, slo_threshold))

    per_trial_summaries = pd.DataFrame(summaries)
    per_trial_summaries["trial_type"] = trial_types

    return {
        "trial_types": trial_types,
        "trial_order": trial_order,
        "raw_samples": raw_samples,
        "per_trial_summaries": per_trial_summaries,
    }

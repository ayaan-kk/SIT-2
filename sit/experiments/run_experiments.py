"""Core experiment execution engine.

Runs simulator experiments, IRBS, tomography, scheduling, and mismatch analysis.
Produces raw traces and derived results.
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sit.simulator.workloads import get_targets, get_spectators, Workload
from sit.simulator.device_profiles import get_device_profiles, DeviceProfile
from sit.simulator.drift import compute_drift_series, DriftState, compute_drift_multiplier
from sit.simulator.latency_generator import generate_trial_samples
from sit.simulator.interference_channels import compute_cosine_similarity
from sit.analysis.metrics import (
    compute_p99, compute_p95, compute_cvar99, compute_cvar95,
    compute_slo_violation_rate, compute_trial_summary, compute_ssi,
)


def get_slo_threshold(target: Workload, device: DeviceProfile) -> float:
    """SLO threshold = 3x baseline latency scaled by device."""
    return 3.0 * target.base_latency_us * device.baseline_latency_scale


def run_irbs_condition(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples: int,
    rng: np.random.Generator,
    drift_series: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run IRBS experiment for a single condition.

    Returns dict with trial_types, raw_samples, per_trial_summaries.
    """
    n_ctrl = n_trials // 2
    n_treat = n_trials - n_ctrl
    slo = get_slo_threshold(target, device)

    # Create trial order: 0=control, 1=treatment
    trial_types = np.array([0] * n_ctrl + [1] * n_treat)
    permutation = rng.permutation(n_trials)
    trial_types_ordered = trial_types[permutation]

    # Generate drift if not provided
    if drift_series is None:
        drift_series = compute_drift_series(n_trials, device, rng)

    summaries = []
    all_samples = []

    drift_state = DriftState()
    for trial_idx in range(n_trials):
        is_treatment = trial_types_ordered[trial_idx] == 1
        dm = drift_series[trial_idx] if trial_idx < len(drift_series) else 1.0

        samples = generate_trial_samples(
            target=target,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_samples=n_samples,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
            spectator=spectator if is_treatment else None,
            drift_multiplier=dm,
        )

        summary = compute_trial_summary(samples, slo)
        summary["trial_idx"] = trial_idx
        summary["is_treatment"] = int(is_treatment)
        summary["drift_multiplier"] = dm
        summaries.append(summary)
        all_samples.append(samples)

    summary_df = pd.DataFrame(summaries)

    # Compute effects
    ctrl_mask = summary_df["is_treatment"] == 0
    treat_mask = summary_df["is_treatment"] == 1

    effects = {}
    for metric in ["mean", "p95", "p99", "cvar95", "cvar99"]:
        ctrl_val = summary_df.loc[ctrl_mask, metric].mean()
        treat_val = summary_df.loc[treat_mask, metric].mean()
        effects[f"delta_{metric}"] = treat_val - ctrl_val
        effects[f"ctrl_{metric}"] = ctrl_val
        effects[f"treat_{metric}"] = treat_val

    if "slo_violation_rate" in summary_df.columns:
        effects["delta_slo_viol"] = (
            summary_df.loc[treat_mask, "slo_violation_rate"].mean() -
            summary_df.loc[ctrl_mask, "slo_violation_rate"].mean()
        )
        effects["ctrl_slo_viol"] = summary_df.loc[ctrl_mask, "slo_violation_rate"].mean()
        effects["treat_slo_viol"] = summary_df.loc[treat_mask, "slo_violation_rate"].mean()

    # SSI
    effects["ssi"] = compute_ssi(
        effects.get("ctrl_p99", 1.0),
        effects.get("treat_p99", 1.0),
        effects.get("ctrl_cvar99", 1.0),
        effects.get("treat_cvar99", 1.0),
    )

    return {
        "trial_types": trial_types_ordered,
        "summary_df": summary_df,
        "effects": effects,
        "raw_samples": all_samples,
        "slo_threshold": slo,
    }


def run_naive_condition(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_trials: int,
    n_samples: int,
    rng: np.random.Generator,
    drift_series: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run naive (non-interleaved) experiment: all control first, then all treatment."""
    n_ctrl = n_trials // 2
    n_treat = n_trials - n_ctrl
    slo = get_slo_threshold(target, device)

    # Ordered: controls first, then treatments
    trial_types_ordered = np.array([0] * n_ctrl + [1] * n_treat)

    if drift_series is None:
        drift_series = compute_drift_series(n_trials, device, rng)

    summaries = []
    for trial_idx in range(n_trials):
        is_treatment = trial_types_ordered[trial_idx] == 1
        dm = drift_series[trial_idx] if trial_idx < len(drift_series) else 1.0

        samples = generate_trial_samples(
            target=target,
            device=device,
            load=load,
            distance=distance,
            regime=regime,
            n_samples=n_samples,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
            spectator=spectator if is_treatment else None,
            drift_multiplier=dm,
        )

        summary = compute_trial_summary(samples, slo)
        summary["trial_idx"] = trial_idx
        summary["is_treatment"] = int(is_treatment)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    ctrl_mask = summary_df["is_treatment"] == 0
    treat_mask = summary_df["is_treatment"] == 1

    effects = {}
    for metric in ["mean", "p95", "p99", "cvar95", "cvar99"]:
        effects[f"delta_{metric}"] = (
            summary_df.loc[treat_mask, metric].mean() -
            summary_df.loc[ctrl_mask, metric].mean()
        )

    return {
        "trial_types": trial_types_ordered,
        "summary_df": summary_df,
        "effects": effects,
    }


def run_scheduling_evaluation(
    target: Workload,
    spectators_dict: Dict[str, Workload],
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_samples: int,
    rng: np.random.Generator,
    selected_spectators: List[str],
    scheduler_name: str,
    apply_static_partition: bool = False,
    triton_batching: bool = False,
) -> Dict[str, Any]:
    """Evaluate a scheduling decision by simulating co-location.

    Runs target with all selected spectators' combined interference.
    """
    slo = get_slo_threshold(target, device)

    # Generate samples with combined interference from all co-tenants
    # We simulate this by running with the "worst" spectator and scaling
    # by number of co-tenants (simplified model for scheduling eval)
    all_latencies = []

    if len(selected_spectators) == 0:
        # No co-tenants - just baseline
        samples = generate_trial_samples(
            target=target, device=device, load=load, distance=distance,
            regime=regime, n_samples=n_samples,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
            spectator=None,
        )
        all_latencies = samples
    else:
        # Run with each spectator and take element-wise max (worst-case combination)
        per_spectator_samples = []
        for s_name in selected_spectators:
            if s_name not in spectators_dict:
                continue
            spec = spectators_dict[s_name]
            samples = generate_trial_samples(
                target=target, device=device, load=load, distance=distance,
                regime=regime, n_samples=n_samples,
                rng=np.random.default_rng(rng.integers(0, 2**63)),
                spectator=spec,
                apply_static_partition=apply_static_partition,
                triton_batching=triton_batching,
            )
            per_spectator_samples.append(samples)

        if per_spectator_samples:
            # Combine: take mean across spectators but boost by count
            # This models additive interference from multiple co-tenants
            stacked = np.stack(per_spectator_samples, axis=0)
            base_samples = generate_trial_samples(
                target=target, device=device, load=load, distance=distance,
                regime=regime, n_samples=n_samples,
                rng=np.random.default_rng(rng.integers(0, 2**63)),
                spectator=None,
            )
            # Each co-tenant adds its delta above baseline
            deltas = stacked - base_samples[np.newaxis, :]
            deltas = np.maximum(deltas, 0)
            # Combined latency = baseline + sum of deltas (with diminishing returns)
            combined_delta = np.sum(deltas, axis=0) * (1.0 / (1.0 + 0.1 * len(selected_spectators)))
            all_latencies = base_samples + combined_delta
        else:
            all_latencies = generate_trial_samples(
                target=target, device=device, load=load, distance=distance,
                regime=regime, n_samples=n_samples,
                rng=np.random.default_rng(rng.integers(0, 2**63)),
                spectator=None,
            )

    all_latencies = np.asarray(all_latencies)
    summary = compute_trial_summary(all_latencies, slo)
    summary["scheduler"] = scheduler_name
    summary["target"] = target.name
    summary["device"] = device.device_id
    summary["load"] = load
    summary["distance"] = distance
    summary["regime"] = regime
    summary["n_cotenants"] = len(selected_spectators)
    summary["cotenants"] = ",".join(selected_spectators)

    return summary


def run_drift_bias_demo(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float = 0.7,
    distance: str = "same_llc",
    regime: str = "structured",
    n_trials: int = 20,
    n_samples: int = 200,
    n_repeats: int = 30,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run IRBS drift bias demonstration.

    Returns DataFrame with columns: repeat, naive_tau, irbs_tau, true_tau
    """
    from sit.simulator.drift import inject_synthetic_drift

    # Estimate "true" tau from a large no-drift run
    rng_true = np.random.default_rng(base_seed + 99999)
    no_drift = np.ones(n_trials * 5)
    result_true = run_irbs_condition(
        target, spectator, device, load, distance, regime,
        n_trials * 5, n_samples, rng_true, drift_series=no_drift,
    )
    true_tau = result_true["effects"]["delta_p99"]

    rows = []
    for rep in range(n_repeats):
        rng_rep = np.random.default_rng(base_seed + rep)

        # Create drift
        drift = inject_synthetic_drift(
            n_trials, ramp_rate=0.003, step_at_fraction=0.5,
            step_magnitude=0.15, rng=rng_rep,
        )

        # Naive: all control first, then treatment - drift correlates with treatment
        naive_result = run_naive_condition(
            target, spectator, device, load, distance, regime,
            n_trials, n_samples, np.random.default_rng(base_seed + rep + 10000),
            drift_series=drift,
        )

        # IRBS: interleaved - drift decorrelated from treatment
        irbs_result = run_irbs_condition(
            target, spectator, device, load, distance, regime,
            n_trials, n_samples, np.random.default_rng(base_seed + rep + 20000),
            drift_series=drift,
        )

        rows.append({
            "repeat": rep,
            "naive_tau": naive_result["effects"]["delta_p99"],
            "irbs_tau": irbs_result["effects"]["delta_p99"],
            "true_tau": true_tau,
        })

    return pd.DataFrame(rows)

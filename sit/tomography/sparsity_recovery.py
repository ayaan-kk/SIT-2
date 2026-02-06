"""Sparsity analysis and recovery curves for interference tomography.

Provides utilities to:

* Identify the top-*k* strongest interferers for a given target.
* Compute sparsity statistics (how concentrated interference is).
* Run recovery-curve experiments that measure how many IRBS trials
  are needed to reliably recover the true top-*k* interferers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sit.simulator.latency_generator import generate_control_treatment_pair
from sit.simulator.workloads import Workload
from sit.simulator.device_profiles import DeviceProfile

from .bootstrap import bootstrap_cell


# ---------------------------------------------------------------------------
# Top-k helpers
# ---------------------------------------------------------------------------

def get_top_k_interferers(
    mean_matrix: pd.DataFrame,
    target_name: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return the top-*k* highest-interference spectators for a target.

    Args:
        mean_matrix: DataFrame with targets as rows, spectators as
            columns (values are mean effects).
        target_name: row label identifying the target.
        k: number of top interferers to return (default 3).

    Returns:
        List of ``(spectator_name, effect_value)`` sorted by descending
        effect magnitude.
    """
    row = mean_matrix.loc[target_name]
    top_k = row.nlargest(k)
    return list(zip(top_k.index, top_k.values))


# ---------------------------------------------------------------------------
# Sparsity statistics
# ---------------------------------------------------------------------------

def sparsity_stats(mean_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target sparsity statistics.

    For each target, reports what fraction of total absolute interference
    is attributable to the top-1, top-3, and top-5 spectators.

    Args:
        mean_matrix: DataFrame with targets as rows, spectators as
            columns.

    Returns:
        DataFrame indexed by target with columns ``top1_share``,
        ``top3_share``, ``top5_share``.
    """
    records = []

    for target in mean_matrix.index:
        row = mean_matrix.loc[target]
        abs_row = row.abs()
        total = abs_row.sum()

        if total <= 0:
            records.append({
                "target": target,
                "top1_share": 0.0,
                "top3_share": 0.0,
                "top5_share": 0.0,
            })
            continue

        sorted_vals = abs_row.sort_values(ascending=False)
        n = len(sorted_vals)

        top1 = float(sorted_vals.iloc[: min(1, n)].sum() / total)
        top3 = float(sorted_vals.iloc[: min(3, n)].sum() / total)
        top5 = float(sorted_vals.iloc[: min(5, n)].sum() / total)

        records.append({
            "target": target,
            "top1_share": top1,
            "top3_share": top3,
            "top5_share": top5,
        })

    return pd.DataFrame(records).set_index("target")


# ---------------------------------------------------------------------------
# Internal helpers for recovery_curve
# ---------------------------------------------------------------------------

def _compute_trial_metrics(samples: np.ndarray) -> Dict[str, float]:
    """Compute summary metrics for a single trial's latency samples."""
    p99 = float(np.percentile(samples, 99))
    p95 = float(np.percentile(samples, 95))
    # CVaR (Conditional Value at Risk) = mean of samples above p95
    tail_mask = samples >= p95
    cvar = float(np.mean(samples[tail_mask])) if np.any(tail_mask) else p99
    mean_val = float(np.mean(samples))
    return {"p99": p99, "cvar": cvar, "mean": mean_val}


def _run_mini_irbs(
    target: Workload,
    spectators: Dict[str, Workload],
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_samples: int,
    trials_per_spectator: int,
    rng: np.random.Generator,
) -> dict:
    """Run a minimal IRBS experiment for recovery testing.

    For each spectator, generates *trials_per_spectator* matched
    control/treatment pairs and records per-trial summary metrics.

    Returns:
        An ``irbs_results_dict`` mapping ``(target_name, spectator_name)``
        to ``{'trial_types': ..., 'per_trial_summaries': ...}``.
    """
    results = {}

    for spec_name, spectator in spectators.items():
        trial_types: List[str] = []
        p99_vals: List[float] = []
        cvar_vals: List[float] = []
        mean_vals: List[float] = []

        for _t in range(trials_per_spectator):
            child_rng = np.random.default_rng(rng.integers(0, 2**63))

            control, treatment = generate_control_treatment_pair(
                target=target,
                spectator=spectator,
                device=device,
                load=load,
                distance=distance,
                regime=regime,
                n_samples=n_samples,
                rng=child_rng,
            )

            # Control trial metrics
            ctrl_metrics = _compute_trial_metrics(control)
            trial_types.append("control")
            p99_vals.append(ctrl_metrics["p99"])
            cvar_vals.append(ctrl_metrics["cvar"])
            mean_vals.append(ctrl_metrics["mean"])

            # Treatment trial metrics
            treat_metrics = _compute_trial_metrics(treatment)
            trial_types.append("treatment")
            p99_vals.append(treat_metrics["p99"])
            cvar_vals.append(treat_metrics["cvar"])
            mean_vals.append(treat_metrics["mean"])

        results[(target.name, spec_name)] = {
            "trial_types": np.array(trial_types),
            "per_trial_summaries": {
                "p99": np.array(p99_vals),
                "cvar": np.array(cvar_vals),
                "mean": np.array(mean_vals),
            },
        }

    return results


def _get_top_k_from_irbs(
    irbs_results_dict: dict,
    target_name: str,
    k: int,
    metric: str = "p99",
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Extract top-*k* interferer names from IRBS results via bootstrap.

    Uses :func:`bootstrap_cell` to estimate per-spectator mean effects,
    then returns the *k* spectators with the largest mean.
    """
    effects: Dict[str, float] = {}

    for (t, s), cell_data in irbs_results_dict.items():
        if t != target_name:
            continue

        cell_rng = (
            np.random.default_rng(rng.integers(0, 2**63))
            if rng is not None
            else None
        )

        mean_eff, _ci_lo, _ci_hi, _se = bootstrap_cell(
            cell_data["trial_types"],
            cell_data["per_trial_summaries"],
            metric=metric,
            n_bootstrap=500,  # fewer resamples since we only need ranking
            rng=cell_rng,
        )
        effects[s] = mean_eff

    sorted_specs = sorted(effects.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _val in sorted_specs[:k]]


# ---------------------------------------------------------------------------
# Recovery curve
# ---------------------------------------------------------------------------

def recovery_curve(
    target: Workload,
    spectators: Dict[str, Workload],
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_samples: int,
    rng: np.random.Generator,
    max_trials_per_spectator: int = 50,
    k_values: Optional[List[int]] = None,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Compute recovery-probability curves for top-*k* identification.

    For a sequence of increasing trial budgets, measures how often we
    correctly recover the *ground-truth* top-*k* interferers (established
    with *max_trials_per_spectator* trials).

    Algorithm:

    1. Run a high-budget IRBS experiment (``max_trials_per_spectator``
       trials per spectator) to establish the ground-truth top-*k* set
       for each *k* in *k_values*.
    2. For each trial budget in ``[4, 8, 12, 16, 20, 30, 40, 50]``:

       a. Repeat *n_repeats* times:

          * Run IRBS with the given budget.
          * Identify the top-*k* via bootstrap means.
          * Check whether the recovered set equals the ground truth.

       b. Record the fraction of repeats that recovered correctly.

    Args:
        target: target :class:`~sit.simulator.workloads.Workload`.
        spectators: dict of spectator Workloads keyed by name.
        device: :class:`~sit.simulator.device_profiles.DeviceProfile`.
        load: load level (0--1).
        distance: placement distance key (e.g. ``'same_llc'``).
        regime: interference regime (``'benign'``, ``'structured'``,
            ``'adversarial'``).
        n_samples: number of latency samples per trial.
        rng: numpy random generator.
        max_trials_per_spectator: trial budget for ground-truth
            estimation (default 50).
        k_values: list of *k* values to evaluate (default ``[1, 3]``).
        n_repeats: number of independent repeats per budget level
            (default 30).

    Returns:
        DataFrame with columns ``trials_per_spectator``, ``k``, and
        ``recovery_probability``.
    """
    if k_values is None:
        k_values = [1, 3]

    trial_budgets = [4, 8, 12, 16, 20, 30, 40, 50]

    # ------------------------------------------------------------------
    # Step 1: Establish ground truth with maximum trial budget
    # ------------------------------------------------------------------
    gt_rng = np.random.default_rng(rng.integers(0, 2**63))
    gt_results = _run_mini_irbs(
        target, spectators, device, load, distance, regime,
        n_samples, max_trials_per_spectator, gt_rng,
    )

    ground_truth: Dict[int, set] = {}
    for k in k_values:
        gt_top_k = _get_top_k_from_irbs(
            gt_results,
            target.name,
            k,
            metric="p99",
            rng=np.random.default_rng(rng.integers(0, 2**63)),
        )
        ground_truth[k] = set(gt_top_k)

    # ------------------------------------------------------------------
    # Step 2: For each trial budget, measure recovery probability
    # ------------------------------------------------------------------
    records: List[dict] = []

    for budget in trial_budgets:
        if budget > max_trials_per_spectator:
            continue

        for k in k_values:
            recoveries = 0

            for _rep in range(n_repeats):
                rep_rng = np.random.default_rng(rng.integers(0, 2**63))

                rep_results = _run_mini_irbs(
                    target, spectators, device, load, distance, regime,
                    n_samples, budget, rep_rng,
                )

                recovered = set(_get_top_k_from_irbs(
                    rep_results,
                    target.name,
                    k,
                    metric="p99",
                    rng=np.random.default_rng(rng.integers(0, 2**63)),
                ))

                if recovered == ground_truth[k]:
                    recoveries += 1

            records.append({
                "trials_per_spectator": budget,
                "k": k,
                "recovery_probability": recoveries / n_repeats,
            })

    return pd.DataFrame(records)

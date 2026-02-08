"""SIT-DPP scheduler: diversity-promoting, risk-aware co-tenant placement.

Uses a greedy determinantal-point-process (DPP) style objective that
balances two competing goals:

1. **Diversity** -- selected spectators should span different resource
   profiles so that interference is spread across channels rather than
   concentrated on a single bottleneck.
2. **Low risk** -- predicted interference (from the tomography map)
   should be minimised, optionally using an upper-confidence-bound (UCB)
   formulation that accounts for estimation uncertainty.

The combined acquisition function at each greedy step is:

    score(c) = lambda_div * marginal_logdet_gain(c)
             - lambda_risk * predicted_risk(c)

The candidate with the highest score is selected.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sit.scheduling.utils import (
    compute_predicted_risk,
    marginal_gain,
)


def sit_dpp_schedule(
    target_name: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean: pd.DataFrame,
    tomography_stderr: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
    beta: float = 0.0,
    lambda_risk: float = 1.0,
    lambda_div: float = 1.0,
    epsilon: float = 1e-6,
) -> Tuple[List[str], Dict]:
    """Run the SIT-DPP greedy scheduler.

    Parameters
    ----------
    target_name : str
        Latency-sensitive target workload name.
    candidate_spectators : list of str
        Pool of spectator workload names eligible for co-location.
    n_slots : int
        Number of co-location slots to fill.
    tomography_mean : pd.DataFrame
        Mean interference matrix (rows=targets, columns=spectators).
    tomography_stderr : pd.DataFrame or None
        Standard-error matrix, same shape as *tomography_mean*.
    kernel_matrix : np.ndarray
        Pre-computed RBF similarity kernel.
    workload_names : list of str
        Names corresponding to rows/columns of *kernel_matrix*.
    rng : np.random.Generator
        Random number generator (used for tie-breaking).
    beta : float
        UCB exploration weight.  0 means use mean only.
    lambda_risk : float
        Weight for the predicted-risk penalty.
    lambda_div : float
        Weight for the log-det diversity reward.
    epsilon : float
        Regularisation constant for log-det computation.

    Returns
    -------
    selected : list of str
        Ordered list of selected spectator names.
    diagnostics : dict
        Per-step diagnostic information including scores and reasons.
    """
    # Map candidate names to kernel indices (skip unknowns)
    name_to_idx: Dict[str, int] = {}
    for name in candidate_spectators:
        if name in workload_names:
            name_to_idx[name] = workload_names.index(name)

    selected: List[str] = []
    selected_indices: List[int] = []
    remaining = list(candidate_spectators)

    diagnostics: Dict = {
        "steps": [],
        "target": target_name,
        "beta": beta,
        "lambda_risk": lambda_risk,
        "lambda_div": lambda_div,
    }

    n_pick = min(n_slots, len(candidate_spectators))

    # Pre-compute per-candidate risk for normalization
    raw_risks: Dict[str, float] = {}
    for s in candidate_spectators:
        raw_risks[s] = compute_predicted_risk(
            target_name=target_name,
            cotenant_names=[s],
            tomography_mean=tomography_mean,
            tomography_stderr=tomography_stderr,
            beta=beta,
        )
    risk_vals = list(raw_risks.values())
    risk_min = min(risk_vals) if risk_vals else 0.0
    risk_max = max(risk_vals) if risk_vals else 1.0
    risk_range = risk_max - risk_min if risk_max > risk_min else 1.0

    for step in range(n_pick):
        best_name: Optional[str] = None
        best_score = -np.inf
        step_info: Dict = {"step": step, "candidates": {}}

        # Shuffle remaining for random tie-breaking
        rng.shuffle(remaining)

        for s in remaining:
            # --- Diversity term ---
            if s in name_to_idx:
                s_idx = name_to_idx[s]
                div_gain = marginal_gain(
                    K=kernel_matrix,
                    current_set=selected_indices,
                    candidate=s_idx,
                    epsilon=epsilon,
                )
            else:
                # Unknown workload -- assign small neutral diversity gain
                div_gain = 0.0

            # --- Risk term (normalized to [0,1]) ---
            risk_raw = raw_risks.get(s, 0.0)
            risk = (risk_raw - risk_min) / risk_range

            # --- Combined score (higher is better) ---
            score = lambda_div * div_gain - lambda_risk * risk

            step_info["candidates"][s] = {
                "div_gain": float(div_gain),
                "risk": float(risk),
                "score": float(score),
            }

            if score > best_score:
                best_score = score
                best_name = s

        if best_name is None:
            break

        # Record selection
        step_info["selected"] = best_name
        step_info["selected_score"] = float(best_score)
        step_info["reason"] = (
            f"Best combined score: div_gain="
            f"{step_info['candidates'][best_name]['div_gain']:.4f}, "
            f"risk={step_info['candidates'][best_name]['risk']:.4f}"
        )
        diagnostics["steps"].append(step_info)

        selected.append(best_name)
        if best_name in name_to_idx:
            selected_indices.append(name_to_idx[best_name])
        remaining.remove(best_name)

    diagnostics["final_selection"] = list(selected)
    return selected, diagnostics

"""SIT-SLO scheduler: SLO-aware, diversity-promoting co-tenant placement.

Extends SIT-DPP with an additional penalty term that fires when the
predicted tail-latency (p99) of the target workload would exceed a
user-defined SLO threshold.  The acquisition function becomes:

    score(c) = lambda_div  * marginal_logdet_gain(c)
             - lambda_risk * predicted_risk(c)
             - lambda_slo  * max(0, predicted_p99(c) - slo_threshold)

When no SLO threshold is given, the scheduler falls back to the plain
SIT-DPP objective.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sit.scheduling.utils import (
    compute_predicted_risk,
    marginal_gain,
)


def _estimate_p99(
    mean_risk: float,
    stderr_risk: float,
    beta: float,
) -> float:
    """Estimate the p99 tail latency from risk mean and stderr.

    Uses the UCB-like formula: p99 ~ mean + beta * stderr.
    When beta is large this provides a conservative upper bound on the
    tail.  With beta=2.0 (default for SLO mode) this roughly
    corresponds to a ~97.7 percentile under Gaussian assumptions.
    """
    return mean_risk + beta * stderr_risk


def sit_slo_schedule(
    target_name: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean: pd.DataFrame,
    tomography_stderr: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
    beta: float = 2.0,
    slo_threshold: Optional[float] = None,
    lambda_risk: float = 1.0,
    lambda_div: float = 1.0,
    lambda_slo: float = 2.0,
    epsilon: float = 1e-6,
) -> Tuple[List[str], Dict]:
    """Run the SIT-SLO greedy scheduler.

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
        UCB exploration weight for risk and p99 estimation.
    slo_threshold : float or None
        SLO latency threshold.  If None, no SLO penalty is applied
        (equivalent to SIT-DPP).
    lambda_risk : float
        Weight for the predicted-risk penalty.
    lambda_div : float
        Weight for the log-det diversity reward.
    lambda_slo : float
        Weight for the SLO-violation penalty.
    epsilon : float
        Regularisation constant for log-det computation.

    Returns
    -------
    selected : list of str
        Ordered list of selected spectator names.
    diagnostics : dict
        Per-step diagnostic information including scores, SLO status,
        and selection reasons.
    """
    # Map candidate names to kernel indices
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
        "slo_threshold": slo_threshold,
        "lambda_risk": lambda_risk,
        "lambda_div": lambda_div,
        "lambda_slo": lambda_slo,
        "slo_violations": 0,
    }

    n_pick = min(n_slots, len(candidate_spectators))

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
                div_gain = 0.0

            # --- Risk term (mean only) ---
            mean_risk = compute_predicted_risk(
                target_name=target_name,
                cotenant_names=selected + [s],
                tomography_mean=tomography_mean,
                tomography_stderr=None,
                beta=0.0,
            )

            # --- UCB risk (if beta > 0 and stderr available) ---
            ucb_risk = compute_predicted_risk(
                target_name=target_name,
                cotenant_names=selected + [s],
                tomography_mean=tomography_mean,
                tomography_stderr=tomography_stderr,
                beta=beta,
            )

            # --- SLO penalty term ---
            slo_penalty = 0.0
            predicted_p99 = None
            slo_violated = False

            if slo_threshold is not None and tomography_stderr is not None:
                # Compute stderr of the total risk for this configuration
                stderr_risk = compute_predicted_risk(
                    target_name=target_name,
                    cotenant_names=selected + [s],
                    tomography_mean=tomography_stderr,  # use stderr values as "mean" to sum them
                    tomography_stderr=None,
                    beta=0.0,
                )
                predicted_p99 = _estimate_p99(mean_risk, stderr_risk, beta)
                excess = predicted_p99 - slo_threshold
                if excess > 0:
                    slo_penalty = lambda_slo * excess
                    slo_violated = True

            # --- Combined score (higher is better) ---
            score = (
                lambda_div * div_gain
                - lambda_risk * ucb_risk
                - slo_penalty
            )

            step_info["candidates"][s] = {
                "div_gain": float(div_gain),
                "mean_risk": float(mean_risk),
                "ucb_risk": float(ucb_risk),
                "predicted_p99": float(predicted_p99) if predicted_p99 is not None else None,
                "slo_penalty": float(slo_penalty),
                "slo_violated": slo_violated,
                "score": float(score),
            }

            if score > best_score:
                best_score = score
                best_name = s

        if best_name is None:
            break

        # Track SLO violations
        cand_info = step_info["candidates"][best_name]
        if cand_info["slo_violated"]:
            diagnostics["slo_violations"] += 1

        step_info["selected"] = best_name
        step_info["selected_score"] = float(best_score)
        step_info["reason"] = (
            f"Best combined score: div_gain={cand_info['div_gain']:.4f}, "
            f"ucb_risk={cand_info['ucb_risk']:.4f}, "
            f"slo_penalty={cand_info['slo_penalty']:.4f}"
        )
        diagnostics["steps"].append(step_info)

        selected.append(best_name)
        if best_name in name_to_idx:
            selected_indices.append(name_to_idx[best_name])
        remaining.remove(best_name)

    diagnostics["final_selection"] = list(selected)
    return selected, diagnostics

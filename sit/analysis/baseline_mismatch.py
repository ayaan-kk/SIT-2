"""Baseline mismatch analysis: proving default models underpredict tail risk.

Naive predictor class:
  p99_pred = p99_ctrl * (1 + c * sim(T,S) * ell^q * h(rho))

This systematically underpredicts high tail inflation because:
- No spike mechanism
- No drift sensitivity
- Smooth additive assumption
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sit.simulator.interference_channels import (
    compute_cosine_similarity,
    DISTANCE_ATTENUATION,
)
from sit.simulator.workloads import Workload


class NaivePredictor:
    """Smooth additive interference predictor (baseline model).

    p99_pred = p99_ctrl * (1 + c * sim(T,S) * ell^q * h(rho))

    Parameters:
        c: scaling constant
        q: load exponent
    """

    def __init__(self, c: float = 1.5, q: float = 1.2):
        self.c = c
        self.q = q

    def predict(
        self,
        p99_ctrl: float,
        target: Workload,
        spectator: Workload,
        load: float,
        distance: str,
    ) -> float:
        """Predict p99 under co-location."""
        sim = compute_cosine_similarity(target, spectator)
        h_rho = DISTANCE_ATTENUATION.get(distance, 0.5)
        inflation = self.c * sim * (load ** self.q) * h_rho
        return p99_ctrl * (1.0 + inflation)


def compute_mismatch_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    high_risk_threshold_quantile: float = 0.75,
) -> Dict[str, float]:
    """Compute mismatch metrics between predicted and observed tail inflation.

    Returns:
        Dictionary with:
        - mae: mean absolute error
        - rmse: root mean squared error
        - underprediction_rate: fraction where predicted < observed
        - underprediction_rate_high_risk: same for high-risk conditions
        - max_underprediction: worst case underprediction
        - calibration_error_upper: mean error in upper quantile predictions
    """
    errors = predicted - observed
    abs_errors = np.abs(errors)

    # High-risk threshold based on observed values
    high_risk_thresh = np.percentile(observed, high_risk_threshold_quantile * 100)
    high_risk_mask = observed >= high_risk_thresh

    under_mask = predicted < observed

    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "underprediction_rate": float(np.mean(under_mask)),
        "max_underprediction": float(np.max(np.where(under_mask, observed - predicted, 0))),
        "mean_error": float(np.mean(errors)),
        "n_conditions": len(observed),
    }

    if np.sum(high_risk_mask) > 0:
        metrics["underprediction_rate_high_risk"] = float(
            np.mean(predicted[high_risk_mask] < observed[high_risk_mask])
        )
        high_risk_errors = predicted[high_risk_mask] - observed[high_risk_mask]
        metrics["calibration_error_upper"] = float(np.mean(np.abs(high_risk_errors)))
    else:
        metrics["underprediction_rate_high_risk"] = 0.0
        metrics["calibration_error_upper"] = 0.0

    return metrics


def run_baseline_mismatch(
    tomography_matrix: pd.DataFrame,
    ctrl_p99_matrix: pd.DataFrame,
    targets: Dict[str, Workload],
    spectators: Dict[str, Workload],
    loads: List[float],
    distances: List[str],
    predictor: NaivePredictor = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run baseline mismatch analysis.

    Args:
        tomography_matrix: observed delta p99 matrix (targets x spectators)
        ctrl_p99_matrix: control p99 per target
        targets: target workload objects
        spectators: spectator workload objects
        loads: load levels tested
        distances: distances tested
        predictor: naive predictor instance

    Returns:
        (scatter_df, metrics_dict)
        scatter_df has columns: target, spectator, observed, predicted, load, distance
    """
    if predictor is None:
        predictor = NaivePredictor()

    rows = []
    for t_name in tomography_matrix.index:
        if t_name not in targets:
            continue
        for s_name in tomography_matrix.columns:
            if s_name not in spectators:
                continue
            observed_delta = tomography_matrix.loc[t_name, s_name]
            if pd.isna(observed_delta):
                continue

            # Get control p99
            if t_name in ctrl_p99_matrix.index:
                p99_ctrl = float(ctrl_p99_matrix.loc[t_name].iloc[0])
            else:
                p99_ctrl = targets[t_name].base_latency_us * 3.0

            # Predict for a reference condition
            for load in loads:
                for dist in distances:
                    predicted_p99 = predictor.predict(
                        p99_ctrl, targets[t_name], spectators[s_name], load, dist
                    )
                    predicted_delta = predicted_p99 - p99_ctrl

                    rows.append({
                        "target": t_name,
                        "spectator": s_name,
                        "load": load,
                        "distance": dist,
                        "observed": observed_delta,
                        "predicted": predicted_delta,
                        "p99_ctrl": p99_ctrl,
                    })

    scatter_df = pd.DataFrame(rows)

    if len(scatter_df) > 0:
        metrics = compute_mismatch_metrics(
            scatter_df["observed"].values,
            scatter_df["predicted"].values,
        )
    else:
        metrics = {}

    return scatter_df, metrics

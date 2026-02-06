"""Core metric computation from raw latency samples.

Mathematical definitions:
- VaR_alpha(L) = inf{x : P(L <= x) >= alpha}  (quantile)
  p99 = VaR_0.99
- CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]  (expected shortfall)
  Empirical: sort samples, k = ceil(alpha*n), CVaR = mean(l[k:n])
- SLO violation rate: v = P(L > SLO), vhat = mean(1[l_i > SLO])
- SSI: s * log(1 + max(0, log(p99_t/p99_c)) + w * max(0, log(CVaR_t/CVaR_c)))
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_p99(samples: np.ndarray) -> float:
    """Compute 99th percentile (VaR_0.99)."""
    return float(np.percentile(samples, 99))


def compute_p95(samples: np.ndarray) -> float:
    """Compute 95th percentile (VaR_0.95)."""
    return float(np.percentile(samples, 95))


def compute_cvar(samples: np.ndarray, alpha: float = 0.99) -> float:
    """Compute CVaR (Expected Shortfall) at level alpha.

    CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]
    Empirical: sort samples l(1)<=...<=l(n), k=ceil(alpha*n),
    CVaR = mean(l[k:n])
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    k = int(np.ceil(alpha * n))
    if k >= n:
        k = n - 1
    tail = sorted_samples[k:]
    if len(tail) == 0:
        return float(sorted_samples[-1])
    return float(np.mean(tail))


def compute_cvar99(samples: np.ndarray) -> float:
    """CVaR at alpha=0.99."""
    return compute_cvar(samples, 0.99)


def compute_cvar95(samples: np.ndarray) -> float:
    """CVaR at alpha=0.95."""
    return compute_cvar(samples, 0.95)


def compute_slo_violation_rate(samples: np.ndarray, slo_threshold: float) -> float:
    """Compute SLO violation rate: fraction of samples above threshold."""
    return float(np.mean(samples > slo_threshold))


def compute_ssi(
    p99_ctrl: float,
    p99_treat: float,
    cvar_ctrl: float,
    cvar_treat: float,
    s: float = 2.0,
    w: float = 0.5,
) -> float:
    """Compute Spectator Sensitivity Index (SSI).

    R_p99 = max(0, log(p99_treat / p99_ctrl))
    R_cvar = max(0, log(CVaR_treat / CVaR_ctrl))
    SSI = s * log(1 + R_p99 + w * R_cvar)
    """
    if p99_ctrl <= 0 or cvar_ctrl <= 0:
        return 0.0
    r_p99 = max(0.0, np.log(p99_treat / p99_ctrl))
    r_cvar = max(0.0, np.log(cvar_treat / cvar_ctrl))
    return float(s * np.log(1.0 + r_p99 + w * r_cvar))


def compute_trial_summary(samples: np.ndarray, slo_threshold: float = None) -> Dict:
    """Compute all metrics for a single trial's samples."""
    result = {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "p95": compute_p95(samples),
        "p99": compute_p99(samples),
        "cvar95": compute_cvar95(samples),
        "cvar99": compute_cvar99(samples),
        "n_samples": len(samples),
    }
    if slo_threshold is not None:
        result["slo_violation_rate"] = compute_slo_violation_rate(samples, slo_threshold)
    return result


def compute_effect_sizes(
    ctrl_summaries: pd.DataFrame,
    treat_summaries: pd.DataFrame,
) -> Dict[str, float]:
    """Compute treatment effects (deltas) across metrics."""
    effects = {}
    for metric in ["mean", "p95", "p99", "cvar95", "cvar99"]:
        if metric in ctrl_summaries.columns and metric in treat_summaries.columns:
            effects[f"delta_{metric}"] = (
                float(treat_summaries[metric].mean()) -
                float(ctrl_summaries[metric].mean())
            )
    if "slo_violation_rate" in ctrl_summaries.columns:
        effects["delta_slo_viol"] = (
            float(treat_summaries["slo_violation_rate"].mean()) -
            float(ctrl_summaries["slo_violation_rate"].mean())
        )
    return effects


def aggregate_scheduling_results(
    results_list: List[Dict],
) -> pd.DataFrame:
    """Aggregate scheduling evaluation results into a summary DataFrame."""
    rows = []
    for r in results_list:
        row = {
            "scheduler": r["scheduler"],
            "target": r["target"],
            "device": r["device"],
            "load": r["load"],
            "distance": r["distance"],
            "regime": r["regime"],
            "seed": r.get("seed", 0),
        }
        for metric in ["mean", "p95", "p99", "cvar95", "cvar99", "slo_violation_rate"]:
            if metric in r:
                row[metric] = r[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def recompute_metrics_from_raw(
    raw_samples: np.ndarray,
    slo_threshold: float = None,
) -> Dict[str, float]:
    """Recompute all metrics from raw samples for QA verification."""
    return compute_trial_summary(raw_samples, slo_threshold)

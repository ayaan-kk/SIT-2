"""Worst-case analysis: top 10% hardest conditions.

Defines "hardest conditions" as the top 10% by Random scheduler p99
under adversarial regime, then evaluates all schedulers on that same condition set.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def identify_hardest_conditions(
    results_df: pd.DataFrame,
    scheduler_name: str = "random",
    metric: str = "p99",
    quantile: float = 0.90,
    regime_filter: Optional[str] = "adversarial",
) -> pd.DataFrame:
    """Identify the hardest conditions based on a reference scheduler.

    Args:
        results_df: full scheduling results with columns including
            scheduler, target, device, load, distance, regime, and metric columns
        scheduler_name: reference scheduler for ranking
        metric: metric to rank by
        quantile: fraction to consider "hardest" (0.9 = top 10%)
        regime_filter: if set, filter to this regime first

    Returns:
        DataFrame of condition identifiers for the hardest conditions.
    """
    ref = results_df[results_df["scheduler"] == scheduler_name].copy()

    if regime_filter and "regime" in ref.columns:
        ref = ref[ref["regime"] == regime_filter]

    if len(ref) == 0:
        return pd.DataFrame()

    threshold = ref[metric].quantile(quantile)
    hard_conditions = ref[ref[metric] >= threshold]

    condition_cols = ["target", "device", "load", "distance", "regime", "seed"]
    condition_cols = [c for c in condition_cols if c in hard_conditions.columns]

    return hard_conditions[condition_cols].drop_duplicates()


def evaluate_on_hardest(
    results_df: pd.DataFrame,
    hard_conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all schedulers on the hardest conditions.

    Returns DataFrame with scheduler-level summary for hard conditions only.
    """
    condition_cols = [c for c in hard_conditions.columns if c in results_df.columns]

    if len(condition_cols) == 0 or len(hard_conditions) == 0:
        return pd.DataFrame()

    # Merge to filter
    merged = results_df.merge(hard_conditions, on=condition_cols, how="inner")

    if len(merged) == 0:
        return pd.DataFrame()

    # Aggregate per scheduler
    metrics = ["mean", "p95", "p99", "cvar99"]
    metrics = [m for m in metrics if m in merged.columns]

    summary = merged.groupby("scheduler")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    return summary


def compute_blowup_avoidance(
    results_df: pd.DataFrame,
    hard_conditions: pd.DataFrame,
    sit_scheduler: str = "sit_dpp",
    baseline_scheduler: str = "random",
    metric: str = "p99",
) -> Dict[str, float]:
    """Compute blowup avoidance: how much SIT reduces worst-case metric.

    Returns dict with:
    - baseline_worst_mean: mean of metric for baseline on hard conditions
    - sit_worst_mean: mean of metric for SIT on hard conditions
    - reduction_pct: percentage reduction
    - max_blowup_baseline: max metric value for baseline
    - max_blowup_sit: max metric value for SIT
    """
    condition_cols = [c for c in hard_conditions.columns if c in results_df.columns]

    baseline_hard = results_df[results_df["scheduler"] == baseline_scheduler].merge(
        hard_conditions, on=condition_cols, how="inner"
    )
    sit_hard = results_df[results_df["scheduler"] == sit_scheduler].merge(
        hard_conditions, on=condition_cols, how="inner"
    )

    if len(baseline_hard) == 0 or len(sit_hard) == 0:
        return {"error": "No matching conditions found"}

    baseline_mean = float(baseline_hard[metric].mean())
    sit_mean = float(sit_hard[metric].mean())

    return {
        "baseline_worst_mean": baseline_mean,
        "sit_worst_mean": sit_mean,
        "reduction_pct": (baseline_mean - sit_mean) / baseline_mean * 100 if baseline_mean > 0 else 0.0,
        "max_blowup_baseline": float(baseline_hard[metric].max()),
        "max_blowup_sit": float(sit_hard[metric].max()),
        "n_hard_conditions": len(hard_conditions),
    }

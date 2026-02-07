"""Utilization, throughput, and Pareto frontier analysis.

Answers the question "why not just static partition?" by showing that
static partitioning sacrifices utilization and throughput.  Static
resource partitioning (Intel CAT, cgroups, etc.) pins capacity to each
partition; unused capacity in one partition cannot be reclaimed by
others.  Empirically this wastes ~40 % of allocatable capacity, which
we model with a fixed *partition_penalty* of 0.6.

Key metrics:
- utilization  = n_cotenants / n_total_slots  (* partition_penalty for static)
- throughput   = n_cotenants / mean_latency * 1e6  (normalised req/s)
- Pareto front = set of schedulers that are not dominated on the
                 (tail-latency, utilization/throughput) plane
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fraction of capacity retained under static partitioning.  When you
#: statically carve out LLC ways / memory bandwidth via Intel CAT or
#: cgroup limits, unused capacity in one partition is stranded.  A
#: typical partition wastes ~40 % capacity → effective multiplier 0.6.
STATIC_PARTITION_PENALTY: float = 0.6


# ---------------------------------------------------------------------------
# Per-row utilization and throughput
# ---------------------------------------------------------------------------

def compute_utilization(
    sched_df: pd.DataFrame,
    n_total_slots: int,
    partition_penalty: float = STATIC_PARTITION_PENALTY,
) -> pd.DataFrame:
    """Add a ``utilization`` column to *sched_df*.

    For **static_partition** rows the effective utilization is penalised:

        utilization = (n_cotenants / n_total_slots) * partition_penalty

    For all other schedulers:

        utilization = n_cotenants / n_total_slots

    Parameters
    ----------
    sched_df : pd.DataFrame
        Scheduling results.  Must contain ``scheduler`` and ``n_cotenants``.
    n_total_slots : int
        Total available co-location slots on the device.
    partition_penalty : float, optional
        Effective capacity multiplier for static partitioning (default 0.6).

    Returns
    -------
    pd.DataFrame
        A copy of *sched_df* with the ``utilization`` column appended.
    """
    df = sched_df.copy()
    if n_total_slots <= 0:
        df["utilization"] = 0.0
        return df

    raw_util = df["n_cotenants"] / n_total_slots

    is_static = df["scheduler"].str.lower() == "static_partition"
    df["utilization"] = np.where(is_static, raw_util * partition_penalty, raw_util)

    return df


def compute_throughput(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``throughput`` column to *sched_df*.

    Throughput is defined as normalised requests per second:

        throughput = n_cotenants / mean_latency * 1e6

    where *mean_latency* is in microseconds and the factor 1e6 converts
    to a per-second rate.

    Parameters
    ----------
    sched_df : pd.DataFrame
        Must contain ``n_cotenants`` and ``mean`` (mean latency in us).

    Returns
    -------
    pd.DataFrame
        A copy of *sched_df* with the ``throughput`` column appended.
    """
    df = sched_df.copy()
    mean_latency = df["mean"].values.astype(float)
    n_cotenants = df["n_cotenants"].values.astype(float)

    # Avoid division by zero: where mean_latency <= 0 set throughput to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.where(mean_latency > 0, n_cotenants / mean_latency * 1e6, 0.0)

    df["throughput"] = tp
    return df


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    minimize_x: bool = True,
    minimize_y: bool = True,
) -> np.ndarray:
    """Return row indices of Pareto-optimal points.

    A point *p* dominates *q* iff *p* is at least as good in every
    objective and strictly better in at least one.  Points that are not
    dominated by any other point form the Pareto frontier.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least *x_col* and *y_col*.
    x_col, y_col : str
        Column names for the two objectives.
    minimize_x, minimize_y : bool
        Whether lower values are better for each objective.

    Returns
    -------
    np.ndarray
        Integer array of row indices (positional) on the Pareto frontier.
    """
    x = df[x_col].values.astype(float).copy()
    y = df[y_col].values.astype(float).copy()

    # Flip signs so that "better" always means *smaller*.
    if not minimize_x:
        x = -x
    if not minimize_y:
        y = -y

    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Does j dominate i?  (j <= i in both and strictly < in at least one)
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break

    return np.where(is_pareto)[0]


# ---------------------------------------------------------------------------
# Pareto summary table
# ---------------------------------------------------------------------------

def compute_pareto_summary(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a per-scheduler summary with Pareto-optimality flags.

    The Pareto frontier is computed on the (mean_p99, mean_utilization)
    plane where we want to *minimise* p99 and *maximise* utilization.

    Returns
    -------
    pd.DataFrame
        Columns: scheduler, mean_p99, mean_cvar99, mean_utilization,
        mean_throughput, is_pareto_optimal, slo_miss_rate.
    """
    agg: dict = {
        "p99": "mean",
        "cvar99": "mean",
    }
    if "utilization" in sched_df.columns:
        agg["utilization"] = "mean"
    if "throughput" in sched_df.columns:
        agg["throughput"] = "mean"
    if "slo_violation_rate" in sched_df.columns:
        agg["slo_violation_rate"] = "mean"

    summary = sched_df.groupby("scheduler").agg(agg).reset_index()

    # Rename for clarity
    rename_map = {
        "p99": "mean_p99",
        "cvar99": "mean_cvar99",
    }
    if "utilization" in summary.columns:
        rename_map["utilization"] = "mean_utilization"
    if "throughput" in summary.columns:
        rename_map["throughput"] = "mean_throughput"
    if "slo_violation_rate" in summary.columns:
        rename_map["slo_violation_rate"] = "slo_miss_rate"

    summary = summary.rename(columns=rename_map)

    # Fill columns that may be missing with NaN so Pareto logic still works.
    if "mean_utilization" not in summary.columns:
        summary["mean_utilization"] = np.nan
    if "mean_throughput" not in summary.columns:
        summary["mean_throughput"] = np.nan
    if "slo_miss_rate" not in summary.columns:
        summary["slo_miss_rate"] = np.nan

    # Pareto frontier: minimise p99, maximise utilization
    pareto_idx = compute_pareto_frontier(
        summary,
        x_col="mean_p99",
        y_col="mean_utilization",
        minimize_x=True,
        minimize_y=False,
    )

    summary["is_pareto_optimal"] = False
    summary.loc[summary.index[pareto_idx], "is_pareto_optimal"] = True

    return summary


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def compute_efficiency_metrics(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-scheduler efficiency ratios.

    Returns a DataFrame with columns:
    - scheduler
    - latency_per_utilization_unit   = mean_p99 / mean_utilization
    - tail_risk_per_throughput       = mean_cvar99 / mean_throughput

    Lower values are better for both ratios.
    """
    agg: dict = {"p99": "mean", "cvar99": "mean"}
    if "utilization" in sched_df.columns:
        agg["utilization"] = "mean"
    if "throughput" in sched_df.columns:
        agg["throughput"] = "mean"

    grouped = sched_df.groupby("scheduler").agg(agg).reset_index()

    # latency_per_utilization_unit
    if "utilization" in grouped.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            grouped["latency_per_utilization_unit"] = np.where(
                grouped["utilization"] > 0,
                grouped["p99"] / grouped["utilization"],
                np.inf,
            )
    else:
        grouped["latency_per_utilization_unit"] = np.nan

    # tail_risk_per_throughput
    if "throughput" in grouped.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            grouped["tail_risk_per_throughput"] = np.where(
                grouped["throughput"] > 0,
                grouped["cvar99"] / grouped["throughput"],
                np.inf,
            )
    else:
        grouped["tail_risk_per_throughput"] = np.nan

    return grouped[
        [
            "scheduler",
            "latency_per_utilization_unit",
            "tail_risk_per_throughput",
        ]
    ]

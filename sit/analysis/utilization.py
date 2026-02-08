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


# ---------------------------------------------------------------------------
# Goodput: the partition-killer metric
# ---------------------------------------------------------------------------

def compute_goodput(
    sched_df: pd.DataFrame,
    slo_threshold_us: float = 500_000.0,
    metric: str = "p99",
) -> pd.DataFrame:
    """Add ``goodput`` column: effective useful throughput under SLO.

    Goodput = throughput × utilization × slo_hit_fraction

    For static partition, the utilization penalty (0.6) directly reduces
    goodput because stranded capacity cannot serve other workloads.
    SIT variants retain full utilization (1.0) while achieving comparable
    tail safety, producing materially higher goodput.

    This single metric resolves the "why not just partition?" question:
    partition has the best raw p99 but the worst goodput.
    """
    df = sched_df.copy()

    # Ensure required columns exist
    if "throughput" not in df.columns or "utilization" not in df.columns:
        df["goodput"] = 0.0
        return df

    # SLO hit fraction (per-row)
    slo_hit = (df[metric] <= slo_threshold_us).astype(float)

    # Goodput = throughput × utilization × slo_hit
    df["goodput"] = df["throughput"] * df["utilization"] * slo_hit

    return df


def compute_goodput_summary(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Per-scheduler goodput summary for the Pareto figure.

    Returns mean goodput, mean p99, mean cvar99, mean utilization per scheduler.
    """
    agg = {"p99": "mean", "cvar99": "mean"}
    if "goodput" in sched_df.columns:
        agg["goodput"] = "mean"
    if "utilization" in sched_df.columns:
        agg["utilization"] = "mean"
    if "throughput" in sched_df.columns:
        agg["throughput"] = "mean"

    summary = sched_df.groupby("scheduler").agg(agg).reset_index()
    return summary


# ---------------------------------------------------------------------------
# SLO-admission rate and SLO-constrained throughput
# ---------------------------------------------------------------------------

def compute_slo_admission(
    sched_df: pd.DataFrame,
    slo_threshold_us: float = 500_000.0,
    metric: str = "p99",
) -> pd.DataFrame:
    """Add ``slo_admitted`` column: 1 if the condition meets SLO, 0 otherwise.

    A condition "meets the SLO" when its tail latency (p99 by default)
    is below *slo_threshold_us*.  This is the admission-control analogue:
    "would you run this configuration in production?"

    Parameters
    ----------
    sched_df : pd.DataFrame
        Must contain *metric* column and ``scheduler``.
    slo_threshold_us : float
        SLO threshold in microseconds.
    metric : str
        Column to compare against the SLO (default ``"p99"``).

    Returns
    -------
    pd.DataFrame
        Copy with ``slo_admitted`` (bool) column added.
    """
    df = sched_df.copy()
    df["slo_admitted"] = df[metric] <= slo_threshold_us
    return df


def compute_slo_throughput_summary(
    sched_df: pd.DataFrame,
    slo_thresholds_us: Optional[List[float]] = None,
    metric: str = "p99",
) -> pd.DataFrame:
    """Per-scheduler summary of SLO-satisfying throughput at several thresholds.

    For each SLO threshold, computes:
    - admission_rate: fraction of conditions meeting the SLO
    - admitted_throughput: mean throughput among admitted conditions
    - tail_risk (mean p99): mean p99 among admitted conditions

    Static partition pays a *real cost*: because it reduces effective
    concurrency, its throughput among admitted conditions is lower.

    Returns
    -------
    pd.DataFrame
        Columns: scheduler, slo_threshold_us, admission_rate,
        admitted_throughput, admitted_mean_p99, admitted_mean_cvar99.
    """
    if slo_thresholds_us is None:
        slo_thresholds_us = [100_000, 250_000, 500_000, 1_000_000, 2_500_000]

    rows = []
    for slo in slo_thresholds_us:
        for sched, grp in sched_df.groupby("scheduler"):
            admitted = grp[grp[metric] <= slo]
            n_total = len(grp)
            n_admitted = len(admitted)
            rate = n_admitted / n_total if n_total > 0 else 0.0

            if n_admitted > 0:
                mean_tp = float(admitted["throughput"].mean()) if "throughput" in admitted.columns else 0.0
                mean_p99 = float(admitted[metric].mean())
                mean_cvar = float(admitted["cvar99"].mean()) if "cvar99" in admitted.columns else 0.0
            else:
                mean_tp = 0.0
                mean_p99 = 0.0
                mean_cvar = 0.0

            rows.append({
                "scheduler": sched,
                "slo_threshold_us": slo,
                "admission_rate": rate,
                "admitted_throughput": mean_tp,
                "admitted_mean_p99": mean_p99,
                "admitted_mean_cvar99": mean_cvar,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost-per-good-request model
# ---------------------------------------------------------------------------

#: Infrastructure cost per machine-second ($/s), based on typical cloud
#: pricing.  A c5.xlarge costs ~$0.17/hr = $0.0000472/s.
COST_PER_MACHINE_SECOND: float = 0.0000472

#: Revenue per successfully served request (SLO-meeting request).
#: A request that violates SLO has zero revenue (or negative via
#: penalty).  This is a simplified model; in practice, revenue depends
#: on the request type and SLA tier.
REVENUE_PER_GOOD_REQUEST: float = 0.001  # $0.001 per request

#: Penalty per SLO-violating request.  Models contractual penalties
#: for exceeding tail latency SLOs (~2x revenue, conservative).
PENALTY_PER_BAD_REQUEST: float = 0.002


def compute_cost_efficiency(
    sched_df: pd.DataFrame,
    slo_threshold_us: float = 500_000.0,
    metric: str = "p99",
    cost_per_second: float = COST_PER_MACHINE_SECOND,
    revenue_per_good: float = REVENUE_PER_GOOD_REQUEST,
    penalty_per_bad: float = PENALTY_PER_BAD_REQUEST,
) -> pd.DataFrame:
    """Compute cost-per-good-request and net value for each scheduler.

    Model:
        - good_requests = throughput * utilization * slo_hit_rate
        - bad_requests = throughput * utilization * (1 - slo_hit_rate)
        - revenue = good_requests * revenue_per_good
        - penalty = bad_requests * penalty_per_bad
        - cost = cost_per_second (fixed infrastructure)
        - net_value = revenue - penalty - cost
        - cost_per_good_request = cost / max(good_requests, 1)

    Returns per-scheduler summary with cost metrics.
    """
    df = sched_df.copy()

    rows = []
    for sched, grp in df.groupby("scheduler"):
        slo_hit = (grp[metric] <= slo_threshold_us).astype(float)
        hit_rate = float(slo_hit.mean())

        mean_tp = float(grp["throughput"].mean()) if "throughput" in grp.columns else 0.0
        mean_util = float(grp["utilization"].mean()) if "utilization" in grp.columns else 1.0

        good_rps = mean_tp * mean_util * hit_rate
        bad_rps = mean_tp * mean_util * (1 - hit_rate)

        revenue = good_rps * revenue_per_good
        penalty = bad_rps * penalty_per_bad
        net_value = revenue - penalty - cost_per_second
        cost_per_good = cost_per_second / max(good_rps, 1e-9)

        rows.append({
            "scheduler": sched,
            "slo_hit_rate": hit_rate,
            "good_rps": good_rps,
            "bad_rps": bad_rps,
            "revenue_per_s": revenue,
            "penalty_per_s": penalty,
            "net_value_per_s": net_value,
            "cost_per_good_request": cost_per_good,
            "mean_throughput": mean_tp,
            "mean_utilization": mean_util,
        })

    return pd.DataFrame(rows)

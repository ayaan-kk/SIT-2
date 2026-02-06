"""Table generation for SIT results.

Generates CSV tables from experimental results:
- summary_overall.csv
- summary_by_regime.csv
- summary_by_load.csv
- summary_by_distance.csv
- tomography_matrix_mean.csv
- tomography_matrix_ci.csv
- sparsity_stats.csv
- baseline_mismatch_metrics.csv
- worst_case_metrics.csv
- ablations.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def save_table(df: pd.DataFrame, name: str, output_dir: str = "results/tables"):
    """Save a DataFrame as CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/{name}.csv", index=True)


def generate_summary_overall(
    sched_results: pd.DataFrame,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Generate overall summary across all schedulers."""
    metrics = ["mean", "p95", "p99", "cvar99"]
    metrics = [m for m in metrics if m in sched_results.columns]

    if "slo_violation_rate" in sched_results.columns:
        metrics.append("slo_violation_rate")

    summary = sched_results.groupby("scheduler")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_table(summary, "summary_overall", output_dir)
    return summary


def generate_summary_by_regime(
    sched_results: pd.DataFrame,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Generate summary by regime."""
    metrics = ["mean", "p95", "p99", "cvar99"]
    metrics = [m for m in metrics if m in sched_results.columns]

    summary = sched_results.groupby(["scheduler", "regime"])[metrics].agg(
        ["mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_table(summary, "summary_by_regime", output_dir)
    return summary


def generate_summary_by_load(
    sched_results: pd.DataFrame,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Generate summary by load level."""
    metrics = ["mean", "p95", "p99", "cvar99"]
    metrics = [m for m in metrics if m in sched_results.columns]

    summary = sched_results.groupby(["scheduler", "load"])[metrics].agg(
        ["mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_table(summary, "summary_by_load", output_dir)
    return summary


def generate_summary_by_distance(
    sched_results: pd.DataFrame,
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Generate summary by distance."""
    metrics = ["mean", "p95", "p99", "cvar99"]
    metrics = [m for m in metrics if m in sched_results.columns]

    summary = sched_results.groupby(["scheduler", "distance"])[metrics].agg(
        ["mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_table(summary, "summary_by_distance", output_dir)
    return summary


def save_tomography_matrices(
    mean_matrix: pd.DataFrame,
    ci_lower: Optional[pd.DataFrame],
    ci_upper: Optional[pd.DataFrame],
    output_dir: str = "results/tables",
):
    """Save tomography matrices as CSV."""
    save_table(mean_matrix, "tomography_matrix_mean", output_dir)
    if ci_lower is not None and ci_upper is not None:
        # Combine CI into a single table
        ci_df = pd.DataFrame(index=mean_matrix.index, columns=mean_matrix.columns)
        for t in mean_matrix.index:
            for s in mean_matrix.columns:
                m = mean_matrix.loc[t, s]
                lo = ci_lower.loc[t, s]
                hi = ci_upper.loc[t, s]
                ci_df.loc[t, s] = f"{m:.1f} [{lo:.1f}, {hi:.1f}]"
        save_table(ci_df, "tomography_matrix_ci", output_dir)


def save_sparsity_stats(
    sparsity_df: pd.DataFrame,
    output_dir: str = "results/tables",
):
    """Save sparsity statistics."""
    save_table(sparsity_df, "sparsity_stats", output_dir)


def save_mismatch_metrics(
    metrics: Dict[str, float],
    output_dir: str = "results/tables",
):
    """Save baseline mismatch metrics."""
    df = pd.DataFrame([metrics])
    save_table(df, "baseline_mismatch_metrics", output_dir)


def save_worst_case_metrics(
    worst_case: Dict,
    output_dir: str = "results/tables",
):
    """Save worst-case analysis metrics."""
    df = pd.DataFrame([worst_case])
    save_table(df, "worst_case_metrics", output_dir)


def save_ablations(
    ablation_df: pd.DataFrame,
    output_dir: str = "results/tables",
):
    """Save ablation results."""
    save_table(ablation_df, "ablations", output_dir)


def generate_all_tables(results: Dict, output_dir: str = "results/tables"):
    """Generate all tables from results dictionary."""
    if "sched_results" in results:
        generate_summary_overall(results["sched_results"], output_dir)
        generate_summary_by_regime(results["sched_results"], output_dir)
        if "load" in results["sched_results"].columns:
            generate_summary_by_load(results["sched_results"], output_dir)
        if "distance" in results["sched_results"].columns:
            generate_summary_by_distance(results["sched_results"], output_dir)

    if "tomo_mean" in results:
        save_tomography_matrices(
            results["tomo_mean"],
            results.get("tomo_ci_lower"),
            results.get("tomo_ci_upper"),
            output_dir,
        )

    if "sparsity_df" in results:
        save_sparsity_stats(results["sparsity_df"], output_dir)

    if "mismatch_metrics" in results:
        save_mismatch_metrics(results["mismatch_metrics"], output_dir)

    if "worst_case_summary" in results:
        save_worst_case_metrics(results["worst_case_summary"], output_dir)

    if "ablation_df" in results:
        save_ablations(results["ablation_df"], output_dir)

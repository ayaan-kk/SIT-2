"""Figure generation for SIT results.

All figures saved as both PNG and PDF with consistent naming.
Figures:
F1: IRBS bias demo
F2: Distance/load ladder showing tail explosion
F3: Tomography heatmap with CI
F4: Sparse recovery curve
F5: Baseline mismatch scatter
F6: Scheduler comparison bars
F7: Worst-case blowup avoidance
F8: Ablation panel
F9: QA summary
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Optional


def _save_fig(fig, name: str, output_dir: str):
    """Save figure as PNG and PDF."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f1_irbs_bias_demo(
    bias_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F1: IRBS bias demo - naive vs interleaved under drift.

    bias_df has columns: repeat, naive_tau, irbs_tau, true_tau
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    true_tau = bias_df["true_tau"].iloc[0]

    # Left: histogram of tau estimates
    ax = axes[0]
    ax.hist(bias_df["naive_tau"], bins=25, alpha=0.6, label="Naive (A-then-B)",
            color="#d62728", density=True)
    ax.hist(bias_df["irbs_tau"], bins=25, alpha=0.6, label="IRBS (Interleaved)",
            color="#2ca02c", density=True)
    ax.axvline(true_tau, color="black", linestyle="--", linewidth=2, label=f"True τ={true_tau:.1f}")
    ax.set_xlabel("Estimated Treatment Effect (Δp99, μs)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Effect Estimates Under Drift")
    ax.legend()

    # Right: bias comparison
    ax = axes[1]
    naive_bias = bias_df["naive_tau"].mean() - true_tau
    irbs_bias = bias_df["irbs_tau"].mean() - true_tau
    naive_std = bias_df["naive_tau"].std()
    irbs_std = bias_df["irbs_tau"].std()

    x = [0, 1]
    biases = [naive_bias, irbs_bias]
    stds = [naive_std, irbs_std]
    colors = ["#d62728", "#2ca02c"]
    labels = ["Naive", "IRBS"]

    bars = ax.bar(x, biases, yerr=stds, color=colors, alpha=0.7, capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bias (Estimated - True Effect)")
    ax.set_title("Measurement Bias Comparison")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

    fig.suptitle("F1: IRBS Drift Bias Demonstration [Simulation]", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F1_irbs_bias_demo", output_dir)


def plot_f2_phenomenon(
    results_df: pd.DataFrame,
    target_name: str = "kv_lookup",
    spectator_name: str = "cache_thrash",
    output_dir: str = "results/figures",
):
    """F2: Distance and load ladders showing tail explosion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    subset = results_df[
        (results_df["target"] == target_name) &
        (results_df["spectator"] == spectator_name)
    ].copy() if "spectator" in results_df.columns else results_df[
        results_df["target"] == target_name
    ].copy()

    # Left: load ladder (fixed distance)
    ax = axes[0]
    if "load" in subset.columns:
        for regime in ["benign", "structured", "adversarial"]:
            regime_data = subset[subset["regime"] == regime] if "regime" in subset.columns else subset
            if len(regime_data) == 0:
                continue
            load_agg = regime_data.groupby("load")[["p99", "cvar99"]].mean().reset_index()
            if len(load_agg) > 0:
                ax.plot(load_agg["load"], load_agg["p99"], "o-", label=f"p99 ({regime})")
                ax.plot(load_agg["load"], load_agg["cvar99"], "s--", label=f"CVaR99 ({regime})", alpha=0.7)
    ax.set_xlabel("Load Level")
    ax.set_ylabel("Latency (μs)")
    ax.set_title(f"Load Ladder: {target_name} + {spectator_name}")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Right: distance ladder (fixed load)
    ax = axes[1]
    dist_order = ["same_core", "same_llc", "same_numa", "cross_numa", "cross_socket"]
    if "distance" in subset.columns:
        for regime in ["benign", "structured", "adversarial"]:
            regime_data = subset[subset["regime"] == regime] if "regime" in subset.columns else subset
            if len(regime_data) == 0:
                continue
            dist_agg = regime_data.groupby("distance")[["p99", "cvar99"]].mean()
            # Reorder
            available = [d for d in dist_order if d in dist_agg.index]
            if len(available) > 0:
                dist_agg = dist_agg.loc[available]
                ax.plot(range(len(available)), dist_agg["p99"], "o-", label=f"p99 ({regime})")
                ax.plot(range(len(available)), dist_agg["cvar99"], "s--", label=f"CVaR99 ({regime})", alpha=0.7)
                ax.set_xticks(range(len(available)))
                ax.set_xticklabels(available, rotation=30, fontsize=8)
    ax.set_xlabel("Placement Distance")
    ax.set_ylabel("Latency (μs)")
    ax.set_title(f"Distance Ladder: {target_name} + {spectator_name}")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    fig.suptitle("F2: Tail Explosion Under High-Risk Spectator [Simulation]", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F2_phenomenon_ladder", output_dir)


def plot_f3_tomography_heatmap(
    mean_matrix: pd.DataFrame,
    ci_lower: Optional[pd.DataFrame] = None,
    ci_upper: Optional[pd.DataFrame] = None,
    output_dir: str = "results/figures",
):
    """F3: Tomography heatmap with CI summaries."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = mean_matrix.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(mean_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(mean_matrix.index, fontsize=8)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{val:.0f}"
            if ci_lower is not None and ci_upper is not None:
                lo = ci_lower.values[i, j]
                hi = ci_upper.values[i, j]
                text = f"{val:.0f}\n[{lo:.0f},{hi:.0f}]"
            color = "white" if val > np.nanmedian(data) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Δp99 (μs)")
    ax.set_xlabel("Spectator Workload")
    ax.set_ylabel("Target Workload")
    fig.suptitle("F3: Interference Tomography Map with Uncertainty [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F3_tomography_heatmap", output_dir)


def plot_f4_sparse_recovery(
    recovery_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F4: Sparse recovery curve - top-1 and top-3 vs trials."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for k_val in recovery_df["k"].unique():
        kdata = recovery_df[recovery_df["k"] == k_val]
        ax.plot(kdata["trials_per_spectator"], kdata["recovery_probability"],
                "o-", label=f"Top-{k_val} recovery", linewidth=2, markersize=6)

    ax.set_xlabel("Trials per Spectator")
    ax.set_ylabel("Recovery Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("F4: Sparse Interferer Recovery vs. Sampling [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F4_sparse_recovery", output_dir)


def plot_f5_baseline_mismatch(
    scatter_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F5: Baseline mismatch scatter - predicted vs observed."""
    fig, ax = plt.subplots(figsize=(8, 8))

    observed = scatter_df["observed"].values
    predicted = scatter_df["predicted"].values

    # Color by underprediction
    under = predicted < observed
    ax.scatter(observed[~under], predicted[~under], alpha=0.3, s=15, color="steelblue",
               label="Adequate prediction")
    ax.scatter(observed[under], predicted[under], alpha=0.5, s=20, color="red",
               label="Underprediction", marker="x")

    # Perfect prediction line
    max_val = max(np.max(observed), np.max(predicted))
    min_val = min(np.min(observed), np.min(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Observed Δp99 (μs)")
    ax.set_ylabel("Predicted Δp99 (μs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    under_rate = np.mean(under) * 100
    ax.text(0.05, 0.95, f"Underprediction rate: {under_rate:.1f}%",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))

    fig.suptitle("F5: Naive Model Mismatch - Underpredicts Tail Risk [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F5_baseline_mismatch", output_dir)


def plot_f6_scheduler_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F6: Scheduler comparison bars for p99 and CVaR across regimes."""
    regimes = [r for r in ["benign", "structured", "adversarial"] if r in summary_df["regime"].values]
    schedulers = summary_df["scheduler"].unique()

    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 6), sharey=True)
    if len(regimes) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(schedulers)))

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        regime_data = summary_df[summary_df["regime"] == regime]

        x = np.arange(len(schedulers))
        width = 0.35

        p99_vals = []
        cvar_vals = []
        for sched in schedulers:
            sdata = regime_data[regime_data["scheduler"] == sched]
            p99_vals.append(sdata["p99"].mean() if len(sdata) > 0 else 0)
            cvar_vals.append(sdata["cvar99"].mean() if "cvar99" in sdata.columns and len(sdata) > 0 else 0)

        bars1 = ax.bar(x - width / 2, p99_vals, width, label="p99", color=colors, alpha=0.8)
        bars2 = ax.bar(x + width / 2, cvar_vals, width, label="CVaR99", color=colors, alpha=0.5,
                       hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Regime: {regime}")
        ax.set_ylabel("Latency (μs)")
        if idx == 0:
            ax.legend(["p99", "CVaR99"])

    fig.suptitle("F6: Scheduler Comparison Across Regimes [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F6_scheduler_comparison", output_dir)


def plot_f7_worst_case(
    worst_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F7: Worst-case analysis - top 10% conditions, SIT vs baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = worst_df.columns[0] if "scheduler" not in worst_df.columns else None

    if "scheduler" in worst_df.columns:
        # Bar chart
        sched_names = worst_df["scheduler"].values
        p99_cols = [c for c in worst_df.columns if "p99_mean" in c]
        if p99_cols:
            metric_col = p99_cols[0]
        elif "p99" in worst_df.columns:
            metric_col = "p99"
        else:
            metric_col = worst_df.columns[1]

        colors = ["#d62728" if "random" in s else "#2ca02c" if "sit" in s else "#1f77b4"
                  for s in sched_names]
        ax.barh(range(len(sched_names)), worst_df[metric_col].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sched_names)))
        ax.set_yticklabels(sched_names, fontsize=9)
        ax.set_xlabel("p99 Latency (μs) on Hardest Conditions")
    else:
        # Simple table display
        ax.text(0.5, 0.5, str(worst_df.to_string()), transform=ax.transAxes,
                fontsize=8, va="center", ha="center", family="monospace")
        ax.axis("off")

    fig.suptitle("F7: Worst-Case Blowup Avoidance (Top 10% Conditions) [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F7_worst_case", output_dir)


def plot_f8_ablations(
    ablation_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F8: Ablation panel showing each component's contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if "variant" not in ablation_df.columns:
        # Fallback
        for ax in axes:
            ax.text(0.5, 0.5, "No ablation data", transform=ax.transAxes,
                    ha="center", va="center")
        _save_fig(fig, "F8_ablations", output_dir)
        return

    variants = ablation_df["variant"].values
    x = range(len(variants))

    # Left: p99
    ax = axes[0]
    if "p99" in ablation_df.columns:
        colors = ["#2ca02c" if "sit_dpp" == v else "#ff7f0e" for v in variants]
        ax.bar(x, ablation_df["p99"].values, color=colors, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("p99 Latency (μs)")
        ax.set_title("p99 Across Ablation Variants")

    # Right: CVaR99
    ax = axes[1]
    if "cvar99" in ablation_df.columns:
        colors = ["#2ca02c" if "sit_dpp" == v else "#ff7f0e" for v in variants]
        ax.bar(x, ablation_df["cvar99"].values, color=colors, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("CVaR99 Latency (μs)")
        ax.set_title("CVaR99 Across Ablation Variants")

    fig.suptitle("F8: Ablation Study - Component Contributions [Simulation]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F8_ablations", output_dir)


def plot_f9_qa_summary(
    qa_results: Dict[str, Dict],
    output_dir: str = "results/figures",
):
    """F9: QA summary table/plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    cell_text = []
    for test_name, result in qa_results.items():
        passed = result.get("passed", "N/A")
        message = result.get("message", "")
        status = "PASS" if passed else "FAIL"
        cell_text.append([test_name, status, message[:80]])

    if cell_text:
        table = ax.table(cellText=cell_text,
                         colLabels=["Test", "Status", "Details"],
                         cellLoc="left",
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)

        # Color cells
        for i, row in enumerate(cell_text):
            color = "#c8e6c9" if row[1] == "PASS" else "#ffcdd2"
            for j in range(3):
                table[i + 1, j].set_facecolor(color)

    fig.suptitle("F9: QA Check Summary [Simulation]", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "F9_qa_summary", output_dir)


def generate_all_figures(results: Dict, output_dir: str = "results/figures"):
    """Generate all figures from results dictionary."""
    if "bias_df" in results:
        plot_f1_irbs_bias_demo(results["bias_df"], output_dir)

    if "phenomenon_df" in results:
        target = results.get("phenomenon_target", "kv_lookup")
        spectator = results.get("phenomenon_spectator", "cache_thrash")
        plot_f2_phenomenon(results["phenomenon_df"], target, spectator, output_dir)

    if "tomo_mean" in results:
        plot_f3_tomography_heatmap(
            results["tomo_mean"],
            results.get("tomo_ci_lower"),
            results.get("tomo_ci_upper"),
            output_dir,
        )

    if "recovery_df" in results:
        plot_f4_sparse_recovery(results["recovery_df"], output_dir)

    if "mismatch_scatter" in results:
        plot_f5_baseline_mismatch(results["mismatch_scatter"], output_dir)

    if "sched_summary" in results:
        plot_f6_scheduler_comparison(results["sched_summary"], output_dir)

    if "worst_case_df" in results:
        plot_f7_worst_case(results["worst_case_df"], output_dir)

    if "ablation_df" in results:
        plot_f8_ablations(results["ablation_df"], output_dir)

    if "qa_results" in results:
        plot_f9_qa_summary(results["qa_results"], output_dir)

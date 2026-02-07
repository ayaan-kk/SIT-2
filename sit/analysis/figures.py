"""Publication-quality figure generation for SIT results.

All figures saved as both PNG (300 DPI) and PDF with consistent naming.
Designed for ISEF / publication presentation with professional styling.

Figures:
F1:  IRBS bias demo (violin + strip-box)
F2:  Distance/load ladder showing tail explosion (2x2 panel)
F3:  Tomography heatmap with CI and sorted spectators
F4:  Sparse recovery curve with confidence bands
F5:  Baseline mismatch (scatter + CDF, 2x1 panel)
F6:  Scheduler comparison bars (1x3 panel by regime)
F7:  Worst-case blowup avoidance (horizontal bar)
F8:  Ablation panel (2x2 with waterfall)
F9:  QA summary table
F10: Channel decomposition (stacked bar)
F11: Sensitivity analysis (2x2 panel)
F12: Real-system anchoring experiment
F13: Pareto frontier (tail safety vs utilization)
F14: Probe budget curve (error vs probes)
F15: Drift robustness (bias under drift)
F16: Simulator calibration against published benchmarks
F17: Tail ECDF (complementary CDF of p99 latencies)
F18: Quantile improvement heatmap (p99 reduction by target x device)
F19: CI coverage reliability diagram
F20: Overhead breakdown (horizontal bar)
F21: Ablation forest plot (component contribution)
F22: Tomography identifiability diagnostics
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "SIT_GREEN": "#2d6a4f",
    "RANDOM_RED": "#c1121f",
    "BASELINE_BLUE": "#1d3557",
    "WARN_ORANGE": "#e76f51",
    "NEUTRAL_GRAY": "#6c757d",
    "ACCENT_PURPLE": "#7209b7",
    # Regime-specific
    "BENIGN": "#2d6a4f",
    "STRUCTURED": "#e76f51",
    "ADVERSARIAL": "#c1121f",
    # Secondary accents
    "LIGHT_GREEN": "#95d5b2",
    "LIGHT_RED": "#ffb3b3",
    "LIGHT_BLUE": "#a8dadc",
    "LIGHT_ORANGE": "#f4a261",
    "DARK_TEXT": "#212529",
    "GRID_GRAY": "#dee2e6",
}

REGIME_COLORS = {
    "benign": COLORS["BENIGN"],
    "structured": COLORS["STRUCTURED"],
    "adversarial": COLORS["ADVERSARIAL"],
}

SCHEDULER_COLORS = {
    "sit_dpp": COLORS["SIT_GREEN"],
    "sit_ucb_dpp": "#40916c",
    "mean_greedy": COLORS["BASELINE_BLUE"],
    "similarity_avoidance": COLORS["ACCENT_PURPLE"],
    "linux_proxy": COLORS["NEUTRAL_GRAY"],
    "static_partition": "#adb5bd",
    "random": COLORS["RANDOM_RED"],
}

CHANNEL_COLORS = [
    "#264653",  # LLC
    "#2a9d8f",  # MEM_BW
    "#e9c46a",  # TLB
    "#f4a261",  # PREFETCH
    "#e76f51",  # NUMA
    "#c1121f",  # THERMAL
    "#7209b7",  # OS_FAULTS
]

CHANNEL_NAMES = ["LLC", "MEM_BW", "TLB", "PREFETCH", "NUMA", "THERMAL", "OS_FAULTS"]

# Preferred scheduler display order
_SCHED_ORDER = [
    "sit_dpp", "sit_ucb_dpp", "mean_greedy",
    "similarity_avoidance", "linux_proxy", "static_partition", "random",
]

# ---------------------------------------------------------------------------
# Global matplotlib defaults (publication quality)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.edgecolor": COLORS["DARK_TEXT"],
    "axes.labelcolor": COLORS["DARK_TEXT"],
    "xtick.color": COLORS["DARK_TEXT"],
    "ytick.color": COLORS["DARK_TEXT"],
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": COLORS["GRID_GRAY"],
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": COLORS["GRID_GRAY"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _save_fig(fig, name: str, output_dir: str):
    """Save figure as PNG (300 DPI) and PDF."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _add_watermark(fig, text="[Simulation]"):
    """Add a subtle diagonal watermark to the figure."""
    fig.text(
        0.5, 0.5, text,
        fontsize=48, color="#cccccc", alpha=0.18,
        ha="center", va="center", rotation=30,
        transform=fig.transFigure, zorder=0,
        fontweight="bold", fontstyle="italic",
    )


def _add_source_label(fig, text="SIT Simulator"):
    """Add a small source attribution in the bottom-right corner."""
    fig.text(
        0.99, 0.01, text,
        fontsize=7, color=COLORS["NEUTRAL_GRAY"], alpha=0.6,
        ha="right", va="bottom", transform=fig.transFigure,
        fontstyle="italic",
    )


def _nice_sched_name(name: str) -> str:
    """Return a human-readable scheduler name for labels."""
    mapping = {
        "sit_dpp": "SIT-DPP",
        "sit_ucb_dpp": "SIT-UCB-DPP",
        "mean_greedy": "Mean Greedy",
        "similarity_avoidance": "Similarity Avoid.",
        "linux_proxy": "Linux Proxy",
        "static_partition": "Static Partition",
        "random": "Random",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _ordered_schedulers(names):
    """Return scheduler names sorted in the preferred display order."""
    name_set = set(names)
    ordered = [s for s in _SCHED_ORDER if s in name_set]
    extras = sorted(name_set - set(ordered))
    return ordered + extras


def _sched_color(name: str) -> str:
    """Return the colour associated with a scheduler."""
    return SCHEDULER_COLORS.get(name, COLORS["NEUTRAL_GRAY"])


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Quick bootstrap CI for an array of values. Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = np.mean(arr[rng.integers(0, n, size=n)])
    return float(np.mean(arr)), float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def _mannwhitneyu_pvalue(a, b):
    """Two-sided Mann-Whitney U p-value.  Falls back to 1.0 on error."""
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return p
    except Exception:
        return 1.0


def _significance_stars(p):
    """Convert p-value to significance star string."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ===================================================================
# F1: IRBS Bias Demo
# ===================================================================

def plot_f1_irbs_bias_demo(
    bias_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F1: IRBS bias demo -- violin + strip-box (2x1 panel).

    bias_df columns: repeat, naive_tau, irbs_tau, true_tau
    """
    if bias_df is None or len(bias_df) == 0:
        print("Warning [F1]: bias_df is empty; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    true_tau = float(bias_df["true_tau"].iloc[0])
    naive_vals = bias_df["naive_tau"].values
    irbs_vals = bias_df["irbs_tau"].values

    # ---- Left panel: violin plots ----
    ax = axes[0]
    parts_naive = ax.violinplot(naive_vals, positions=[0], showmeans=False,
                                showmedians=False, showextrema=False)
    parts_irbs = ax.violinplot(irbs_vals, positions=[1], showmeans=False,
                               showmedians=False, showextrema=False)

    for pc in parts_naive["bodies"]:
        pc.set_facecolor(COLORS["RANDOM_RED"])
        pc.set_alpha(0.45)
        pc.set_edgecolor(COLORS["RANDOM_RED"])
        pc.set_linewidth(0.8)

    for pc in parts_irbs["bodies"]:
        pc.set_facecolor(COLORS["SIT_GREEN"])
        pc.set_alpha(0.45)
        pc.set_edgecolor(COLORS["SIT_GREEN"])
        pc.set_linewidth(0.8)

    # Overlay means + CI
    for pos, vals, color in [(0, naive_vals, COLORS["RANDOM_RED"]),
                             (1, irbs_vals, COLORS["SIT_GREEN"])]:
        m, lo, hi = _bootstrap_ci(vals)
        ax.plot(pos, m, "o", color=color, markersize=8, zorder=5)
        ax.plot([pos, pos], [lo, hi], "-", color=color, linewidth=2.5, zorder=4)

    ax.axhline(true_tau, color=COLORS["DARK_TEXT"], linestyle="--", linewidth=1.5,
               label=f"True $\\tau$ = {true_tau:.1f} $\\mu$s", zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive\n(A-then-B)", "IRBS\n(Interleaved)"])
    ax.set_ylabel("Estimated Treatment Effect ($\\Delta$p99, $\\mu$s)")
    ax.set_title("Distribution of Effect Estimates Under Drift")
    ax.legend(loc="upper right", frameon=True)

    # Significance annotation
    p_val = _mannwhitneyu_pvalue(naive_vals, irbs_vals)
    stars = _significance_stars(p_val)
    y_max = max(np.max(naive_vals), np.max(irbs_vals))
    y_bar = y_max + 0.05 * abs(y_max)
    ax.plot([0, 0, 1, 1], [y_bar * 0.97, y_bar, y_bar, y_bar * 0.97],
            lw=1.2, color=COLORS["DARK_TEXT"])
    ax.text(0.5, y_bar * 1.02, stars, ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=COLORS["DARK_TEXT"])

    # ---- Right panel: |bias| box + strip ----
    ax = axes[1]
    naive_abs_bias = np.abs(naive_vals - true_tau)
    irbs_abs_bias = np.abs(irbs_vals - true_tau)

    bp = ax.boxplot(
        [naive_abs_bias, irbs_abs_bias],
        positions=[0, 1], widths=0.45, patch_artist=True,
        showfliers=False, zorder=2,
        medianprops=dict(color=COLORS["DARK_TEXT"], linewidth=1.5),
        whiskerprops=dict(color=COLORS["DARK_TEXT"]),
        capprops=dict(color=COLORS["DARK_TEXT"]),
    )
    bp["boxes"][0].set_facecolor(COLORS["LIGHT_RED"])
    bp["boxes"][0].set_edgecolor(COLORS["RANDOM_RED"])
    bp["boxes"][1].set_facecolor(COLORS["LIGHT_GREEN"])
    bp["boxes"][1].set_edgecolor(COLORS["SIT_GREEN"])

    # Jittered strip overlay
    rng_jitter = np.random.default_rng(7)
    for pos, vals, color in [(0, naive_abs_bias, COLORS["RANDOM_RED"]),
                             (1, irbs_abs_bias, COLORS["SIT_GREEN"])]:
        jitter = rng_jitter.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, s=18, alpha=0.55, color=color,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive", "IRBS"])
    ax.set_ylabel("|Bias| ($\\mu$s)")
    ax.set_title("Absolute Bias Comparison")

    # Bias reduction annotation
    naive_mean_bias = float(np.mean(naive_abs_bias))
    irbs_mean_bias = float(np.mean(irbs_abs_bias))
    if naive_mean_bias > 0:
        reduction_pct = (naive_mean_bias - irbs_mean_bias) / naive_mean_bias * 100
    else:
        reduction_pct = 0.0
    ax.text(
        0.5, 0.95,
        f"Bias reduction: {reduction_pct:.0f}%",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["LIGHT_GREEN"],
                  edgecolor=COLORS["SIT_GREEN"], alpha=0.85),
    )

    fig.suptitle("F1: IRBS Drift-Bias Demonstration", fontsize=16, fontweight="bold", y=1.01)
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F1_irbs_bias_demo", output_dir)


# ===================================================================
# F2: Phenomenon -- 2x2 panel
# ===================================================================

def plot_f2_phenomenon(
    results_df: pd.DataFrame,
    target_name: str = "kv_lookup",
    spectator_name: str = "cache_thrash",
    output_dir: str = "results/figures",
):
    """F2: 2x2 phenomenon panel (load ladder p99 / CVaR99, distance ladder, ratio)."""
    if results_df is None or len(results_df) == 0:
        print("Warning [F2]: results_df is empty; skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Build subset -- some DataFrames may not have 'spectator' column
    if "spectator" in results_df.columns:
        # Use data for specific spectator if possible; fall back to all
        subset_spec = results_df[
            (results_df["target"] == target_name) &
            (results_df["spectator"] == spectator_name)
        ]
        subset = subset_spec if len(subset_spec) > 0 else results_df[results_df["target"] == target_name]
    else:
        subset = results_df[results_df["target"] == target_name].copy() if "target" in results_df.columns else results_df.copy()

    if len(subset) == 0:
        subset = results_df.copy()

    dist_order = ["same_core", "same_llc", "same_numa", "cross_numa", "cross_socket"]
    regimes = [r for r in ["benign", "structured", "adversarial"] if "regime" in subset.columns and r in subset["regime"].values]
    if not regimes:
        regimes = ["all"]

    def _plot_ladder(ax, subset_df, x_col, y_col, title, xlabel, use_log=True):
        """Helper for load/distance ladder with confidence bands."""
        for regime in regimes:
            if regime == "all":
                rdata = subset_df
            else:
                rdata = subset_df[subset_df["regime"] == regime]
            if len(rdata) == 0:
                continue

            color = REGIME_COLORS.get(regime, COLORS["NEUTRAL_GRAY"])
            if x_col == "distance":
                available = [d for d in dist_order if d in rdata[x_col].values]
                if not available:
                    continue
                rdata = rdata[rdata[x_col].isin(available)]
                agg = rdata.groupby(x_col)[y_col].agg(["mean", "std", "count"]).reindex(available)
                xs = np.arange(len(available))
                ax.set_xticks(xs)
                ax.set_xticklabels([d.replace("_", "\n") for d in available], fontsize=9)
            else:
                agg = rdata.groupby(x_col)[y_col].agg(["mean", "std", "count"]).sort_index()
                xs = agg.index.values.astype(float)

            means = agg["mean"].values.astype(float)
            stds = agg["std"].fillna(0).values.astype(float)
            counts = agg["count"].values.astype(float)
            se = stds / np.sqrt(np.maximum(counts, 1))

            label = regime.capitalize() if regime != "all" else y_col
            ax.plot(xs, means, "o-", color=color, linewidth=2, markersize=5, label=label, zorder=3)
            ax.fill_between(xs, means - 1.96 * se, means + 1.96 * se, alpha=0.18, color=color, zorder=2)

        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{y_col.upper()} Latency ($\\mu$s)")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, frameon=True)

    # Top-left: Load ladder p99
    if "load" in subset.columns and "p99" in subset.columns:
        _plot_ladder(axes[0, 0], subset, "load", "p99",
                     "Load Ladder: p99", "Load Level")
    else:
        axes[0, 0].text(0.5, 0.5, "No load / p99 data", transform=axes[0, 0].transAxes, ha="center")

    # Top-right: Load ladder CVaR99
    if "load" in subset.columns and "cvar99" in subset.columns:
        _plot_ladder(axes[0, 1], subset, "load", "cvar99",
                     "Load Ladder: CVaR99", "Load Level")
    else:
        axes[0, 1].text(0.5, 0.5, "No load / CVaR99 data", transform=axes[0, 1].transAxes, ha="center")

    # Bottom-left: Distance ladder p99
    if "distance" in subset.columns and "p99" in subset.columns:
        _plot_ladder(axes[1, 0], subset, "distance", "p99",
                     "Distance Ladder: p99", "Placement Distance")
    else:
        axes[1, 0].text(0.5, 0.5, "No distance / p99 data", transform=axes[1, 0].transAxes, ha="center")

    # Bottom-right: Ratio plot (treatment / control baseline)
    ax = axes[1, 1]
    if "load" in subset.columns and "p99" in subset.columns:
        # Use benign as "control" baseline; compute ratio for others
        benign_df = subset[subset["regime"] == "benign"] if "regime" in subset.columns else None
        if benign_df is not None and len(benign_df) > 0:
            benign_by_load = benign_df.groupby("load")["p99"].mean()
            for regime in ["structured", "adversarial"]:
                rdata = subset[subset["regime"] == regime] if "regime" in subset.columns else pd.DataFrame()
                if len(rdata) == 0:
                    continue
                regime_by_load = rdata.groupby("load")["p99"].mean()
                common_loads = sorted(set(benign_by_load.index) & set(regime_by_load.index))
                if not common_loads:
                    continue
                ratios = [regime_by_load[ld] / benign_by_load[ld] if benign_by_load[ld] > 0 else 1.0
                          for ld in common_loads]
                color = REGIME_COLORS.get(regime, COLORS["NEUTRAL_GRAY"])
                ax.plot(common_loads, ratios, "o-", color=color, linewidth=2, markersize=5,
                        label=f"{regime.capitalize()} / Benign")
            ax.axhline(1.0, color=COLORS["NEUTRAL_GRAY"], linestyle=":", linewidth=1, alpha=0.7)
            ax.set_xlabel("Load Level")
            ax.set_ylabel("Tail Inflation Factor (p99 ratio)")
            ax.set_title("Tail Inflation vs. Benign Baseline", fontsize=13)
            ax.legend(fontsize=9, frameon=True)
        else:
            ax.text(0.5, 0.5, "No benign baseline for ratio", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "Insufficient data for ratio plot", transform=ax.transAxes, ha="center")

    fig.suptitle(
        f"F2: Tail Explosion -- {target_name} + {spectator_name}",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F2_phenomenon_ladder", output_dir)


# ===================================================================
# F3: Tomography Heatmap
# ===================================================================

def plot_f3_tomography_heatmap(
    mean_matrix: pd.DataFrame,
    ci_lower: Optional[pd.DataFrame] = None,
    ci_upper: Optional[pd.DataFrame] = None,
    output_dir: str = "results/figures",
):
    """F3: Tomography heatmap with CI, sorted spectators, bold top interferer."""
    if mean_matrix is None or mean_matrix.empty:
        print("Warning [F3]: mean_matrix is empty; skipping.")
        return

    # Sort spectators (columns) by total interference (most impactful on left)
    col_totals = mean_matrix.astype(float).sum(axis=0).sort_values(ascending=False)
    sorted_cols = col_totals.index.tolist()
    mean_sorted = mean_matrix[sorted_cols].copy()

    ci_lo_sorted = ci_lower[sorted_cols].copy() if ci_lower is not None else None
    ci_hi_sorted = ci_upper[sorted_cols].copy() if ci_upper is not None else None

    data = mean_sorted.values.astype(float)
    n_rows, n_cols = data.shape

    fig_width = max(10, 1.6 * n_cols)
    fig_height = max(5, 1.2 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use YlOrRd with proper normalisation
    vmin = max(0, float(np.nanmin(data)))
    vmax = float(np.nanmax(data))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in mean_sorted.columns],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mean_sorted.index, fontsize=10)

    # Per-row max for bolding
    row_max_idx = np.nanargmax(data, axis=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            text_lines = f"{val:.0f}"
            if ci_lo_sorted is not None and ci_hi_sorted is not None:
                lo = float(ci_lo_sorted.values[i, j])
                hi = float(ci_hi_sorted.values[i, j])
                text_lines += f"\n[{lo:.0f}, {hi:.0f}]"

            # Color contrast
            normed = norm(val)
            text_color = "white" if normed > 0.55 else COLORS["DARK_TEXT"]
            weight = "bold" if j == row_max_idx[i] else "normal"
            fontsize = 8 if ci_lo_sorted is not None else 9

            ax.text(
                j, i, text_lines, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("$\\Delta$p99 ($\\mu$s)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Spectator Workload (sorted by total interference)", fontsize=13)
    ax.set_ylabel("Target Workload", fontsize=13)
    fig.suptitle(
        "F3: Interference Tomography Map with Uncertainty",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F3_tomography_heatmap", output_dir)


# ===================================================================
# F4: Sparse Recovery
# ===================================================================

def plot_f4_sparse_recovery(
    recovery_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F4: Sparse recovery curve with bands, threshold lines, annotations."""
    if recovery_df is None or len(recovery_df) == 0:
        print("Warning [F4]: recovery_df is empty; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    k_styles = {1: "-", 3: "--", 5: ":"}
    k_colors = {1: COLORS["BASELINE_BLUE"], 3: COLORS["SIT_GREEN"], 5: COLORS["ACCENT_PURPLE"]}

    for k_val in sorted(recovery_df["k"].unique()):
        kdata = recovery_df[recovery_df["k"] == k_val].sort_values("trials_per_spectator")
        xs = kdata["trials_per_spectator"].values
        ys = kdata["recovery_probability"].values

        color = k_colors.get(int(k_val), COLORS["NEUTRAL_GRAY"])
        style = k_styles.get(int(k_val), "-")

        ax.plot(
            xs, ys, style, color=color, linewidth=2.5, markersize=7,
            marker="o", label=f"Top-{int(k_val)} recovery", zorder=3,
        )

        # Confidence band (approximate Wilson / binomial)
        n_rep = 30  # assumed from experiment
        se = np.sqrt(ys * (1 - ys) / max(n_rep, 1))
        ax.fill_between(xs, np.clip(ys - 1.96 * se, 0, 1),
                         np.clip(ys + 1.96 * se, 0, 1),
                         alpha=0.15, color=color, zorder=2)

        # Annotate sample-complexity threshold (first crossing above 80%)
        cross_idx = np.where(ys >= 0.8)[0]
        if len(cross_idx) > 0:
            cx = xs[cross_idx[0]]
            cy = ys[cross_idx[0]]
            ax.annotate(
                f"$n_{{80\\%}}$ = {int(cx)}",
                xy=(cx, cy), xytext=(cx + 0.8, cy - 0.12),
                fontsize=9, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=5,
            )

    # Reference lines
    for thresh, lbl in [(0.5, "50%"), (0.8, "80%"), (0.95, "95%")]:
        ax.axhline(thresh, color=COLORS["NEUTRAL_GRAY"], linestyle=":", linewidth=1, alpha=0.6)
        ax.text(ax.get_xlim()[0] + 0.2, thresh + 0.015, lbl, fontsize=8,
                color=COLORS["NEUTRAL_GRAY"], va="bottom")

    ax.set_xlabel("Trials per Spectator")
    ax.set_ylabel("Recovery Probability")
    ax.set_ylim(-0.03, 1.07)
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("F4: Sparse Interferer Recovery vs. Sampling Budget",
                 fontsize=16, fontweight="bold")
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F4_sparse_recovery", output_dir)


# ===================================================================
# F5: Baseline Mismatch (2x1)
# ===================================================================

def plot_f5_baseline_mismatch(
    scatter_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F5: 2x1 -- scatter (left) + CDF of prediction error (right)."""
    if scatter_df is None or len(scatter_df) == 0:
        print("Warning [F5]: scatter_df is empty; skipping.")
        return

    observed = scatter_df["observed"].values.astype(float)
    predicted = scatter_df["predicted"].values.astype(float)
    error = predicted - observed  # negative => underprediction

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Left: scatter with density contours + underprediction region ----
    ax = axes[0]

    # Shade the underprediction region (above diagonal = predicted < observed)
    lim_lo = min(np.min(observed), np.min(predicted)) - 5
    lim_hi = max(np.max(observed), np.max(predicted)) + 5
    ax.fill_between(
        [lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
        color=COLORS["LIGHT_RED"], alpha=0.18, label="Underprediction zone", zorder=0,
    )

    under = predicted < observed
    ax.scatter(
        observed[~under], predicted[~under], s=18, alpha=0.4,
        color=COLORS["BASELINE_BLUE"], edgecolors="none", label="Adequate", zorder=2,
    )
    ax.scatter(
        observed[under], predicted[under], s=22, alpha=0.6,
        color=COLORS["RANDOM_RED"], marker="x", linewidths=0.8,
        label="Underprediction", zorder=3,
    )

    # Density contours (if enough points)
    if len(observed) > 30:
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([observed, predicted])
            kde = gaussian_kde(xy)
            xg = np.linspace(lim_lo, lim_hi, 80)
            yg = np.linspace(lim_lo, lim_hi, 80)
            Xg, Yg = np.meshgrid(xg, yg)
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contour(Xg, Yg, Z, levels=5, colors=COLORS["NEUTRAL_GRAY"],
                       linewidths=0.6, alpha=0.5, zorder=1)
        except Exception:
            pass  # scipy not available; skip contours

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--",
            color=COLORS["DARK_TEXT"], linewidth=1.5, alpha=0.7, label="Perfect prediction")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Observed $\\Delta$p99 ($\\mu$s)")
    ax.set_ylabel("Predicted $\\Delta$p99 ($\\mu$s)")
    ax.set_title("Predicted vs. Observed Interference", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="upper left", frameon=True)

    # R-squared and systematic bias
    ss_res = np.sum((predicted - observed) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sys_bias = float(np.mean(error))
    under_rate = float(np.mean(under) * 100)
    info_text = (f"$R^2$ = {r2:.3f}\n"
                 f"Systematic bias = {sys_bias:.1f} $\\mu$s\n"
                 f"Underprediction rate = {under_rate:.1f}%")
    ax.text(
        0.03, 0.97, info_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["GRID_GRAY"], alpha=0.9),
    )

    # ---- Right: CDF of prediction error ----
    ax = axes[1]
    sorted_err = np.sort(error)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

    ax.plot(sorted_err, cdf, "-", color=COLORS["BASELINE_BLUE"], linewidth=2, zorder=3)
    ax.axvline(0, color=COLORS["DARK_TEXT"], linestyle=":", linewidth=1, alpha=0.6)
    ax.fill_betweenx(
        cdf, sorted_err, 0,
        where=(sorted_err < 0), color=COLORS["LIGHT_RED"], alpha=0.25,
        label="Underprediction tail", zorder=2,
    )

    ax.set_xlabel("Prediction Error (Predicted $-$ Observed, $\\mu$s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Prediction Error", fontsize=13)
    ax.legend(fontsize=9, frameon=True)

    # Annotate median error
    median_err = float(np.median(error))
    ax.axvline(median_err, color=COLORS["WARN_ORANGE"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(median_err, 0.5, f"  Median = {median_err:.1f}",
            fontsize=9, color=COLORS["WARN_ORANGE"], va="center")

    fig.suptitle(
        "F5: Naive Model Mismatch -- Systematic Underprediction of Tail Risk",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F5_baseline_mismatch", output_dir)


# ===================================================================
# F6: Scheduler Comparison (1x3 panel by regime)
# ===================================================================

def plot_f6_scheduler_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F6: 1x3 grouped bar chart -- one panel per regime, hatching for CVaR99."""
    if summary_df is None or len(summary_df) == 0:
        print("Warning [F6]: summary_df is empty; skipping.")
        return

    regimes = [r for r in ["benign", "structured", "adversarial"]
               if "regime" in summary_df.columns and r in summary_df["regime"].values]
    if not regimes:
        regimes = summary_df["regime"].unique().tolist() if "regime" in summary_df.columns else ["all"]

    raw_scheds = summary_df["scheduler"].unique().tolist() if "scheduler" in summary_df.columns else []
    schedulers = _ordered_schedulers(raw_scheds)
    n_sched = len(schedulers)
    if n_sched == 0:
        print("Warning [F6]: no schedulers found; skipping.")
        return

    n_panels = len(regimes)
    fig, axes_arr = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6.5), sharey=True)
    if n_panels == 1:
        axes_arr = [axes_arr]

    # Determine random baseline for significance testing
    random_cache = {}  # regime -> array of p99 values

    for idx, regime in enumerate(regimes):
        ax = axes_arr[idx]
        regime_data = summary_df[summary_df["regime"] == regime] if regime != "all" else summary_df

        x = np.arange(n_sched)
        width = 0.36

        p99_vals = []
        cvar_vals = []
        for sched in schedulers:
            sdata = regime_data[regime_data["scheduler"] == sched] if "scheduler" in regime_data.columns else pd.DataFrame()
            p99_vals.append(float(sdata["p99"].mean()) if len(sdata) > 0 and "p99" in sdata.columns else 0)
            cvar_vals.append(float(sdata["cvar99"].mean()) if len(sdata) > 0 and "cvar99" in sdata.columns else 0)

        colors_p99 = [_sched_color(s) for s in schedulers]
        colors_cvar = [_sched_color(s) for s in schedulers]

        bars1 = ax.bar(x - width / 2, p99_vals, width, color=colors_p99, alpha=0.85,
                       edgecolor="white", linewidth=0.5, label="p99" if idx == 0 else "")
        bars2 = ax.bar(x + width / 2, cvar_vals, width, color=colors_cvar, alpha=0.50,
                       edgecolor=colors_cvar, linewidth=0.8, hatch="//",
                       label="CVaR99" if idx == 0 else "")

        # Significance stars vs random
        rand_data = regime_data[regime_data["scheduler"] == "random"] if "scheduler" in regime_data.columns else pd.DataFrame()
        if len(rand_data) > 0 and "p99" in rand_data.columns:
            rand_p99 = rand_data["p99"].values
            for si, sched in enumerate(schedulers):
                if sched == "random":
                    continue
                sdata = regime_data[regime_data["scheduler"] == sched] if "scheduler" in regime_data.columns else pd.DataFrame()
                if len(sdata) > 0 and "p99" in sdata.columns:
                    p = _mannwhitneyu_pvalue(sdata["p99"].values, rand_p99)
                    stars = _significance_stars(p)
                    if stars != "n.s.":
                        bar_top = max(p99_vals[si], cvar_vals[si])
                        ax.text(si, bar_top * 1.03, stars, ha="center", va="bottom",
                                fontsize=9, fontweight="bold", color=COLORS["DARK_TEXT"])

        ax.set_xticks(x)
        ax.set_xticklabels([_nice_sched_name(s) for s in schedulers],
                           rotation=40, ha="right", fontsize=9)
        ax.set_title(f"{regime.capitalize()} Regime", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Latency ($\\mu$s)")
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "F6: Scheduler Comparison Across Regimes",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F6_scheduler_comparison", output_dir)


# ===================================================================
# F7: Worst-Case (horizontal bar)
# ===================================================================

def plot_f7_worst_case(
    worst_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F7: Horizontal bar chart sorted by p99 descending, with SLO line."""
    if worst_df is None or worst_df.empty:
        print("Warning [F7]: worst_df is empty; skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify scheduler column and metric column
    if "scheduler" in worst_df.columns:
        p99_cols = [c for c in worst_df.columns if "p99_mean" in c]
        metric_col = p99_cols[0] if p99_cols else ("p99" if "p99" in worst_df.columns else worst_df.columns[1])

        df_sorted = worst_df.sort_values(metric_col, ascending=True).copy()
        sched_names = df_sorted["scheduler"].values
        vals = df_sorted[metric_col].values.astype(float)

        # Classify colours
        bar_colors = []
        for s in sched_names:
            s_lower = str(s).lower()
            if "sit" in s_lower:
                bar_colors.append(COLORS["SIT_GREEN"])
            elif "random" in s_lower:
                bar_colors.append(COLORS["RANDOM_RED"])
            else:
                bar_colors.append(COLORS["BASELINE_BLUE"])

        y_pos = np.arange(len(sched_names))
        ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, edgecolor="white", height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_nice_sched_name(s) for s in sched_names], fontsize=10)
        ax.set_xlabel("p99 Latency ($\\mu$s) on Hardest Conditions")

        # SLO threshold line (use 2x median as proxy if we lack actual SLO)
        slo_proxy = float(np.median(vals) * 1.5) if len(vals) > 2 else None
        if slo_proxy is not None:
            ax.axvline(slo_proxy, color=COLORS["WARN_ORANGE"], linestyle="--",
                       linewidth=1.8, label=f"SLO Threshold ({slo_proxy:.0f} $\\mu$s)", zorder=4)
            ax.legend(fontsize=9, loc="lower right", frameon=True)

        # Annotate SIT vs random reduction
        random_val = None
        sit_val = None
        for s, v in zip(sched_names, vals):
            s_lower = str(s).lower()
            if "random" in s_lower:
                random_val = v
            if s_lower == "sit_dpp":
                sit_val = v
        if random_val is not None and sit_val is not None and random_val > 0:
            reduction = (random_val - sit_val) / random_val * 100
            ax.text(
                0.97, 0.05,
                f"SIT-DPP: {reduction:.0f}% reduction\nvs. Random",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=COLORS["SIT_GREEN"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["LIGHT_GREEN"],
                          edgecolor=COLORS["SIT_GREEN"], alpha=0.85),
            )
    else:
        ax.text(0.5, 0.5, str(worst_df.to_string()), transform=ax.transAxes,
                fontsize=8, va="center", ha="center", family="monospace")
        ax.axis("off")

    ax.set_title(
        "F7: Worst-Case Blowup Avoidance (Top 10% Conditions)",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F7_worst_case", output_dir)


# ===================================================================
# F8: Ablation Panel (2x2 with waterfall)
# ===================================================================

def plot_f8_ablations(
    ablation_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F8: 2x2 ablation panel -- p99 bars, CVaR99 bars, relative improvement, waterfall."""
    if ablation_df is None or len(ablation_df) == 0:
        print("Warning [F8]: ablation_df is empty; skipping.")
        return
    if "variant" not in ablation_df.columns:
        print("Warning [F8]: 'variant' column missing; skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    variants = ablation_df["variant"].values
    n_var = len(variants)
    x = np.arange(n_var)

    def _variant_color(v):
        v = str(v).lower()
        if "sit_dpp" == v:
            return COLORS["SIT_GREEN"]
        if "sit_ucb" in v:
            return "#40916c"
        if "random" in v:
            return COLORS["RANDOM_RED"]
        return COLORS["BASELINE_BLUE"]

    colors = [_variant_color(v) for v in variants]
    nice_names = [_nice_sched_name(v) for v in variants]

    # Fetch random baseline for relative improvement
    random_row = ablation_df[ablation_df["variant"].str.contains("random", case=False, na=False)]
    rand_p99 = float(random_row["p99"].iloc[0]) if len(random_row) > 0 and "p99" in random_row.columns else None
    rand_cvar = float(random_row["cvar99"].iloc[0]) if len(random_row) > 0 and "cvar99" in random_row.columns else None

    # ---- Top-left: p99 bars ----
    ax = axes[0, 0]
    if "p99" in ablation_df.columns:
        vals = ablation_df["p99"].values.astype(float)
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(nice_names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("p99 Latency ($\\mu$s)")
        ax.set_title("p99 Across Ablation Variants", fontsize=13)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01 * max(vals), f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # ---- Top-right: CVaR99 bars ----
    ax = axes[0, 1]
    if "cvar99" in ablation_df.columns:
        vals = ablation_df["cvar99"].values.astype(float)
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(nice_names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("CVaR99 Latency ($\\mu$s)")
        ax.set_title("CVaR99 Across Ablation Variants", fontsize=13)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01 * max(vals), f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # ---- Bottom-left: Relative improvement over random (%) ----
    ax = axes[1, 0]
    if rand_p99 is not None and "p99" in ablation_df.columns:
        improvements = [(rand_p99 - float(row["p99"])) / rand_p99 * 100
                        if rand_p99 > 0 else 0.0
                        for _, row in ablation_df.iterrows()]
        bar_colors_imp = [COLORS["SIT_GREEN"] if imp > 0 else COLORS["RANDOM_RED"] for imp in improvements]
        ax.bar(x, improvements, color=bar_colors_imp, alpha=0.85, edgecolor="white", width=0.65)
        ax.axhline(0, color=COLORS["DARK_TEXT"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(nice_names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Relative Improvement vs. Random (%)")
        ax.set_title("p99 Reduction Relative to Random", fontsize=13)
        for i, imp in enumerate(improvements):
            ax.text(i, imp + (1 if imp >= 0 else -3), f"{imp:.1f}%",
                    ha="center", va="bottom" if imp >= 0 else "top", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No random baseline for comparison",
                transform=ax.transAxes, ha="center", va="center")

    # ---- Bottom-right: Component contribution waterfall ----
    ax = axes[1, 1]
    # Build waterfall: start from random baseline, show incremental gains
    # Components: DPP diversity, risk-aware scoring, UCB exploration
    if rand_p99 is not None and "p99" in ablation_df.columns:
        # Identify key variants
        variant_map = {str(row["variant"]).lower(): float(row["p99"]) for _, row in ablation_df.iterrows()}
        random_p99 = rand_p99

        # Decompose contributions
        waterfall_items = [("Random\nBaseline", random_p99, COLORS["RANDOM_RED"])]

        # Diversity-only contribution
        div_only = variant_map.get("no_risk_diversity_only", None)
        if div_only is not None:
            delta = random_p99 - div_only
            waterfall_items.append(("+ DPP\nDiversity", -delta, COLORS["ACCENT_PURPLE"]))

        # Risk-only contribution (from diversity-only to full sit_dpp)
        risk_only = variant_map.get("no_dpp_risk_only", None)
        if risk_only is not None:
            delta = random_p99 - risk_only
            waterfall_items.append(("+ Risk\nScoring", -delta, COLORS["WARN_ORANGE"]))

        # UCB contribution
        sit_dpp_val = variant_map.get("sit_dpp", None)
        sit_ucb_val = variant_map.get("sit_ucb_dpp", None)
        if sit_dpp_val is not None and sit_ucb_val is not None:
            delta = sit_dpp_val - sit_ucb_val
            waterfall_items.append(("+ UCB\nExploration", -delta, COLORS["BASELINE_BLUE"]))

        # Final SIT-DPP result
        if sit_dpp_val is not None:
            waterfall_items.append(("SIT-DPP\nFinal", sit_dpp_val, COLORS["SIT_GREEN"]))

        # Draw the waterfall
        if len(waterfall_items) > 1:
            labels = [item[0] for item in waterfall_items]
            wf_x = np.arange(len(labels))
            cumulative = random_p99
            bottoms = []
            heights = []
            wf_colors = []

            for i, (lbl, val, col) in enumerate(waterfall_items):
                if i == 0:
                    bottoms.append(0)
                    heights.append(val)
                    wf_colors.append(col)
                elif i == len(waterfall_items) - 1:
                    bottoms.append(0)
                    heights.append(val)
                    wf_colors.append(col)
                else:
                    # val is negative delta (reduction)
                    bottoms.append(cumulative + val)
                    heights.append(abs(val))
                    wf_colors.append(col)
                    cumulative += val

            ax.bar(wf_x, heights, bottom=bottoms, color=wf_colors, alpha=0.85,
                   edgecolor="white", width=0.55)

            # Connector lines between bars
            for i in range(len(waterfall_items) - 1):
                top_i = bottoms[i] + heights[i]
                ax.plot([i + 0.3, i + 0.7], [top_i, top_i],
                        color=COLORS["NEUTRAL_GRAY"], linewidth=0.8, linestyle=":")

            ax.set_xticks(wf_x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("p99 Latency ($\\mu$s)")
            ax.set_title("Component Contribution Waterfall", fontsize=13)
        else:
            ax.text(0.5, 0.5, "Insufficient variants for waterfall",
                    transform=ax.transAxes, ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No data for waterfall", transform=ax.transAxes,
                ha="center", va="center")

    fig.suptitle(
        "F8: Ablation Study -- Component Contributions",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F8_ablations", output_dir)


# ===================================================================
# F9: QA Summary Table
# ===================================================================

def plot_f9_qa_summary(
    qa_results: Dict[str, Dict],
    output_dir: str = "results/figures",
):
    """F9: Clean QA table with colour-coded cells and summary stats."""
    if qa_results is None or len(qa_results) == 0:
        print("Warning [F9]: qa_results is empty; skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(qa_results) + 2)))
    ax.axis("off")

    cell_text = []
    cell_colors = []
    n_pass = 0
    n_total = 0

    for test_name, result in qa_results.items():
        passed = result.get("passed", "N/A")
        message = result.get("message", "")
        status = "PASS" if passed else "FAIL"
        n_total += 1
        if passed:
            n_pass += 1
        cell_text.append([test_name.replace("_", " ").title(), status, message[:90]])
        row_color = "#d4edda" if passed else "#f8d7da"
        cell_colors.append([row_color, row_color, row_color])

    # Summary row
    pass_rate = n_pass / n_total * 100 if n_total > 0 else 0
    cell_text.append(["TOTAL", f"{n_pass}/{n_total}", f"Pass rate: {pass_rate:.0f}%"])
    summary_color = "#d4edda" if n_pass == n_total else "#fff3cd"
    cell_colors.append([summary_color, summary_color, summary_color])

    if cell_text:
        table = ax.table(
            cellText=cell_text,
            colLabels=["Test", "Status", "Details"],
            cellLoc="left",
            loc="center",
            colWidths=[0.30, 0.10, 0.60],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        # Style header
        for j in range(3):
            cell = table[0, j]
            cell.set_facecolor(COLORS["BASELINE_BLUE"])
            cell.set_text_props(color="white", fontweight="bold")

        # Style data rows
        for i in range(len(cell_text)):
            for j in range(3):
                cell = table[i + 1, j]
                cell.set_facecolor(cell_colors[i][j])
                cell.set_edgecolor(COLORS["GRID_GRAY"])
                if i == len(cell_text) - 1:
                    cell.set_text_props(fontweight="bold")

    ax.set_title(
        "F9: QA Check Summary",
        fontsize=16, fontweight="bold", pad=20,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F9_qa_summary", output_dir)


# ===================================================================
# F10: Channel Decomposition (NEW)
# ===================================================================

def plot_f10_channel_decomposition(
    channel_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F10: Stacked bar chart of per-channel interference contributions.

    channel_df expected columns: target, spectator, and one column per
    channel name (LLC, MEM_BW, ...) or a 'channel' + 'value' long-format.
    If the DataFrame is missing, this can also be built from the tomography
    mean matrix + workload definitions.
    """
    if channel_df is None or len(channel_df) == 0:
        print("Warning [F10]: channel_df is empty; skipping.")
        return

    # Support both wide and long formats
    has_channel_cols = all(ch in channel_df.columns for ch in CHANNEL_NAMES)

    if not has_channel_cols:
        # Try long format: columns = [target, spectator, channel, value/per_channel_severity]
        value_col = None
        for candidate in ("value", "per_channel_severity", "channel_overlap"):
            if candidate in channel_df.columns:
                value_col = candidate
                break
        if "channel" in channel_df.columns and value_col is not None:
            pivot = channel_df.pivot_table(
                index=["target", "spectator"], columns="channel", values=value_col, aggfunc="mean",
            ).fillna(0)
            wide_df = pivot.reset_index()
        else:
            print("Warning [F10]: channel_df has unexpected format; skipping.")
            return
    else:
        wide_df = channel_df.copy()

    # Build pair label
    if "target" in wide_df.columns and "spectator" in wide_df.columns:
        wide_df["pair"] = wide_df["target"].astype(str) + "\n+ " + wide_df["spectator"].astype(str)
    else:
        wide_df["pair"] = [f"Pair {i}" for i in range(len(wide_df))]

    # Sort by total interference (descending)
    avail_channels = [ch for ch in CHANNEL_NAMES if ch in wide_df.columns]
    if not avail_channels:
        print("Warning [F10]: no channel columns found; skipping.")
        return

    wide_df["_total"] = wide_df[avail_channels].sum(axis=1)
    wide_df = wide_df.sort_values("_total", ascending=False).head(20)  # top-20 pairs

    n_pairs = len(wide_df)
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_pairs), 7))

    x = np.arange(n_pairs)
    bottoms = np.zeros(n_pairs)

    for ci, ch in enumerate(avail_channels):
        vals = wide_df[ch].values.astype(float)
        color = CHANNEL_COLORS[ci % len(CHANNEL_COLORS)]
        ax.bar(x, vals, bottom=bottoms, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4, width=0.7, label=ch)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(wide_df["pair"].values, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Per-Channel Interference Contribution ($\\mu$s)")
    ax.set_xlabel("Target + Spectator Pair")
    ax.legend(title="Channel", fontsize=8, title_fontsize=9, loc="upper right",
              frameon=True, ncol=2)

    ax.set_title(
        "F10: Channel Decomposition of Interference",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F10_channel_decomposition", output_dir)


# ===================================================================
# F11: Sensitivity Analysis (NEW)
# ===================================================================

def plot_f11_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_dir: str = "results/figures",
):
    """F11: 2x2 sensitivity panel -- SIT improvement vs. load, distance, regime, n_cotenants.

    sensitivity_df expected columns include at least some of:
      scheduler, regime, load, distance, n_cotenants, p99
    We compute SIT improvement relative to random per group.
    """
    if sensitivity_df is None or len(sensitivity_df) == 0:
        print("Warning [F11]: sensitivity_df is empty; skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def _improvement_by_group(df, group_col, ax, xlabel, sort_order=None):
        """Plot SIT-DPP improvement % vs a grouping variable."""
        if group_col not in df.columns or "scheduler" not in df.columns or "p99" not in df.columns:
            ax.text(0.5, 0.5, f"No '{group_col}' data", transform=ax.transAxes, ha="center")
            return

        sit_df = df[df["scheduler"] == "sit_dpp"]
        rand_df = df[df["scheduler"] == "random"]
        if len(sit_df) == 0 or len(rand_df) == 0:
            ax.text(0.5, 0.5, "Need sit_dpp and random data", transform=ax.transAxes, ha="center")
            return

        sit_group = sit_df.groupby(group_col)["p99"].agg(["mean", "std", "count"])
        rand_group = rand_df.groupby(group_col)["p99"].agg(["mean", "std", "count"])

        common_keys = sorted(set(sit_group.index) & set(rand_group.index),
                             key=lambda k: (sort_order.index(k) if sort_order and k in sort_order else 0))

        if not common_keys:
            ax.text(0.5, 0.5, "No overlapping groups", transform=ax.transAxes, ha="center")
            return

        improvements = []
        ci_lo_list = []
        ci_hi_list = []
        for key in common_keys:
            rm = rand_group.loc[key, "mean"]
            sm = sit_group.loc[key, "mean"]
            imp = (rm - sm) / rm * 100 if rm > 0 else 0
            improvements.append(imp)
            # Approximate CI from SE propagation
            r_se = rand_group.loc[key, "std"] / np.sqrt(max(rand_group.loc[key, "count"], 1))
            s_se = sit_group.loc[key, "std"] / np.sqrt(max(sit_group.loc[key, "count"], 1))
            combined_se = np.sqrt(r_se ** 2 + s_se ** 2) / rm * 100 if rm > 0 else 0
            ci_lo_list.append(imp - 1.96 * combined_se)
            ci_hi_list.append(imp + 1.96 * combined_se)

        if isinstance(common_keys[0], (int, float, np.integer, np.floating)):
            xs = np.array([float(k) for k in common_keys])
            ax.plot(xs, improvements, "o-", color=COLORS["SIT_GREEN"], linewidth=2.5,
                    markersize=7, zorder=3)
            ax.fill_between(xs, ci_lo_list, ci_hi_list, alpha=0.18,
                            color=COLORS["SIT_GREEN"], zorder=2)
        else:
            xs = np.arange(len(common_keys))
            ax.bar(xs, improvements, color=COLORS["SIT_GREEN"], alpha=0.85,
                   edgecolor="white", width=0.55)
            ax.errorbar(xs, improvements,
                        yerr=[np.array(improvements) - np.array(ci_lo_list),
                              np.array(ci_hi_list) - np.array(improvements)],
                        fmt="none", ecolor=COLORS["DARK_TEXT"], capsize=4, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(k).replace("_", "\n") for k in common_keys],
                               fontsize=9, rotation=30, ha="right")

        ax.axhline(0, color=COLORS["DARK_TEXT"], linewidth=0.8, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SIT-DPP Improvement (%)")

    dist_order = ["same_core", "same_llc", "same_numa", "cross_numa", "cross_socket"]

    # Top-left: vs load
    _improvement_by_group(sensitivity_df, "load", axes[0, 0], "Load Level")
    axes[0, 0].set_title("Improvement vs. Load", fontsize=13)

    # Top-right: vs distance
    _improvement_by_group(sensitivity_df, "distance", axes[0, 1], "Placement Distance",
                          sort_order=dist_order)
    axes[0, 1].set_title("Improvement vs. Distance", fontsize=13)

    # Bottom-left: vs regime
    _improvement_by_group(sensitivity_df, "regime", axes[1, 0], "Interference Regime",
                          sort_order=["benign", "structured", "adversarial"])
    axes[1, 0].set_title("Improvement vs. Regime", fontsize=13)

    # Bottom-right: vs n_cotenants
    if "n_cotenants" in sensitivity_df.columns:
        _improvement_by_group(sensitivity_df, "n_cotenants", axes[1, 1], "Number of Co-Tenants")
        axes[1, 1].set_title("Improvement vs. Co-Tenant Count", fontsize=13)
    else:
        axes[1, 1].text(0.5, 0.5, "No co-tenant count data", transform=axes[1, 1].transAxes,
                        ha="center", va="center", fontsize=11, color=COLORS["NEUTRAL_GRAY"])
        axes[1, 1].set_title("Improvement vs. Co-Tenant Count", fontsize=13)

    fig.suptitle(
        "F11: Sensitivity Analysis -- SIT-DPP Improvement Factors",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F11_sensitivity_analysis", output_dir)


# ===================================================================
# F12: Real-system anchoring experiment
# ===================================================================

def plot_f12_anchoring(anchoring_summary, output_dir: str = "results/figures"):
    """F12: Real-system anchoring -- calibrated simulation results.

    Grouped bar chart comparing Random vs SIT-DPP p99 for each
    calibrated real-system scenario (Triton, Redis, gRPC), with
    error bars and reduction annotations.
    """
    if isinstance(anchoring_summary, pd.DataFrame) and len(anchoring_summary) == 0:
        return

    df = anchoring_summary
    scenarios = df["scenario"].tolist()
    n = len(scenarios)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: p99 comparison ---
    ax = axes[0]
    x = np.arange(n)
    width = 0.32

    random_p99 = df["random_p99"].values
    sit_p99 = df["sit_p99"].values
    random_err_lo = random_p99 - df["random_p99_ci_lo"].values
    random_err_hi = df["random_p99_ci_hi"].values - random_p99
    sit_err_lo = sit_p99 - df["sit_p99_ci_lo"].values
    sit_err_hi = df["sit_p99_ci_hi"].values - sit_p99

    bars_r = ax.bar(x - width/2, random_p99, width, label="Random Placement",
                    color=COLORS["RANDOM_RED"], alpha=0.85, edgecolor="white",
                    yerr=[random_err_lo, random_err_hi], capsize=4)
    bars_s = ax.bar(x + width/2, sit_p99, width, label="SIT-DPP",
                    color=COLORS["SIT_GREEN"], alpha=0.85, edgecolor="white",
                    yerr=[sit_err_lo, sit_err_hi], capsize=4)

    # Annotate reductions
    for i in range(n):
        red_pct = df["p99_reduction_pct"].values[i]
        y_max = max(random_p99[i], sit_p99[i]) * 1.08
        ax.annotate(
            f"-{red_pct:.0f}%", xy=(x[i], y_max),
            fontsize=11, fontweight="bold", ha="center", va="bottom",
            color=COLORS["SIT_GREEN"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=10)
    ax.set_ylabel("p99 Latency (us)", fontsize=12)
    ax.set_title("A. p99 Latency: Random vs SIT-DPP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # --- Panel B: CVaR99 comparison ---
    ax2 = axes[1]
    random_cvar = df["random_cvar99"].values
    sit_cvar = df["sit_cvar99"].values

    bars_r2 = ax2.bar(x - width/2, random_cvar, width, label="Random Placement",
                      color=COLORS["RANDOM_RED"], alpha=0.85, edgecolor="white")
    bars_s2 = ax2.bar(x + width/2, sit_cvar, width, label="SIT-DPP",
                      color=COLORS["SIT_GREEN"], alpha=0.85, edgecolor="white")

    for i in range(n):
        red_pct = df["cvar99_reduction_pct"].values[i]
        y_max = max(random_cvar[i], sit_cvar[i]) * 1.08
        ax2.annotate(
            f"-{red_pct:.0f}%", xy=(x[i], y_max),
            fontsize=11, fontweight="bold", ha="center", va="bottom",
            color=COLORS["SIT_GREEN"],
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=10)
    ax2.set_ylabel("CVaR99 Latency (us)", fontsize=12)
    ax2.set_title("B. CVaR99 Latency: Random vs SIT-DPP", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "F12: Real-System Anchoring -- Calibrated Simulation Results",
        fontsize=16, fontweight="bold", y=1.02,
    )

    # Add calibration note
    fig.text(
        0.5, -0.02,
        "Simulator parameters calibrated to published Triton, Redis, and gRPC latency benchmarks.\n"
        "See Section 5.10 for calibration methodology and references.",
        fontsize=8, ha="center", va="top", color=COLORS["NEUTRAL_GRAY"],
        style="italic",
    )

    _add_watermark(fig, "[Calibrated Simulation]")
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F12_anchoring_experiment", output_dir)


# ===================================================================
# F13: Pareto Frontier (Tail safety vs utilization)
# ===================================================================

def plot_f13_pareto_frontier(pareto_summary, output_dir="results/figures"):
    """F13: Pareto frontier -- scatter of tail safety vs utilization.

    pareto_summary columns: scheduler, mean_p99, mean_utilization, is_pareto_optimal
    """
    if pareto_summary is None or len(pareto_summary) == 0:
        print("Warning [F13]: pareto_summary is empty; skipping.")
        return

    df = pareto_summary.copy()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot non-Pareto-optimal points first
    non_optimal = df[~df["is_pareto_optimal"]]
    optimal = df[df["is_pareto_optimal"]]

    schedulers_all = df["scheduler"].unique().tolist()
    # Assign colors per scheduler
    sched_color_map = {}
    fallback_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
                       "#7209b7", "#c1121f", "#1d3557", "#6c757d", "#40916c"]
    for i, s in enumerate(schedulers_all):
        sched_color_map[s] = SCHEDULER_COLORS.get(s, fallback_colors[i % len(fallback_colors)])

    # Scatter: non-optimal as circles
    for _, row in non_optimal.iterrows():
        color = sched_color_map.get(row["scheduler"], COLORS["NEUTRAL_GRAY"])
        ax.scatter(
            row["mean_utilization"], row["mean_p99"],
            s=100, color=color, alpha=0.7, edgecolors="white",
            linewidths=0.8, zorder=3,
        )
        ax.annotate(
            _nice_sched_name(row["scheduler"]),
            xy=(row["mean_utilization"], row["mean_p99"]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=8, color=color, alpha=0.85,
        )

    # Scatter: Pareto-optimal as stars
    for _, row in optimal.iterrows():
        color = sched_color_map.get(row["scheduler"], COLORS["SIT_GREEN"])
        ax.scatter(
            row["mean_utilization"], row["mean_p99"],
            s=220, color=color, marker="*", edgecolors=COLORS["DARK_TEXT"],
            linewidths=0.6, zorder=5, label=None,
        )
        ax.annotate(
            _nice_sched_name(row["scheduler"]),
            xy=(row["mean_utilization"], row["mean_p99"]),
            xytext=(8, -10), textcoords="offset points",
            fontsize=9, fontweight="bold", color=color,
        )

    # Draw Pareto frontier line connecting optimal points (sorted by utilization)
    if len(optimal) > 1:
        opt_sorted = optimal.sort_values("mean_utilization")
        ax.plot(
            opt_sorted["mean_utilization"].values,
            opt_sorted["mean_p99"].values,
            "--", color=COLORS["DARK_TEXT"], linewidth=1.5, alpha=0.5,
            zorder=4, label="Pareto Frontier",
        )

    # Legend entries for marker types
    ax.scatter([], [], s=100, color=COLORS["NEUTRAL_GRAY"], label="Non-optimal")
    ax.scatter([], [], s=220, color=COLORS["NEUTRAL_GRAY"], marker="*",
               edgecolors=COLORS["DARK_TEXT"], label="Pareto-optimal")

    ax.set_xlabel("Mean Utilization")
    ax.set_ylabel("Mean p99 Latency ($\\mu$s)  [lower is better]")
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.set_title(
        "F13: Pareto Frontier -- Tail Safety vs Utilization",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F13_pareto_frontier", output_dir)


# ===================================================================
# F14: Probe Budget Curve (Error vs probes)
# ===================================================================

def plot_f14_probe_budget(probe_summary, output_dir="results/figures"):
    """F14: Probe budget -- reconstruction quality vs measurement cost.

    probe_summary columns: n_probes, strategy, mean_ndcg_k3, ci_lo, ci_hi
    """
    if probe_summary is None or len(probe_summary) == 0:
        print("Warning [F14]: probe_summary is empty; skipping.")
        return

    df = probe_summary.copy()
    fig, ax = plt.subplots(figsize=(10, 6.5))

    strategy_colors = {
        "random": COLORS["RANDOM_RED"],
        "round_robin": COLORS["BASELINE_BLUE"],
        "ucb": COLORS["WARN_ORANGE"],
        "dpp": COLORS["SIT_GREEN"],
    }
    strategy_styles = {
        "random": ("--", "o"),
        "round_robin": ("-.", "s"),
        "ucb": (":", "D"),
        "dpp": ("-", "^"),
    }

    strategies = df["strategy"].unique().tolist()
    # Sort so dpp is last (drawn on top)
    preferred_order = ["random", "round_robin", "ucb", "dpp"]
    strategies = [s for s in preferred_order if s in strategies] + \
                 [s for s in strategies if s not in preferred_order]

    for strat in strategies:
        sdata = df[df["strategy"] == strat].sort_values("n_probes")
        xs = sdata["n_probes"].values
        ys = sdata["mean_ndcg_k3"].values
        lo = sdata["ci_lo"].values
        hi = sdata["ci_hi"].values

        color = strategy_colors.get(strat, COLORS["NEUTRAL_GRAY"])
        ls, marker = strategy_styles.get(strat, ("-", "o"))

        ax.plot(
            xs, ys, linestyle=ls, marker=marker, color=color,
            linewidth=2.5, markersize=6,
            label=strat.replace("_", " ").title(), zorder=3,
        )
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color, zorder=2)

    # Reference line at NDCG=0.9
    ax.axhline(0.9, color=COLORS["NEUTRAL_GRAY"], linestyle=":", linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[0] + 0.5, 0.905, "NDCG@3 = 0.9", fontsize=8,
            color=COLORS["NEUTRAL_GRAY"])

    ax.set_xlabel("Number of Probes")
    ax.set_ylabel("NDCG@3  [higher is better]")
    ax.set_ylim(-0.03, 1.07)
    ax.legend(fontsize=10, frameon=True, loc="lower right")
    ax.set_title(
        "F14: Probe Budget -- Reconstruction Quality vs Measurement Cost",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F14_probe_budget", output_dir)


# ===================================================================
# F15: Drift Robustness (Bias under drift)
# ===================================================================

def plot_f15_drift_robustness(drift_summary, output_dir="results/figures"):
    """F15: Drift robustness -- IRBS vs naive estimator under drift.

    drift_summary columns: drift_type, magnitude, mean_naive_bias, mean_irbs_bias
    """
    if drift_summary is None or len(drift_summary) == 0:
        print("Warning [F15]: drift_summary is empty; skipping.")
        return

    df = drift_summary.copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))

    # ---- Panel A: Line plot by magnitude, averaged across drift types ----
    ax = axes[0]
    agg = df.groupby("magnitude").agg(
        naive_bias=("mean_naive_bias", lambda x: np.mean(np.abs(x))),
        irbs_bias=("mean_irbs_bias", lambda x: np.mean(np.abs(x))),
    ).sort_index()

    xs = agg.index.values.astype(float)
    ax.plot(xs, agg["naive_bias"].values, "o-", color=COLORS["RANDOM_RED"],
            linewidth=2.5, markersize=7, label="Naive (A-then-B)", zorder=3)
    ax.plot(xs, agg["irbs_bias"].values, "s-", color=COLORS["SIT_GREEN"],
            linewidth=2.5, markersize=7, label="IRBS (Interleaved)", zorder=3)

    ax.fill_between(xs, 0, agg["naive_bias"].values, alpha=0.10,
                    color=COLORS["RANDOM_RED"])
    ax.fill_between(xs, 0, agg["irbs_bias"].values, alpha=0.10,
                    color=COLORS["SIT_GREEN"])

    ax.set_xlabel("Drift Magnitude")
    ax.set_ylabel("|Bias| ($\\mu$s)")
    ax.set_title("A. Mean |Bias| vs. Drift Magnitude (averaged across types)", fontsize=13)
    ax.legend(fontsize=10, frameon=True)
    ax.axhline(0, color=COLORS["DARK_TEXT"], linewidth=0.6, linestyle=":")

    # ---- Panel B: Grouped bar chart by drift_type at highest magnitude ----
    ax = axes[1]
    max_mag = df["magnitude"].max()
    high_mag = df[df["magnitude"] == max_mag].copy()

    if len(high_mag) == 0:
        ax.text(0.5, 0.5, "No data at highest magnitude",
                transform=ax.transAxes, ha="center")
    else:
        drift_types = high_mag["drift_type"].unique().tolist()
        n_types = len(drift_types)
        x = np.arange(n_types)
        width = 0.35

        naive_vals = [float(np.abs(high_mag[high_mag["drift_type"] == dt]["mean_naive_bias"].mean()))
                      for dt in drift_types]
        irbs_vals = [float(np.abs(high_mag[high_mag["drift_type"] == dt]["mean_irbs_bias"].mean()))
                     for dt in drift_types]

        ax.bar(x - width / 2, naive_vals, width, color=COLORS["RANDOM_RED"],
               alpha=0.85, edgecolor="white", label="Naive")
        ax.bar(x + width / 2, irbs_vals, width, color=COLORS["SIT_GREEN"],
               alpha=0.85, edgecolor="white", label="IRBS")

        ax.set_xticks(x)
        ax.set_xticklabels([dt.replace("_", " ").title() for dt in drift_types],
                           fontsize=10, rotation=20, ha="right")
        ax.set_ylabel("|Bias| ($\\mu$s)")
        ax.set_title(f"B. |Bias| by Drift Type at Magnitude = {max_mag}", fontsize=13)
        ax.legend(fontsize=10, frameon=True)

        # Annotate reduction per drift type
        for i in range(n_types):
            if naive_vals[i] > 0:
                red = (naive_vals[i] - irbs_vals[i]) / naive_vals[i] * 100
                y_top = max(naive_vals[i], irbs_vals[i]) * 1.05
                ax.text(x[i], y_top, f"-{red:.0f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=COLORS["SIT_GREEN"])

    fig.suptitle(
        "F15: Drift Robustness -- IRBS vs Naive Estimator",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F15_drift_robustness", output_dir)


# ===================================================================
# F16: Calibration Plot (Simulator vs published)
# ===================================================================

def plot_f16_calibration(anchoring_summary, output_dir="results/figures"):
    """F16: Simulator calibration against published benchmarks.

    anchoring_summary columns: scenario, baseline_p99_us, random_p99, sit_p99
    """
    if anchoring_summary is None or len(anchoring_summary) == 0:
        print("Warning [F16]: anchoring_summary is empty; skipping.")
        return

    df = anchoring_summary.copy()
    fig, ax = plt.subplots(figsize=(11, 7))

    scenarios = df["scenario"].tolist()
    n = len(scenarios)
    x = np.arange(n)
    width = 0.28

    # Published baseline p99
    published_vals = df["baseline_p99_us"].values.astype(float)
    # Simulator baseline (random placement p99 as proxy)
    sim_vals = df["random_p99"].values.astype(float)
    # SIT p99
    sit_vals = df["sit_p99"].values.astype(float)

    ax.bar(x - width, published_vals, width, color=COLORS["BASELINE_BLUE"],
           alpha=0.85, edgecolor="white", label="Published Baseline p99")
    ax.bar(x, sim_vals, width, color=COLORS["WARN_ORANGE"],
           alpha=0.85, edgecolor="white", label="Simulator Random p99")
    ax.bar(x + width, sit_vals, width, color=COLORS["SIT_GREEN"],
           alpha=0.85, edgecolor="white", label="Simulator SIT-DPP p99")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=10)
    ax.set_ylabel("p99 Latency ($\\mu$s) [log scale]")
    ax.legend(fontsize=10, frameon=True, loc="upper right")

    # Annotate ratio match
    for i in range(n):
        if published_vals[i] > 0:
            ratio = sim_vals[i] / published_vals[i]
            ax.text(x[i], max(published_vals[i], sim_vals[i]) * 1.3,
                    f"Ratio: {ratio:.2f}x", ha="center", va="bottom",
                    fontsize=8, color=COLORS["DARK_TEXT"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=COLORS["GRID_GRAY"], alpha=0.8))

    ax.set_title(
        "F16: Simulator Calibration Against Published Benchmarks",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig, "[Calibrated Simulation]")
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F16_calibration", output_dir)


# ===================================================================
# F17: Tail ECDF (complementary CDF)
# ===================================================================

def plot_f17_tail_ecdf(sched_df_or_results, output_dir="results/figures"):
    """F17: Tail ECDF -- complementary CDF of p99 latencies.

    sched_df_or_results: DataFrame with columns including scheduler, regime, p99.
    Plots 1x2 panel for structured and adversarial regimes.
    """
    if sched_df_or_results is None or len(sched_df_or_results) == 0:
        print("Warning [F17]: sched_df is empty; skipping.")
        return

    df = sched_df_or_results.copy()
    if "scheduler" not in df.columns or "p99" not in df.columns:
        print("Warning [F17]: required columns missing; skipping.")
        return

    target_regimes = ["structured", "adversarial"]
    if "regime" not in df.columns:
        target_regimes = ["all"]

    n_panels = len(target_regimes)
    fig, axes_arr = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes_arr = [axes_arr]

    focus_schedulers = ["random", "sit_dpp", "static_partition"]
    sched_styles = {
        "random": ("--", COLORS["RANDOM_RED"]),
        "sit_dpp": ("-", COLORS["SIT_GREEN"]),
        "static_partition": (":", COLORS["NEUTRAL_GRAY"]),
    }

    for idx, regime in enumerate(target_regimes):
        ax = axes_arr[idx]
        if regime == "all":
            rdata = df
        else:
            rdata = df[df["regime"] == regime]

        if len(rdata) == 0:
            ax.text(0.5, 0.5, f"No data for {regime}",
                    transform=ax.transAxes, ha="center")
            continue

        # Determine p90 threshold across all schedulers for tail focus
        all_p99 = rdata["p99"].values.astype(float)
        p90_threshold = float(np.percentile(all_p99, 90))

        for sched in focus_schedulers:
            sdata = rdata[rdata["scheduler"] == sched]
            if len(sdata) == 0:
                continue
            vals = np.sort(sdata["p99"].values.astype(float))

            # Complementary CDF: 1 - F(x)
            ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)

            # Filter to tail region (>= p90)
            tail_mask = vals >= p90_threshold
            if tail_mask.sum() < 2:
                tail_mask = np.ones(len(vals), dtype=bool)

            ls, color = sched_styles.get(sched, ("-", COLORS["NEUTRAL_GRAY"]))
            ax.plot(
                vals[tail_mask], ccdf[tail_mask],
                linestyle=ls, color=color, linewidth=2.5,
                label=_nice_sched_name(sched), zorder=3,
            )

        ax.set_yscale("log")
        ax.set_xlabel("p99 Latency ($\\mu$s)")
        ax.set_ylabel("Complementary CDF: P(X > x)")
        regime_title = regime.capitalize() if regime != "all" else "All Regimes"
        ax.set_title(f"{regime_title} Regime", fontsize=13)
        ax.legend(fontsize=10, frameon=True, loc="upper right")

        # Mark the p90 threshold
        ax.axvline(p90_threshold, color=COLORS["GRID_GRAY"], linestyle=":",
                   linewidth=1, alpha=0.7)
        ax.text(p90_threshold, ax.get_ylim()[1] * 0.5, f"  p90={p90_threshold:.0f}",
                fontsize=8, color=COLORS["NEUTRAL_GRAY"], va="center")

    fig.suptitle(
        "F17: Tail Latency ECDF -- Random vs SIT-DPP",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F17_tail_ecdf", output_dir)


# ===================================================================
# F18: Quantile Improvement Heatmap
# ===================================================================

def plot_f18_quantile_heatmap(sched_df, output_dir="results/figures"):
    """F18: Heatmap showing % p99 reduction (SIT-DPP vs random) by target x device.

    sched_df: DataFrame with columns including scheduler, target, device, p99.
    """
    if sched_df is None or len(sched_df) == 0:
        print("Warning [F18]: sched_df is empty; skipping.")
        return

    required = {"scheduler", "target", "device", "p99"}
    if not required.issubset(set(sched_df.columns)):
        print(f"Warning [F18]: missing columns {required - set(sched_df.columns)}; skipping.")
        return

    df = sched_df.copy()

    # Compute mean p99 per (target, device, scheduler)
    sit_agg = df[df["scheduler"] == "sit_dpp"].groupby(["target", "device"])["p99"].mean()
    rand_agg = df[df["scheduler"] == "random"].groupby(["target", "device"])["p99"].mean()

    if len(sit_agg) == 0 or len(rand_agg) == 0:
        print("Warning [F18]: need both sit_dpp and random data; skipping.")
        return

    # Compute % reduction
    common_idx = sit_agg.index.intersection(rand_agg.index)
    reduction = ((rand_agg.loc[common_idx] - sit_agg.loc[common_idx])
                 / rand_agg.loc[common_idx] * 100)
    reduction_df = reduction.reset_index()
    reduction_df.columns = ["target", "device", "reduction_pct"]

    pivot = reduction_df.pivot(index="target", columns="device", values="reduction_pct")

    if pivot.empty:
        print("Warning [F18]: pivot table is empty; skipping.")
        return

    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape

    fig_width = max(8, 1.8 * n_cols + 2)
    fig_height = max(5, 1.2 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Diverging colormap: green for improvement, red for degradation
    vabs = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = plt.cm.RdYlGn  # red=bad, green=good

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=10)

    # Annotate each cell
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val) > vabs * 0.6 else COLORS["DARK_TEXT"]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("p99 Reduction vs Random (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Device", fontsize=12)
    ax.set_ylabel("Target Workload", fontsize=12)
    ax.set_title(
        "F18: p99 Reduction by Target x Device",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F18_quantile_heatmap", output_dir)


# ===================================================================
# F19: CI Coverage Reliability Diagram
# ===================================================================

def plot_f19_ci_coverage(coverage_df, output_dir="results/figures"):
    """F19: Bootstrap CI calibration -- nominal vs empirical coverage.

    coverage_df columns: confidence_level, method, nominal_coverage, empirical_coverage
    """
    if coverage_df is None or len(coverage_df) == 0:
        print("Warning [F19]: coverage_df is empty; skipping.")
        return

    df = coverage_df.copy()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line (diagonal)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["DARK_TEXT"], linewidth=1.5,
            alpha=0.5, label="Perfect Calibration", zorder=2)

    # Shade the overconfident / underconfident regions
    ax.fill_between([0, 1], [0, 1], [0, 0], color=COLORS["LIGHT_RED"],
                    alpha=0.08, label="Overconfident")
    ax.fill_between([0, 1], [0, 1], [1, 1], color=COLORS["LIGHT_GREEN"],
                    alpha=0.08, label="Conservative")

    method_styles = {
        "standard_bootstrap": ("-", "o", COLORS["BASELINE_BLUE"]),
        "block_bootstrap": ("--", "s", COLORS["SIT_GREEN"]),
    }

    methods = df["method"].unique().tolist()
    for method in methods:
        mdata = df[df["method"] == method].sort_values("nominal_coverage")
        xs = mdata["nominal_coverage"].values.astype(float)
        ys = mdata["empirical_coverage"].values.astype(float)

        ls, marker, color = method_styles.get(method, ("-", "D", COLORS["WARN_ORANGE"]))
        nice_name = method.replace("_", " ").title()

        ax.plot(xs, ys, linestyle=ls, marker=marker, color=color,
                linewidth=2.5, markersize=8, label=nice_name, zorder=3)

    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10, frameon=True, loc="lower right")
    ax.set_title(
        "F19: Bootstrap CI Calibration -- Nominal vs Empirical Coverage",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F19_ci_coverage", output_dir)


# ===================================================================
# F20: Overhead Breakdown
# ===================================================================

def plot_f20_overhead(overhead_data, output_dir="results/figures"):
    """F20: SIT pipeline overhead breakdown.

    overhead_data: dict with keys: measurement_time, reconstruction_time,
        scheduling_time, total_time, n_conditions, per_decision_ms
    """
    if overhead_data is None or not isinstance(overhead_data, dict):
        print("Warning [F20]: overhead_data is missing or not a dict; skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    components = ["measurement_time", "reconstruction_time", "scheduling_time"]
    component_labels = ["Measurement\n(Probing)", "Reconstruction\n(Tomography)", "Scheduling\n(Optimization)"]
    component_colors = [COLORS["BASELINE_BLUE"], COLORS["ACCENT_PURPLE"], COLORS["SIT_GREEN"]]

    vals = []
    labels_used = []
    colors_used = []
    for comp, lbl, col in zip(components, component_labels, component_colors):
        if comp in overhead_data:
            vals.append(float(overhead_data[comp]))
            labels_used.append(lbl)
            colors_used.append(col)

    if not vals:
        # Fall back to whatever keys are present
        for key, val in overhead_data.items():
            if key not in ("total_time", "n_conditions", "per_decision_ms") and isinstance(val, (int, float)):
                vals.append(float(val))
                labels_used.append(key.replace("_", " ").title())
                colors_used.append(COLORS["NEUTRAL_GRAY"])

    if not vals:
        print("Warning [F20]: no numeric overhead components found; skipping.")
        plt.close(fig)
        return

    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals, color=colors_used, alpha=0.85, edgecolor="white", height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_used, fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=12)

    # Annotate bar values
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, f"{v:.2f}s", va="center", fontsize=10,
                color=COLORS["DARK_TEXT"])

    # Per-decision latency annotation
    per_decision = overhead_data.get("per_decision_ms", None)
    total_time = overhead_data.get("total_time", sum(vals))
    n_cond = overhead_data.get("n_conditions", None)

    annotation_parts = []
    if total_time is not None:
        annotation_parts.append(f"Total pipeline time: {float(total_time):.1f}s")
    if n_cond is not None:
        annotation_parts.append(f"Conditions: {int(n_cond):,}")
    if per_decision is not None:
        annotation_parts.append(f"Per-decision latency: {float(per_decision):.2f} ms")

    if annotation_parts:
        ax.text(
            0.97, 0.05,
            "\n".join(annotation_parts),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["LIGHT_GREEN"],
                      edgecolor=COLORS["SIT_GREEN"], alpha=0.85),
        )

    ax.set_title(
        "F20: SIT Pipeline Overhead Breakdown",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F20_overhead_breakdown", output_dir)


# ===================================================================
# F21: Ablation Forest Plot
# ===================================================================

def plot_f21_ablation_forest(effect_size_df, output_dir="results/figures"):
    """F21: Ablation forest plot -- component contribution with effect sizes.

    effect_size_df columns: baseline, metric, cohens_d, ci_lo_diff, ci_hi_diff, improvement
    """
    if effect_size_df is None or len(effect_size_df) == 0:
        print("Warning [F21]: effect_size_df is empty; skipping.")
        return

    df = effect_size_df.copy()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(df) + 2)))

    n = len(df)
    y_pos = np.arange(n)

    # Use p99 diff as point estimate, with CI
    if "p99_diff_mean" in df.columns:
        point_est = df["p99_diff_mean"].values.astype(float)
    elif "improvement" in df.columns:
        point_est = df["improvement"].values.astype(float)
    else:
        point_est = df["cohens_d_p99"].values.astype(float) if "cohens_d_p99" in df.columns \
            else df.iloc[:, 1].values.astype(float)

    if "p99_ci_lo" in df.columns:
        ci_lo = df["p99_ci_lo"].values.astype(float)
        ci_hi = df["p99_ci_hi"].values.astype(float)
    elif "ci_lo_diff" in df.columns:
        ci_lo = df["ci_lo_diff"].values.astype(float)
        ci_hi = df["ci_hi_diff"].values.astype(float)
    else:
        ci_lo = point_est - abs(point_est) * 0.2
        ci_hi = point_est + abs(point_est) * 0.2

    # Color by direction: green if improvement > 0, red if <= 0
    colors = [COLORS["SIT_GREEN"] if v > 0 else COLORS["RANDOM_RED"] for v in point_est]

    # Build labels
    if "baseline" in df.columns and "metric" in df.columns:
        labels = [f"{row['baseline']} ({row['metric']})" for _, row in df.iterrows()]
    elif "baseline" in df.columns:
        labels = df["baseline"].tolist()
    else:
        labels = [f"Variant {i}" for i in range(n)]
    labels = [str(lbl).replace("_", " ").title() for lbl in labels]

    # Horizontal error bars (forest plot style)
    for i in range(n):
        ax.plot(
            [ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
            "-", color=colors[i], linewidth=2.0, zorder=2,
        )
        ax.plot(
            point_est[i], y_pos[i], "D",
            color=colors[i], markersize=8, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8,
        )

    # Vertical line at 0 (no effect)
    ax.axvline(0, color=COLORS["DARK_TEXT"], linewidth=1.2, linestyle="--",
               alpha=0.6, zorder=1)

    # Shade negative region (degradation)
    xlim_lo = min(np.min(ci_lo), np.min(point_est)) - abs(np.max(point_est)) * 0.1
    xlim_hi = max(np.max(ci_hi), np.max(point_est)) + abs(np.max(point_est)) * 0.1
    ax.axvspan(xlim_lo, 0, alpha=0.04, color=COLORS["RANDOM_RED"], zorder=0)
    ax.axvspan(0, xlim_hi, alpha=0.04, color=COLORS["SIT_GREEN"], zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Improvement in p99 ($\\mu$s)  [right is better]", fontsize=12)
    ax.invert_yaxis()

    # Cohen's d annotation on the right
    if "cohens_d" in df.columns:
        for i in range(n):
            d_val = float(df["cohens_d"].iloc[i])
            ax.text(xlim_hi * 0.98, y_pos[i], f"d={d_val:.2f}",
                    fontsize=8, color=COLORS["NEUTRAL_GRAY"], va="center", ha="right")

    ax.set_title(
        "F21: Ablation Forest Plot -- Component Contribution",
        fontsize=16, fontweight="bold",
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F21_ablation_forest", output_dir)


# ===================================================================
# F22: Tomography Identifiability Diagnostics
# ===================================================================

def plot_f22_tomography_diagnostics(tomo_diagnostics, output_dir="results/figures"):
    """F22: Tomography identifiability diagnostics.

    tomo_diagnostics: dict with keys from identifiability_report:
        condition_number, mutual_coherence, rank, singular_values (array),
        reconstruction_comparison (DataFrame with columns: method, mse, sparsity)
    """
    if tomo_diagnostics is None or not isinstance(tomo_diagnostics, dict):
        print("Warning [F22]: tomo_diagnostics is missing or not a dict; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel A: Singular value spectrum ----
    ax = axes[0]
    svs = tomo_diagnostics.get("singular_values", None)
    if svs is not None:
        svs = np.asarray(svs, dtype=float)
        n_sv = len(svs)
        x = np.arange(n_sv)

        # Color by magnitude (darker = larger)
        norm_sv = svs / (svs.max() + 1e-12)
        bar_colors = [plt.cm.viridis(0.2 + 0.7 * v) for v in norm_sv]

        ax.bar(x, svs, color=bar_colors, alpha=0.85, edgecolor="white", width=0.7)
        ax.set_xlabel("Singular Value Index", fontsize=11)
        ax.set_ylabel("Singular Value", fontsize=11)
        ax.set_title("A. Singular Value Spectrum", fontsize=13, fontweight="bold")

        # Annotate condition number and rank
        cond_num = tomo_diagnostics.get("condition_number", None)
        rank = tomo_diagnostics.get("rank", None)
        coherence = tomo_diagnostics.get("mutual_coherence", None)
        info_parts = []
        if cond_num is not None:
            info_parts.append(f"Condition #: {float(cond_num):.1f}")
        if rank is not None:
            info_parts.append(f"Rank: {int(rank)}")
        if coherence is not None:
            info_parts.append(f"Mutual coherence: {float(coherence):.3f}")

        if info_parts:
            ax.text(
                0.97, 0.95, "\n".join(info_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=COLORS["GRID_GRAY"], alpha=0.9),
            )

        # Mark effective rank threshold (1% of max)
        thresh = svs.max() * 0.01
        ax.axhline(thresh, color=COLORS["WARN_ORANGE"], linestyle=":",
                   linewidth=1.2, alpha=0.7)
        ax.text(n_sv - 1, thresh * 1.1, "1% threshold", fontsize=8,
                color=COLORS["WARN_ORANGE"], ha="right")
    else:
        ax.text(0.5, 0.5, "No singular values available",
                transform=ax.transAxes, ha="center", fontsize=11)
        ax.set_title("A. Singular Value Spectrum", fontsize=13, fontweight="bold")

    # ---- Panel B: Reconstruction comparison ----
    ax = axes[1]
    recon_df = tomo_diagnostics.get("reconstruction_comparison", None)
    if recon_df is not None and isinstance(recon_df, pd.DataFrame) and len(recon_df) > 0:
        methods = recon_df["method"].tolist() if "method" in recon_df.columns else [f"M{i}" for i in range(len(recon_df))]
        n_methods = len(methods)
        x = np.arange(n_methods)
        width = 0.35

        method_colors = {
            "OLS": COLORS["BASELINE_BLUE"],
            "L1": COLORS["WARN_ORANGE"],
            "Nonneg-L1": COLORS["SIT_GREEN"],
        }

        # MSE bars
        if "mse" in recon_df.columns:
            mse_vals = recon_df["mse"].values.astype(float)
            mse_colors = [method_colors.get(m, COLORS["NEUTRAL_GRAY"]) for m in methods]
            ax.bar(x - width / 2, mse_vals, width, color=mse_colors, alpha=0.85,
                   edgecolor="white", label="MSE")

            for i, v in enumerate(mse_vals):
                ax.text(x[i] - width / 2, v + max(mse_vals) * 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # Sparsity bars (on twin axis)
        if "sparsity" in recon_df.columns:
            ax2 = ax.twinx()
            sparsity_vals = recon_df["sparsity"].values.astype(float)
            sp_colors = [method_colors.get(m, COLORS["NEUTRAL_GRAY"]) for m in methods]
            ax2.bar(x + width / 2, sparsity_vals, width, color=sp_colors, alpha=0.45,
                    edgecolor=sp_colors, linewidth=0.8, hatch="//", label="Sparsity")
            ax2.set_ylabel("Sparsity (fraction of zeros)", fontsize=11)

            for i, v in enumerate(sparsity_vals):
                ax2.text(x[i] + width / 2, v + max(sparsity_vals) * 0.02,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=8)

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                      frameon=True, loc="upper right")
        else:
            ax.legend(fontsize=9, frameon=True)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("MSE", fontsize=11)
        ax.set_title("B. Reconstruction Comparison", fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No reconstruction comparison data",
                transform=ax.transAxes, ha="center", fontsize=11)
        ax.set_title("B. Reconstruction Comparison", fontsize=13, fontweight="bold")

    fig.suptitle(
        "F22: Tomography Identifiability Diagnostics",
        fontsize=16, fontweight="bold", y=1.01,
    )
    _add_watermark(fig)
    _add_source_label(fig)
    fig.tight_layout()
    _save_fig(fig, "F22_tomography_diagnostics", output_dir)


# ===================================================================
# Master generator
# ===================================================================

def generate_all_figures(results: Dict, output_dir: str = "results/figures"):
    """Generate all figures from the results dictionary.

    Missing keys are handled gracefully with a warning.
    """
    print(f"Generating figures in {output_dir}/ ...")

    # F1
    if "bias_df" in results:
        try:
            plot_f1_irbs_bias_demo(results["bias_df"], output_dir)
            print("  F1 done.")
        except Exception as e:
            print(f"  F1 FAILED: {e}")
    else:
        print("  F1 skipped (no bias_df).")

    # F2
    if "phenomenon_df" in results:
        try:
            target = results.get("phenomenon_target", "kv_lookup")
            spectator = results.get("phenomenon_spectator", "cache_thrash")
            plot_f2_phenomenon(results["phenomenon_df"], target, spectator, output_dir)
            print("  F2 done.")
        except Exception as e:
            print(f"  F2 FAILED: {e}")
    else:
        print("  F2 skipped (no phenomenon_df).")

    # F3
    if "tomo_mean" in results:
        try:
            plot_f3_tomography_heatmap(
                results["tomo_mean"],
                results.get("tomo_ci_lower"),
                results.get("tomo_ci_upper"),
                output_dir,
            )
            print("  F3 done.")
        except Exception as e:
            print(f"  F3 FAILED: {e}")
    else:
        print("  F3 skipped (no tomo_mean).")

    # F4
    if "recovery_df" in results:
        try:
            plot_f4_sparse_recovery(results["recovery_df"], output_dir)
            print("  F4 done.")
        except Exception as e:
            print(f"  F4 FAILED: {e}")
    else:
        print("  F4 skipped (no recovery_df).")

    # F5
    if "mismatch_scatter" in results:
        try:
            plot_f5_baseline_mismatch(results["mismatch_scatter"], output_dir)
            print("  F5 done.")
        except Exception as e:
            print(f"  F5 FAILED: {e}")
    else:
        print("  F5 skipped (no mismatch_scatter).")

    # F6
    if "sched_summary" in results:
        try:
            plot_f6_scheduler_comparison(results["sched_summary"], output_dir)
            print("  F6 done.")
        except Exception as e:
            print(f"  F6 FAILED: {e}")
    else:
        print("  F6 skipped (no sched_summary).")

    # F7
    if "worst_case_df" in results:
        try:
            plot_f7_worst_case(results["worst_case_df"], output_dir)
            print("  F7 done.")
        except Exception as e:
            print(f"  F7 FAILED: {e}")
    else:
        print("  F7 skipped (no worst_case_df).")

    # F8
    if "ablation_df" in results:
        try:
            plot_f8_ablations(results["ablation_df"], output_dir)
            print("  F8 done.")
        except Exception as e:
            print(f"  F8 FAILED: {e}")
    else:
        print("  F8 skipped (no ablation_df).")

    # F9
    if "qa_results" in results:
        try:
            plot_f9_qa_summary(results["qa_results"], output_dir)
            print("  F9 done.")
        except Exception as e:
            print(f"  F9 FAILED: {e}")
    else:
        print("  F9 skipped (no qa_results).")

    # F10
    if "channel_df" in results:
        try:
            plot_f10_channel_decomposition(results["channel_df"], output_dir)
            print("  F10 done.")
        except Exception as e:
            print(f"  F10 FAILED: {e}")
    else:
        print("  F10 skipped (no channel_df).")

    # F11
    if "sched_results" in results:
        try:
            plot_f11_sensitivity(results["sched_results"], output_dir)
            print("  F11 done.")
        except Exception as e:
            print(f"  F11 FAILED: {e}")
    elif "sensitivity_df" in results:
        try:
            plot_f11_sensitivity(results["sensitivity_df"], output_dir)
            print("  F11 done.")
        except Exception as e:
            print(f"  F11 FAILED: {e}")
    else:
        print("  F11 skipped (no sched_results / sensitivity_df).")

    # F12
    if "anchoring_summary" in results:
        try:
            plot_f12_anchoring(results["anchoring_summary"], output_dir)
            print("  F12 done.")
        except Exception as e:
            print(f"  F12 FAILED: {e}")
    else:
        print("  F12 skipped (no anchoring_summary).")

    # F13
    if "pareto_summary" in results:
        try:
            plot_f13_pareto_frontier(results["pareto_summary"], output_dir)
            print("  F13 done.")
        except Exception as e:
            print(f"  F13 FAILED: {e}")
    else:
        print("  F13 skipped (no pareto_summary).")

    # F14
    if "probe_summary_df" in results:
        try:
            plot_f14_probe_budget(results["probe_summary_df"], output_dir)
            print("  F14 done.")
        except Exception as e:
            print(f"  F14 FAILED: {e}")
    else:
        print("  F14 skipped (no probe_summary_df).")

    # F15
    if "drift_summary_df" in results:
        try:
            plot_f15_drift_robustness(results["drift_summary_df"], output_dir)
            print("  F15 done.")
        except Exception as e:
            print(f"  F15 FAILED: {e}")
    else:
        print("  F15 skipped (no drift_summary_df).")

    # F16
    if "calibration_summary" in results:
        try:
            plot_f16_calibration(results["calibration_summary"], output_dir)
            print("  F16 done.")
        except Exception as e:
            print(f"  F16 FAILED: {e}")
    elif "anchoring_summary" in results:
        try:
            plot_f16_calibration(results["anchoring_summary"], output_dir)
            print("  F16 done.")
        except Exception as e:
            print(f"  F16 FAILED: {e}")
    else:
        print("  F16 skipped (no calibration_summary / anchoring_summary).")

    # F17
    if "sched_results" in results:
        try:
            plot_f17_tail_ecdf(results["sched_results"], output_dir)
            print("  F17 done.")
        except Exception as e:
            print(f"  F17 FAILED: {e}")
    else:
        print("  F17 skipped (no sched_results).")

    # F18
    if "sched_results" in results:
        try:
            plot_f18_quantile_heatmap(results["sched_results"], output_dir)
            print("  F18 done.")
        except Exception as e:
            print(f"  F18 FAILED: {e}")
    else:
        print("  F18 skipped (no sched_results).")

    # F19
    if "coverage_df" in results:
        try:
            plot_f19_ci_coverage(results["coverage_df"], output_dir)
            print("  F19 done.")
        except Exception as e:
            print(f"  F19 FAILED: {e}")
    else:
        print("  F19 skipped (no coverage_df).")

    # F20
    if "overhead_data" in results:
        try:
            plot_f20_overhead(results["overhead_data"], output_dir)
            print("  F20 done.")
        except Exception as e:
            print(f"  F20 FAILED: {e}")
    else:
        print("  F20 skipped (no overhead_data).")

    # F21
    if "effect_size_df" in results:
        try:
            plot_f21_ablation_forest(results["effect_size_df"], output_dir)
            print("  F21 done.")
        except Exception as e:
            print(f"  F21 FAILED: {e}")
    else:
        print("  F21 skipped (no effect_size_df).")

    # F22
    if "tomo_diagnostics" in results:
        try:
            plot_f22_tomography_diagnostics(results["tomo_diagnostics"], output_dir)
            print("  F22 done.")
        except Exception as e:
            print(f"  F22 FAILED: {e}")
    else:
        print("  F22 skipped (no tomo_diagnostics).")

    print("Figure generation complete.")

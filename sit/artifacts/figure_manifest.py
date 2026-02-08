"""Figure manifest generation for SIT reproducibility.

Produces both a JSON manifest (``figure_manifest.json``) and a
human-readable Markdown summary (``figure_manifest.md``) that record,
for every figure in the report, the output path, generating script,
configuration, data filters, seed list, and git commit.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
#  Figure catalogue                                                    #
# ------------------------------------------------------------------ #

_FIGURE_CATALOGUE: List[Dict[str, Any]] = [
    {
        "figure_id": "F1",
        "filename": "F1_irbs_bias_demo",
        "title": "IRBS Drift-Bias Demonstration",
        "description": (
            "Histogram of naive vs IRBS treatment-effect estimates under "
            "synthetic drift, plus bar chart of bias magnitude."
        ),
        "result_key": "bias_df",
        "filters": "Single target-spectator pair, structured regime, mid load/distance",
        "input_data": ["data/derived/drift_bias_demo.csv"],
    },
    {
        "figure_id": "F2",
        "filename": "F2_phenomenon_ladder",
        "title": "Tail Explosion Under Load and Distance",
        "description": (
            "Load ladder and distance ladder showing p99 and CVaR99 "
            "growth for a high-interference target-spectator pair."
        ),
        "result_key": "phenomenon_df",
        "filters": "First target and spectator from config, all loads/distances/regimes",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F3",
        "filename": "F3_tomography_heatmap",
        "title": "Interference Tomography Heatmap with Uncertainty",
        "description": (
            "Heatmap of delta-p99 for every target-spectator pair with "
            "bootstrap confidence intervals annotated in each cell."
        ),
        "result_key": "tomo_mean",
        "filters": "Reference device, mid distance, mid load, structured regime, aggregated over seeds",
        "input_data": [
            "data/derived/tomography_mean.csv",
            "data/derived/tomography_ci_lower.csv",
            "data/derived/tomography_ci_upper.csv",
        ],
    },
    {
        "figure_id": "F4",
        "filename": "F4_sparse_recovery",
        "title": "Sparse Interferer Recovery Curve",
        "description": (
            "Recovery probability for top-1 and top-3 interferer "
            "identification as a function of trials per spectator."
        ),
        "result_key": "recovery_df",
        "filters": "First target, reference device, mid distance/load, structured regime",
        "input_data": ["data/derived/recovery_curve.csv"],
    },
    {
        "figure_id": "F5",
        "filename": "F5_baseline_mismatch",
        "title": "Baseline Model Mismatch Scatter",
        "description": (
            "Scatter of predicted (naive model) vs observed delta-p99, "
            "highlighting systematic underprediction in the upper tail."
        ),
        "result_key": "mismatch_scatter",
        "filters": "All target-spectator pairs, all loads and distances",
        "input_data": ["data/derived/mismatch_scatter.csv"],
    },
    {
        "figure_id": "F6",
        "filename": "F6_scheduler_comparison",
        "title": "Scheduler Comparison Across Regimes",
        "description": (
            "Grouped bar chart of p99 and CVaR99 for each scheduler "
            "policy, split by benign / structured / adversarial regime."
        ),
        "result_key": "sched_summary",
        "filters": "All schedulers, grouped by regime",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F7",
        "filename": "F7_worst_case",
        "title": "Worst-Case Blowup Avoidance (Top 10% Conditions)",
        "description": (
            "Horizontal bar chart of p99 for each scheduler on the "
            "hardest 10% of conditions as identified by the random baseline."
        ),
        "result_key": "worst_case_df",
        "filters": "Top 10% conditions by random-scheduler p99, adversarial regime",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F8",
        "filename": "F8_ablations",
        "title": "Ablation Study: Component Contributions",
        "description": (
            "Side-by-side bar charts of p99 and CVaR99 across ablation "
            "variants (full SIT-DPP, no-DPP, no-risk, random, etc.)."
        ),
        "result_key": "ablation_df",
        "filters": "All scheduling conditions, aggregated",
        "input_data": ["data/derived/ablations.csv"],
    },
    {
        "figure_id": "F9",
        "filename": "F9_qa_summary",
        "title": "QA Check Summary",
        "description": (
            "Table of pass/fail results for automated QA checks "
            "(fixed-ratio, monotonicity, seed reproducibility, CVaR >= p99)."
        ),
        "result_key": "qa_results",
        "filters": "All QA checks",
        "input_data": [],
    },
    {
        "figure_id": "F10",
        "filename": "F10_channel_decomposition",
        "title": "Channel Decomposition Analysis",
        "description": (
            "Stacked bar chart of interference channel contributions "
            "(LLC, MEM_BW, TLB, PREFETCH, NUMA, THERMAL, OS_FAULTS)."
        ),
        "result_key": "channel_decomp",
        "filters": "All spectators, reference device, mid distance/load",
        "input_data": ["data/derived/channel_decomposition.csv"],
    },
    {
        "figure_id": "F11",
        "filename": "F11_sensitivity_analysis",
        "title": "Sensitivity Analysis",
        "description": (
            "Multi-panel plot showing p99 reduction sensitivity to "
            "load, distance, and other experimental parameters."
        ),
        "result_key": "sensitivity_df",
        "filters": "All parameter combinations, structured+adversarial regimes",
        "input_data": ["data/derived/sensitivity_analysis.csv"],
    },
    {
        "figure_id": "F12",
        "filename": "F12_anchoring_experiment",
        "title": "Real-System Anchoring",
        "description": (
            "Comparison of SIT vs random scheduling on anchoring scenarios "
            "modeled after Triton (ResNet-50), Redis (GET), and gRPC."
        ),
        "result_key": "anchoring_summary",
        "filters": "3 anchoring scenarios with published parameters",
        "input_data": ["data/derived/anchoring_summary.csv"],
    },
    {
        "figure_id": "F13",
        "filename": "F13_pareto_frontier",
        "title": "Pareto Frontier: Tail Safety vs Utilization",
        "description": (
            "Scatter plot of schedulers on the (utilization, p99) plane "
            "with the Pareto frontier highlighted."
        ),
        "result_key": "pareto_summary",
        "filters": "All schedulers, aggregated across conditions",
        "input_data": ["data/derived/pareto_summary.csv"],
    },
    {
        "figure_id": "F14",
        "filename": "F14_probe_budget",
        "title": "Probe Budget Curve",
        "description": (
            "Line plot of ranking quality (NDCG@k, Kendall tau) vs number "
            "of probes for random, round-robin, UCB, and DPP strategies."
        ),
        "result_key": "probe_summary_df",
        "filters": "Probe counts 2-20, 4 selection strategies",
        "input_data": ["data/derived/probe_budget_summary.csv"],
    },
    {
        "figure_id": "F15",
        "filename": "F15_drift_robustness",
        "title": "Drift Robustness Sweep",
        "description": (
            "Two-panel plot: (left) p99 reduction vs drift magnitude for "
            "6 drift types, (right) per-type bars at reference magnitude."
        ),
        "result_key": "drift_summary_df",
        "filters": "6 drift types, magnitudes 0-2x",
        "input_data": ["data/derived/drift_summary.csv"],
    },
    {
        "figure_id": "F16",
        "filename": "F16_calibration",
        "title": "Simulator Calibration vs Published Data",
        "description": (
            "Bar chart comparing published vs simulator p99 latencies "
            "for Triton, Redis, and gRPC anchoring scenarios."
        ),
        "result_key": "anchoring_summary",
        "filters": "3 anchoring scenarios",
        "input_data": ["data/derived/anchoring_summary.csv"],
    },
    {
        "figure_id": "F17",
        "filename": "F17_tail_ecdf",
        "title": "Tail Latency ECDF",
        "description": (
            "Complementary CDF (1-F(x)) of p99 latencies for each scheduler "
            "on a log-x scale, showing tail behavior."
        ),
        "result_key": "sched_results",
        "filters": "All scheduling results, all schedulers",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F18",
        "filename": "F18_quantile_heatmap",
        "title": "Per-Condition p99 Improvement Heatmap",
        "description": (
            "Heatmap of p99 reduction (SIT-DPP vs random) for each "
            "target x device combination."
        ),
        "result_key": "sched_results",
        "filters": "All target-device pairs, SIT-DPP vs random",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F19",
        "filename": "F19_ci_coverage",
        "title": "Bootstrap CI Coverage Reliability Diagram",
        "description": (
            "Reliability diagram showing observed CI coverage vs nominal "
            "level for block bootstrap confidence intervals."
        ),
        "result_key": "ci_coverage_df",
        "filters": "100 outer simulation repeats, nominal levels 80-99%",
        "input_data": ["data/derived/ci_coverage.csv"],
    },
    {
        "figure_id": "F20",
        "filename": "F20_overhead_breakdown",
        "title": "Pipeline Overhead Breakdown",
        "description": (
            "Horizontal bar chart of per-phase execution time showing "
            "that overhead is a small fraction of total runtime."
        ),
        "result_key": "overhead_data",
        "filters": "All pipeline phases",
        "input_data": [],
    },
    {
        "figure_id": "F21",
        "filename": "F21_ablation_forest",
        "title": "Ablation Forest Plot with Effect Sizes",
        "description": (
            "Forest plot showing p99 difference (SIT-DPP vs each baseline) "
            "with bootstrap confidence intervals and Cohen's d."
        ),
        "result_key": "effect_size_df",
        "filters": "All baselines vs SIT-DPP",
        "input_data": ["data/derived/effect_sizes.csv"],
    },
    {
        "figure_id": "F22",
        "filename": "F22_tomography_diagnostics",
        "title": "Tomography Identifiability Diagnostics",
        "description": (
            "Two-panel: (left) SVD spectrum of measurement matrix, "
            "(right) OLS vs L1 vs non-negative L1 reconstruction comparison."
        ),
        "result_key": "tomo_diagnostics",
        "filters": "Tomography matrix from Phase 2",
        "input_data": ["data/derived/reconstruction_comparison.csv"],
    },
    {
        "figure_id": "F23",
        "filename": "F23_slo_throughput",
        "title": "SLO-Satisfying Throughput vs Tail Risk",
        "description": (
            "Two-panel: (left) SLO admission rate across thresholds, "
            "(right) admitted throughput vs tail risk. Shows partition "
            "loses throughput while SIT recovers it with similar tail safety."
        ),
        "result_key": "slo_throughput_df",
        "filters": "All schedulers, multiple SLO thresholds",
        "input_data": ["data/derived/slo_throughput.csv"],
    },
    {
        "figure_id": "F24",
        "filename": "F24_regime_winloss",
        "title": "Regime Win/Loss Map",
        "description": (
            "Heatmaps of SIT improvement vs best baseline by "
            "(distance, load) and (regime, load), showing both wins and losses."
        ),
        "result_key": "sched_results",
        "filters": "All conditions, SIT-DPP vs best non-SIT baseline",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F25",
        "filename": "F25_predicted_vs_realized",
        "title": "Predicted vs Realized Risk (Decision Quality)",
        "description": (
            "Scatter plot of decision-time predicted risk vs realized p99/CVaR99 "
            "for SIT schedulers. Reveals estimator vs policy failures."
        ),
        "result_key": "sched_results",
        "filters": "SIT-DPP and SIT-UCB-DPP conditions with predicted_risk logged",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F26",
        "filename": "F26_cvar_ecdf",
        "title": "CVaR99 Distribution (Catastrophe Decomposition)",
        "description": (
            "Complementary CDF of CVaR99 across conditions for each scheduler, "
            "plus bar chart of mean top-1% catastrophe severity."
        ),
        "result_key": "sched_results",
        "filters": "All conditions, all schedulers",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
    {
        "figure_id": "F27",
        "filename": "F27_regime_failure_heatmap",
        "title": "Regime Failure Heatmap (ΔCVaR vs Best Baseline)",
        "description": (
            "Heatmap of relative CVaR99 difference between SIT-DPP and best "
            "non-SIT baseline, by (regime, load) and (distance, load). "
            "Green = SIT wins, Red = SIT loses."
        ),
        "result_key": "sched_results",
        "filters": "All conditions, per-cell mean ΔCVaR as percentage",
        "input_data": ["data/raw/scheduling_results.parquet"],
    },
]


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _resolve_config_path(config: Dict) -> str:
    """Best-effort extraction of the config file path."""
    # The config dict itself does not always carry its own path.
    # Convention in run_all.py: config is loaded from --config.
    # We look for a breadcrumb; fall back to a generic string.
    return config.get("_config_path", "sit/experiments/configs/<config>.yaml")


# ------------------------------------------------------------------ #
#  Manifest builders                                                   #
# ------------------------------------------------------------------ #

def _build_entries(
    config: Dict,
    all_results: Dict,
) -> List[Dict[str, Any]]:
    """Build a list of manifest entries from the catalogue."""
    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    seeds = config.get("seeds", [])
    config_file = _resolve_config_path(config)
    git_commit = _get_git_commit()

    entries: List[Dict[str, Any]] = []
    for fig in _FIGURE_CATALOGUE:
        fid = fig["figure_id"]

        # Check availability
        result_key = fig.get("result_key")
        data_available = False
        if result_key and result_key in all_results:
            obj = all_results[result_key]
            if obj is not None:
                try:
                    data_available = len(obj) > 0
                except TypeError:
                    data_available = True

        entry: Dict[str, Any] = {
            "figure_id": fid,
            "title": fig["title"],
            "description": fig["description"],
            "output_path": f"{figures_dir}/{fig['filename']}.png",
            "output_path_pdf": f"{figures_dir}/{fig['filename']}.pdf",
            "script_path": "sit/analysis/figures.py",
            "config_file": config_file,
            "filters": fig["filters"],
            "seed_list": seeds,
            "input_data": fig["input_data"],
            "git_commit": git_commit,
            "data_available": data_available,
        }
        entries.append(entry)

    return entries


def _render_markdown(entries: List[Dict[str, Any]]) -> str:
    """Render the manifest entries as a Markdown document."""
    lines: list[str] = [
        "# SIT Figure Manifest\n",
        "This document catalogues every figure generated by the SIT pipeline.\n",
        "| ID | Title | Output | Data Available |",
        "|----|-------|--------|:--------------:|",
    ]
    for e in entries:
        avail = "Yes" if e["data_available"] else "No"
        lines.append(
            f"| {e['figure_id']} | {e['title']} | `{e['output_path']}` | {avail} |"
        )

    lines.append("")
    lines.append("## Details\n")
    for e in entries:
        lines.append(f"### {e['figure_id']}: {e['title']}\n")
        lines.append(f"- **Output (PNG)**: `{e['output_path']}`")
        lines.append(f"- **Output (PDF)**: `{e['output_path_pdf']}`")
        lines.append(f"- **Script**: `{e['script_path']}`")
        lines.append(f"- **Config**: `{e['config_file']}`")
        lines.append(f"- **Filters**: {e['filters']}")
        lines.append(f"- **Seeds**: {e['seed_list']}")
        if e["input_data"]:
            data_str = ", ".join(f"`{d}`" for d in e["input_data"])
            lines.append(f"- **Input data**: {data_str}")
        else:
            lines.append("- **Input data**: *(none -- computed at runtime)*")
        lines.append(f"- **Git commit**: `{e['git_commit']}`")
        lines.append(f"- **Data available**: {e['data_available']}")
        lines.append(f"\n> {e['description']}\n")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def create_manifest(config: Dict, all_results: Dict) -> str:
    """Generate JSON and Markdown figure manifests.

    Parameters
    ----------
    config : Dict
        Parsed YAML configuration.
    all_results : Dict
        Accumulated results dictionary from all experiment phases.

    Returns
    -------
    str
        Absolute path to the saved ``figure_manifest.json`` file.
    """
    manifest_dir = config.get("output", {}).get("manifest_dir", "results/manifest")
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)

    entries = _build_entries(config, all_results)

    # --- JSON manifest ---
    json_path = Path(manifest_dir) / "figure_manifest.json"
    json_path.write_text(
        json.dumps(entries, indent=2, default=str),
        encoding="utf-8",
    )

    # --- Markdown manifest ---
    md_path = Path(manifest_dir) / "figure_manifest.md"
    md_path.write_text(_render_markdown(entries), encoding="utf-8")

    return str(json_path.resolve())

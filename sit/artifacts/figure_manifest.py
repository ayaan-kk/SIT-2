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

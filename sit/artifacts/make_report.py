"""Markdown report generation for SIT reproducibility results.

Generates a comprehensive SIT_Report.md covering motivation, math,
experimental design, results from all phases, QA checks, limitations,
and reproducibility instructions.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_get(d: Dict, *keys, default: Any = None) -> Any:
    """Safely traverse nested dicts / top-level keys."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _fmt(value: Any, precision: int = 2) -> str:
    """Format a numeric value, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any, precision: int = 1) -> str:
    """Format a value as a percentage string."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except (TypeError, ValueError):
        return str(value)


# ------------------------------------------------------------------ #
#  Section builders                                                    #
# ------------------------------------------------------------------ #

def _section_title() -> str:
    return (
        "# SIT: Spectator Interference Tomography - Reproducibility Report\n"
    )


def _section_abstract() -> str:
    return (
        "## Abstract\n\n"
        "Spectator Interference Tomography (SIT) is a systems-science framework for\n"
        "**measuring**, **modeling**, and **scheduling** co-tenant workloads on shared\n"
        "machines with the explicit goal of controlling **tail-risk** (p99 / CVaR99).\n"
        "The approach combines Interleaved Randomized Block Scheduling (IRBS) for\n"
        "drift-robust causal measurement, a structured interference tomography map\n"
        "with bootstrap uncertainty, and a DPP-based scheduler that jointly minimises\n"
        "predicted risk while promoting workload diversity.  This report documents a\n"
        "fully automated, seed-controlled reproducibility run and presents the key\n"
        "results together with QA checks.\n"
    )


def _section_problem() -> str:
    return (
        "## Problem and Motivation\n\n"
        "Modern shared machines (cloud VMs, multi-tenant servers, edge SoCs) host\n"
        "many co-located workloads that contend for hardware resources such as\n"
        "last-level cache (LLC), memory bandwidth, TLB entries, prefetch buffers,\n"
        "NUMA interconnects, thermal headroom, and OS scheduling quanta.\n\n"
        "While **median** latency is often acceptable, **tail events** (p99, p99.9)\n"
        "can blow up by orders of magnitude when a latency-sensitive *target*\n"
        "workload is co-located with a high-pressure *spectator* workload.  These\n"
        "tail blowups are:\n\n"
        "- **Sparse**: only a few spectator-target pairs produce catastrophic tails.\n"
        "- **Non-linear**: they depend on load, placement distance, and regime in\n"
        "  non-additive ways.\n"
        "- **Drift-sensitive**: environmental drift (thermal ramps, DVFS steps,\n"
        "  background daemon bursts) can confound naive A/B measurements.\n\n"
        "SIT addresses all three challenges through causal measurement (IRBS),\n"
        "structured tomography with uncertainty, and tail-risk-aware scheduling\n"
        "(DPP + UCB).\n"
    )


def _section_definitions() -> str:
    return (
        "## Definitions and Mathematical Background\n\n"
        "### Value-at-Risk / Quantile\n\n"
        "$$\n"
        r"\mathrm{VaR}_\alpha(L) \;=\; \inf\{x : P(L \le x) \ge \alpha\}"
        "\n$$\n\n"
        r"For $\alpha = 0.99$ this gives the 99th-percentile latency: $p99 = \mathrm{VaR}_{0.99}$."
        "\n\n"
        "### Conditional Value-at-Risk (CVaR)\n\n"
        "$$\n"
        r"\mathrm{CVaR}_\alpha(L) \;=\; E\bigl[L \mid L \ge \mathrm{VaR}_\alpha(L)\bigr]"
        "\n$$\n\n"
        "Empirical estimator: sort samples $l_{(1)} \\le \\cdots \\le l_{(n)}$, "
        "set $k = \\lceil \\alpha\\,n \\rceil$, then\n\n"
        "$$\n"
        r"\widehat{\mathrm{CVaR}}_\alpha = \frac{1}{n - k} \sum_{i=k+1}^{n} l_{(i)}"
        "\n$$\n\n"
        "### Spectator Sensitivity Index (SSI)\n\n"
        "$$\n"
        r"R_{p99} = \max\!\bigl(0,\;\ln(p99_t / p99_c)\bigr)"
        "\n$$\n"
        "$$\n"
        r"R_{\mathrm{CVaR}} = \max\!\bigl(0,\;\ln(\mathrm{CVaR}_t / \mathrm{CVaR}_c)\bigr)"
        "\n$$\n"
        "$$\n"
        r"\mathrm{SSI} = s \cdot \ln\!\bigl(1 + R_{p99} + w \cdot R_{\mathrm{CVaR}}\bigr)"
        "\n$$\n\n"
        "Default parameters: $s = 2.0$, $w = 0.5$.\n\n"
        "### IRBS Drift Model\n\n"
        "$$\n"
        r"Y_t = \mu + \tau\,Z_t + g(t) + \varepsilon_t"
        "\n$$\n\n"
        "where $Z_t \\in \\{0,1\\}$ is the (randomly interleaved) treatment indicator,\n"
        "$g(t)$ captures drift (thermal ramp, DVFS steps), and $\\varepsilon_t$ is\n"
        "i.i.d. noise.  IRBS decorrelates $Z_t$ from $g(t)$ by construction.\n\n"
        "### DPP Diversity Objective\n\n"
        "$$\n"
        r"\max_{S} \;\ln\det\!\bigl(K_S + \varepsilon\,I\bigr)"
        "\n$$\n\n"
        "where $K_S$ is the kernel sub-matrix for the selected set $S$ and\n"
        "$\\varepsilon$ is a regularisation nugget.\n\n"
        "### UCB Risk Bound\n\n"
        "$$\n"
        r"\mathrm{UCB}_i = \hat\mu_i + \beta \cdot \mathrm{se}_i"
        "\n$$\n\n"
        "where $\\hat\\mu_i$ is the posterior mean interference for spectator $i$\n"
        "and $\\mathrm{se}_i$ is its bootstrap standard error.\n"
    )


def _section_design(config: Dict) -> str:
    """Experimental design derived from the config dict."""
    lines = [
        "## Experimental Design\n",
        "### Factors\n",
        "| Factor | Levels | Values |",
        "|--------|-------:|--------|",
    ]

    factors = [
        ("Devices", config.get("devices", [])),
        ("Targets", config.get("targets", [])),
        ("Spectators", config.get("spectators", [])),
        ("Distances", config.get("distances", [])),
        ("Loads", config.get("loads", [])),
        ("Regimes", config.get("regimes", [])),
        ("Seeds", config.get("seeds", [])),
    ]
    total = 1
    for name, vals in factors:
        n = len(vals)
        total *= max(n, 1)
        display = ", ".join(str(v) for v in vals) if vals else "N/A"
        lines.append(f"| {name} | {n} | {display} |")

    irbs_cfg = config.get("irbs", {})
    n_trials = irbs_cfg.get("n_trials", "?")
    n_samples = irbs_cfg.get("n_samples", "?")

    lines.append("")
    lines.append(f"**Total factorial conditions:** {total}  ")
    lines.append(f"**Trials per condition:** {n_trials}  ")
    lines.append(f"**Samples per trial:** {n_samples}  ")
    try:
        total_trials = total * int(n_trials)
        total_samples = total_trials * int(n_samples)
        lines.append(f"**Total trials:** {total_trials:,}  ")
        lines.append(f"**Total raw samples:** {total_samples:,}  ")
    except (TypeError, ValueError):
        pass
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_results(all_results: Dict) -> str:
    """Embed key per-phase metrics from all_results."""
    parts: list[str] = ["## Results\n"]

    # --- Phase 6 / IRBS bias demo ---
    parts.append("### IRBS Drift-Bias Demonstration\n")
    bias_df = all_results.get("bias_df")
    if bias_df is not None and len(bias_df) > 0:
        true_tau = bias_df["true_tau"].iloc[0]
        naive_bias = bias_df["naive_tau"].mean() - true_tau
        irbs_bias = bias_df["irbs_tau"].mean() - true_tau
        parts.append(
            f"- True treatment effect (tau): **{_fmt(true_tau)}** us\n"
            f"- Naive (A-then-B) bias: **{_fmt(naive_bias)}** us\n"
            f"- IRBS (interleaved) bias: **{_fmt(irbs_bias)}** us\n"
            f"- Bias reduction: **{_fmt(abs(naive_bias) - abs(irbs_bias))}** us\n"
        )
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F1](../figures/F1_irbs_bias_demo.png)\n")

    # --- Phase 2 / Tomography ---
    parts.append("### Tomography Heatmap\n")
    tomo_mean = all_results.get("tomo_mean")
    if tomo_mean is not None:
        n_targets = tomo_mean.shape[0]
        n_specs = tomo_mean.shape[1]
        parts.append(
            f"Interference tomography matrix: **{n_targets}** targets x "
            f"**{n_specs}** spectators.\n"
        )
        sparsity_df = all_results.get("sparsity_df")
        if sparsity_df is not None and len(sparsity_df) > 0:
            mean_top3 = sparsity_df["top3_share"].mean()
            parts.append(
                f"- Mean top-3 interference share: **{_fmt_pct(mean_top3)}**\n"
            )
        # Top interferers
        try:
            flat = tomo_mean.stack().astype(float)
            top5 = flat.nlargest(5)
            parts.append("\n**Top 5 interferer pairs (delta-p99):**\n\n")
            parts.append("| Target | Spectator | delta-p99 (us) |\n")
            parts.append("|--------|-----------|---------------:|\n")
            for (t, s), v in top5.items():
                parts.append(f"| {t} | {s} | {_fmt(v)} |\n")
        except Exception:
            pass
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F3](../figures/F3_tomography_heatmap.png)\n")

    # --- Phase 3 / Sparse recovery ---
    parts.append("### Sparse Recovery Curve\n")
    recovery_df = all_results.get("recovery_df")
    if recovery_df is not None and len(recovery_df) > 0:
        for k_val in sorted(recovery_df["k"].unique()):
            subset = recovery_df[recovery_df["k"] == k_val]
            max_rec = subset["recovery_probability"].max()
            parts.append(
                f"- Top-{k_val} max recovery probability: **{_fmt_pct(max_rec)}**\n"
            )
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F4](../figures/F4_sparse_recovery.png)\n")

    # --- Phase 5 / Baseline mismatch ---
    parts.append("### Baseline Mismatch\n")
    mismatch = all_results.get("mismatch_metrics")
    if mismatch:
        parts.append(
            f"- Underprediction rate (all): **{_fmt_pct(mismatch.get('underprediction_rate'))}**\n"
            f"- Underprediction rate (high-risk): "
            f"**{_fmt_pct(mismatch.get('underprediction_rate_high_risk'))}**\n"
            f"- MAE: **{_fmt(mismatch.get('mae'))}** us\n"
            f"- RMSE: **{_fmt(mismatch.get('rmse'))}** us\n"
        )
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F5](../figures/F5_baseline_mismatch.png)\n")

    # --- Phase 4 / Scheduling ---
    parts.append("### Scheduling Comparison\n")
    sched_df = all_results.get("sched_results")
    if sched_df is not None and hasattr(sched_df, "__len__") and len(sched_df) > 0:
        parts.append("| Scheduler | p99 mean | CVaR99 mean |\n")
        parts.append("|-----------|--------:|-----------:|\n")
        try:
            grouped = sched_df.groupby("scheduler")[["p99", "cvar99"]].mean()
            for sched_name in grouped.index:
                p99_v = grouped.loc[sched_name, "p99"]
                cvar_v = grouped.loc[sched_name, "cvar99"]
                parts.append(f"| {sched_name} | {_fmt(p99_v)} | {_fmt(cvar_v)} |\n")
        except Exception:
            parts.append("| (aggregation error) | | |\n")
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F6](../figures/F6_scheduler_comparison.png)\n")

    # --- Phenomenon ladder ---
    parts.append("### Phenomenon: Tail Explosion Under Load / Distance\n")
    parts.append("\n![F2](../figures/F2_phenomenon_ladder.png)\n")

    # --- Phase 8 / Worst-case ---
    parts.append("### Worst-Case Blowup Avoidance\n")
    wc = all_results.get("worst_case_summary")
    if wc and isinstance(wc, dict) and "error" not in wc:
        parts.append(
            f"- Hardest conditions (top 10%): **{wc.get('n_hard_conditions', 'N/A')}**\n"
            f"- Baseline (random) worst-case p99 mean: **{_fmt(wc.get('baseline_worst_mean'))}** us\n"
            f"- SIT-DPP worst-case p99 mean: **{_fmt(wc.get('sit_worst_mean'))}** us\n"
            f"- Reduction: **{_fmt(wc.get('reduction_pct'))}%**\n"
            f"- Max blowup (baseline): **{_fmt(wc.get('max_blowup_baseline'))}** us\n"
            f"- Max blowup (SIT): **{_fmt(wc.get('max_blowup_sit'))}** us\n"
        )
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F7](../figures/F7_worst_case.png)\n")

    # --- Phase 7 / Ablations ---
    parts.append("### Ablation Study\n")
    ablation_df = all_results.get("ablation_df")
    if ablation_df is not None and len(ablation_df) > 0:
        parts.append("| Variant | p99 | CVaR99 | Description |\n")
        parts.append("|---------|----:|------:|-------------|\n")
        for _, row in ablation_df.iterrows():
            parts.append(
                f"| {row.get('variant', '')} "
                f"| {_fmt(row.get('p99'))} "
                f"| {_fmt(row.get('cvar99'))} "
                f"| {row.get('description', '')} |\n"
            )
    else:
        parts.append("*Data not available.*\n")
    parts.append("\n![F8](../figures/F8_ablations.png)\n")

    return "".join(parts) + "\n"


def _section_qa(all_results: Dict) -> str:
    qa = all_results.get("qa_results")
    if not qa:
        return "## QA Results\n\n*QA data not available.*\n"

    lines = [
        "## QA Results\n",
        "| Check | Status | Details |",
        "|-------|--------|---------|",
    ]
    all_pass = True
    for name, result in qa.items():
        passed = result.get("passed", "N/A")
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        message = str(result.get("message", ""))[:120]
        lines.append(f"| {name} | {status} | {message} |")

    lines.append("")
    overall = "ALL PASSED" if all_pass else "SOME CHECKS FAILED"
    lines.append(f"**Overall QA status: {overall}**\n")
    lines.append("\n![F9](../figures/F9_qa_summary.png)\n")
    return "\n".join(lines) + "\n"


def _section_limitations() -> str:
    return (
        "## Limitations\n\n"
        "1. **Pairwise approximation**: The tomography map captures pairwise\n"
        "   target-spectator interference.  Higher-order interactions among three or\n"
        "   more co-tenants are modelled only approximately (additive with\n"
        "   diminishing returns).\n\n"
        "2. **Kernel similarity as proxy**: The DPP diversity kernel uses cosine\n"
        "   similarity over resource-pressure vectors.  This is a proxy; on real\n"
        "   hardware the effective similarity depends on micro-architectural details\n"
        "   not captured by a 7-dimensional vector.\n\n"
        "3. **Drift window assumptions**: The IRBS estimator assumes drift is slow\n"
        "   relative to a trial block.  Extremely rapid thermal transients or\n"
        "   aggressive DVFS policies could violate this assumption.\n\n"
        "4. **Simulator, not real hardware**: All results in this report are produced\n"
        "   by the SIT simulator.  The simulator is designed to reproduce the\n"
        "   qualitative phenomena (tail explosion, drift bias, sparsity) observed on\n"
        "   real machines, but absolute latency numbers should not be taken at face\n"
        "   value.\n"
    )


def _section_reproducibility(config: Dict) -> str:
    exp = config.get("experiment", {})
    name = exp.get("name", "SIT")
    mode = exp.get("mode", "unknown")
    seeds = config.get("seeds", [])

    return (
        "## Reproducibility\n\n"
        f"- **Experiment name**: {name}\n"
        f"- **Mode**: {mode}\n"
        f"- **Seeds**: {seeds}\n"
        "- **How to rerun**:\n"
        "  ```bash\n"
        "  python -m sit.experiments.run_all --config <path-to-config.yaml>\n"
        "  ```\n"
        "  All random state is controlled by the seed list in the config file.\n"
        "  Given identical software versions and seeds, the pipeline produces\n"
        "  bit-identical results.\n\n"
        "- **Config file**: The YAML configuration file specifies the full\n"
        "  factorial grid (devices, targets, spectators, distances, loads,\n"
        "  regimes, seeds) as well as IRBS, tomography, and scheduling\n"
        "  hyper-parameters.\n"
    )


def _section_appendix(config: Dict) -> str:
    parts = ["## Appendix\n"]

    # A. Parameter tables
    parts.append("### A. Key Parameters\n")
    parts.append("| Parameter | Value |\n")
    parts.append("|-----------|-------|\n")

    irbs_cfg = config.get("irbs", {})
    tomo_cfg = config.get("tomography", {})
    sched_cfg = config.get("scheduling", {})
    params = [
        ("IRBS trials per condition", irbs_cfg.get("n_trials")),
        ("Samples per trial", irbs_cfg.get("n_samples")),
        ("Drift-bias demo repeats", irbs_cfg.get("drift_bias_repeats")),
        ("Bootstrap resamples (tomography)", tomo_cfg.get("n_bootstrap")),
        ("Recovery max trials", tomo_cfg.get("recovery_max_trials")),
        ("Recovery repeats", tomo_cfg.get("recovery_repeats")),
        ("Scheduling slots", sched_cfg.get("n_slots")),
        ("UCB beta", sched_cfg.get("beta_ucb")),
        ("SSI scale s", 2.0),
        ("SSI CVaR weight w", 0.5),
    ]
    for name, val in params:
        parts.append(f"| {name} | {val} |\n")
    parts.append("")

    # B. Workload definitions
    parts.append("### B. Target Workloads\n")
    parts.append(
        "| Name | Base Latency (us) | Burstiness | Description |\n"
        "|------|------------------:|------------|-------------|\n"
    )
    try:
        from sit.simulator.workloads import get_targets
        for name, wl in get_targets().items():
            parts.append(
                f"| {name} | {wl.base_latency_us:.0f} | {wl.burstiness:.2f} "
                f"| {wl.description} |\n"
            )
    except Exception:
        parts.append("| *(import error)* | | | |\n")
    parts.append("")

    parts.append("### C. Spectator Workloads\n")
    parts.append(
        "| Name | Dominant Channel | Description |\n"
        "|------|-----------------|-------------|\n"
    )
    try:
        from sit.simulator.workloads import get_spectators, CHANNELS
        for name, wl in get_spectators().items():
            dom_idx = int(wl.channel_pressure.argmax())
            dom_ch = CHANNELS[dom_idx] if dom_idx < len(CHANNELS) else "?"
            parts.append(f"| {name} | {dom_ch} | {wl.description} |\n")
    except Exception:
        parts.append("| *(import error)* | | |\n")
    parts.append("")

    # D. Device profiles
    parts.append("### D. Device Profiles\n")
    parts.append(
        "| ID | Name | Latency Scale | Thermal Ceiling | Description |\n"
        "|----|------|-------------:|----------------:|-------------|\n"
    )
    try:
        from sit.simulator.device_profiles import get_device_profiles
        for did, dp in get_device_profiles().items():
            parts.append(
                f"| {did} | {dp.name} | {dp.baseline_latency_scale:.2f} "
                f"| {dp.thermal_ceiling:.2f} | {dp.description} |\n"
            )
    except Exception:
        parts.append("| *(import error)* | | | | |\n")
    parts.append("")

    return "".join(parts) + "\n"


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def create_report(config: Dict, all_results: Dict) -> str:
    """Generate the full SIT reproducibility report as Markdown.

    Parameters
    ----------
    config : Dict
        Parsed YAML configuration (as returned by ``load_config``).
    all_results : Dict
        Accumulated results dictionary built across all experiment phases.

    Returns
    -------
    str
        Absolute path to the saved ``SIT_Report.md`` file.
    """
    report_dir = config.get("output", {}).get("report_dir", "results/report")
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        _section_title(),
        f"*Generated: {timestamp}*\n",
        _section_abstract(),
        _section_problem(),
        _section_definitions(),
        _section_design(config),
        _section_results(all_results),
        _section_qa(all_results),
        _section_limitations(),
        _section_reproducibility(config),
        _section_appendix(config),
    ]

    report_text = "\n".join(sections)

    out_path = Path(report_dir) / "SIT_Report.md"
    out_path.write_text(report_text, encoding="utf-8")

    return str(out_path.resolve())

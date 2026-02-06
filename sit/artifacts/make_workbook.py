"""Generate the SIT Results Excel workbook.

Creates SIT_Results.xlsx with sheets covering every experimental result:
Overview, IRBS_Drift_Bias, Tomography, Sparse_Recovery, Baseline_Mismatch,
Scheduling, Worst_Case, Ablations, QA_Checks, Figure_Manifest, How_to_Recompute.

Figures are optionally embedded when Pillow is available.
"""

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_HEADER_FONT = Font(name="Calibri", bold=True, size=12, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)

_SUBHEADER_FONT = Font(name="Calibri", bold=True, size=10)
_SUBHEADER_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")

_LABEL_FONT = Font(name="Calibri", bold=True, size=10)
_VALUE_FONT = Font(name="Calibri", size=10)
_PASS_FILL = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")
_FAIL_FILL = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
_WARN_FILL = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")
_TITLE_FONT = Font(name="Calibri", bold=True, size=14, color="2F5496")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_header_row(ws, row: int, max_col: int):
    """Apply header styling to a row."""
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGNMENT


def _style_subheader_row(ws, row: int, max_col: int):
    """Apply sub-header styling to a row."""
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = _SUBHEADER_FONT
        cell.fill = _SUBHEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)


def _write_title(ws, row: int, col: int, title: str) -> int:
    """Write a bold title cell and return the next row."""
    cell = ws.cell(row=row, column=col, value=title)
    cell.font = _TITLE_FONT
    return row + 1


def _write_label_value(ws, row: int, label: str, value, col_label: int = 1, col_value: int = 2):
    """Write a label/value pair."""
    ws.cell(row=row, column=col_label, value=label).font = _LABEL_FONT
    ws.cell(row=row, column=col_value, value=value).font = _VALUE_FONT


def _write_dataframe(ws, df: pd.DataFrame, start_row: int, start_col: int = 1,
                     include_index: bool = True, header_style: bool = True) -> int:
    """Write a DataFrame to the worksheet and return the next available row.

    Parameters
    ----------
    ws : openpyxl Worksheet
    df : pandas DataFrame
    start_row : first row to write (1-indexed)
    start_col : first column to write (1-indexed)
    include_index : whether to write the DataFrame index as the first column
    header_style : apply sub-header styling to the header row

    Returns
    -------
    int : the next empty row after the written data
    """
    row_offset = start_row
    for r_idx, row in enumerate(dataframe_to_rows(df, index=include_index, header=True)):
        if row_offset == start_row and r_idx == 0 and include_index:
            # dataframe_to_rows yields an extra first row with index name when
            # index=True; we skip pure-None rows.
            if all(v is None for v in row):
                continue
        for c_idx, value in enumerate(row, start=start_col):
            cell = ws.cell(row=row_offset, column=c_idx)
            # Convert numpy types to native Python for openpyxl
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            cell.value = value
            cell.font = _VALUE_FONT
        if header_style and row_offset == start_row:
            max_col = start_col + len(row) - 1
            _style_subheader_row(ws, row_offset, max_col)
        row_offset += 1
    return row_offset


def _try_embed_image(ws, image_path: str, anchor: str):
    """Attempt to embed a PNG image; silently skip if Pillow is unavailable."""
    try:
        from openpyxl.drawing.image import Image as XlImage
        p = Path(image_path)
        if p.exists():
            img = XlImage(str(p))
            img.width = 640
            img.height = 400
            ws.add_image(img, anchor)
    except ImportError:
        pass
    except Exception:
        pass


def _auto_column_widths(ws, min_width: int = 10, max_width: int = 40):
    """Heuristic auto-width for all columns in the worksheet."""
    for col_cells in ws.columns:
        max_len = min_width
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)) + 2, max_width))
        ws.column_dimensions[col_letter].width = max_len


def _not_available_message(ws, row: int, col: int = 1) -> int:
    """Write 'Data not available' and return next row."""
    cell = ws.cell(row=row, column=col, value="Data not available")
    cell.font = Font(name="Calibri", italic=True, size=10, color="999999")
    return row + 2


# ---------------------------------------------------------------------------
# Individual sheet builders
# ---------------------------------------------------------------------------

def _build_overview(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 1: Overview -- headline numbers."""
    ws = wb.active
    ws.title = "Overview"

    row = _write_title(ws, 1, 1, "SIT Results -- Overview")
    row += 1

    # Experiment metadata
    exp_name = config.get("experiment", {}).get("name", "N/A")
    exp_mode = config.get("experiment", {}).get("mode", "N/A")
    _write_label_value(ws, row, "Experiment", exp_name); row += 1
    _write_label_value(ws, row, "Mode", exp_mode); row += 1
    _write_label_value(ws, row, "Conditions", all_results.get("n_conditions", "N/A")); row += 1
    _write_label_value(ws, row, "Total trials", all_results.get("trial_count", "N/A")); row += 1
    _write_label_value(ws, row, "Total samples", all_results.get("sample_count", "N/A")); row += 1
    row += 1

    # Headline: SIT-DPP vs Random for p99 and cvar99 in structured / adversarial
    sched_df = all_results.get("sched_results", None)
    if sched_df is not None and isinstance(sched_df, pd.DataFrame) and len(sched_df) > 0:
        from sit.analysis.significance import bootstrap_mean_ci

        row = _write_title(ws, row, 1, "Headline: SIT-DPP vs Random")
        row += 1

        # Header
        headers = ["Regime", "Metric", "SIT-DPP Mean", "SIT-DPP 95% CI",
                    "Random Mean", "Random 95% CI", "Reduction %"]
        for ci, h in enumerate(headers, 1):
            ws.cell(row=row, column=ci, value=h)
        _style_header_row(ws, row, len(headers))
        row += 1

        for regime in ["structured", "adversarial"]:
            regime_data = sched_df[sched_df["regime"] == regime] if "regime" in sched_df.columns else pd.DataFrame()
            if len(regime_data) == 0:
                continue
            for metric in ["p99", "cvar99"]:
                if metric not in regime_data.columns:
                    continue
                sit_vals = regime_data[regime_data["scheduler"] == "sit_dpp"][metric].values
                rand_vals = regime_data[regime_data["scheduler"] == "random"][metric].values

                if len(sit_vals) > 0 and len(rand_vals) > 0:
                    rng = np.random.default_rng(42)
                    sit_mean, sit_lo, sit_hi = bootstrap_mean_ci(sit_vals, 2000, rng=rng)
                    rng2 = np.random.default_rng(43)
                    rand_mean, rand_lo, rand_hi = bootstrap_mean_ci(rand_vals, 2000, rng=rng2)
                    reduction = (rand_mean - sit_mean) / rand_mean * 100 if rand_mean > 0 else 0.0

                    ws.cell(row=row, column=1, value=regime).font = _VALUE_FONT
                    ws.cell(row=row, column=2, value=metric).font = _VALUE_FONT
                    ws.cell(row=row, column=3, value=round(sit_mean, 2)).font = _VALUE_FONT
                    ws.cell(row=row, column=4, value=f"[{sit_lo:.2f}, {sit_hi:.2f}]").font = _VALUE_FONT
                    ws.cell(row=row, column=5, value=round(rand_mean, 2)).font = _VALUE_FONT
                    ws.cell(row=row, column=6, value=f"[{rand_lo:.2f}, {rand_hi:.2f}]").font = _VALUE_FONT
                    ws.cell(row=row, column=7, value=round(reduction, 2)).font = _VALUE_FONT
                    row += 1
    else:
        row = _not_available_message(ws, row)

    row += 1

    # QA summary
    qa = all_results.get("qa_results", None)
    if qa is not None and isinstance(qa, dict) and len(qa) > 0:
        row = _write_title(ws, row, 1, "QA Summary")
        row += 1
        all_pass = all(v.get("passed", True) for v in qa.values())
        _write_label_value(ws, row, "Overall", "ALL PASSED" if all_pass else "SOME FAILED")
        ws.cell(row=row, column=2).fill = _PASS_FILL if all_pass else _FAIL_FILL
        row += 1

    _auto_column_widths(ws)


def _build_irbs_drift_bias(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 2: IRBS_Drift_Bias."""
    ws = wb.create_sheet("IRBS_Drift_Bias")

    row = _write_title(ws, 1, 1, "IRBS Drift Bias Demonstration")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F1_irbs_bias_demo.png", f"A{row}")
    row += 22  # leave space for embedded figure

    bias_df = all_results.get("bias_df", None)
    if bias_df is not None and isinstance(bias_df, pd.DataFrame) and len(bias_df) > 0:
        row = _write_title(ws, row, 1, "Drift Bias Data")
        row += 1
        row = _write_dataframe(ws, bias_df, row)
        row += 1

        # Summary statistics
        true_tau = bias_df["true_tau"].iloc[0] if "true_tau" in bias_df.columns else 0.0
        naive_bias = bias_df["naive_tau"].mean() - true_tau if "naive_tau" in bias_df.columns else 0.0
        irbs_bias = bias_df["irbs_tau"].mean() - true_tau if "irbs_tau" in bias_df.columns else 0.0

        row = _write_title(ws, row, 1, "Summary Statistics")
        row += 1
        _write_label_value(ws, row, "True treatment effect", round(float(true_tau), 4)); row += 1
        _write_label_value(ws, row, "Naive mean bias", round(float(naive_bias), 4)); row += 1
        _write_label_value(ws, row, "IRBS mean bias", round(float(irbs_bias), 4)); row += 1
        _write_label_value(ws, row, "Bias reduction (abs)",
                           round(abs(float(naive_bias)) - abs(float(irbs_bias)), 4)); row += 1
        if "naive_tau" in bias_df.columns:
            _write_label_value(ws, row, "Naive std", round(float(bias_df["naive_tau"].std()), 4)); row += 1
        if "irbs_tau" in bias_df.columns:
            _write_label_value(ws, row, "IRBS std", round(float(bias_df["irbs_tau"].std()), 4)); row += 1
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_tomography(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 3: Tomography."""
    ws = wb.create_sheet("Tomography")

    row = _write_title(ws, 1, 1, "Interference Tomography Matrix")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F3_tomography_heatmap.png", f"A{row}")
    row += 22

    tomo_mean = all_results.get("tomo_mean", None)
    if tomo_mean is not None and isinstance(tomo_mean, pd.DataFrame) and len(tomo_mean) > 0:
        row = _write_title(ws, row, 1, "Mean Interference Matrix (delta-p99, us)")
        row += 1
        row = _write_dataframe(ws, tomo_mean, row, include_index=True)
        row += 1

        # Top interferers per target (top 3)
        row = _write_title(ws, row, 1, "Top 3 Interferers per Target")
        row += 1
        headers = ["Target", "Rank 1", "Effect 1", "Rank 2", "Effect 2", "Rank 3", "Effect 3"]
        for ci, h in enumerate(headers, 1):
            ws.cell(row=row, column=ci, value=h)
        _style_header_row(ws, row, len(headers))
        row += 1

        for target in tomo_mean.index:
            row_vals = tomo_mean.loc[target].astype(float)
            sorted_specs = row_vals.sort_values(ascending=False)
            top3 = sorted_specs.head(3)
            ws.cell(row=row, column=1, value=str(target)).font = _LABEL_FONT
            for rank_i, (spec_name, effect_val) in enumerate(top3.items()):
                ws.cell(row=row, column=2 + rank_i * 2, value=str(spec_name)).font = _VALUE_FONT
                ws.cell(row=row, column=3 + rank_i * 2, value=round(float(effect_val), 2)).font = _VALUE_FONT
            row += 1
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_sparse_recovery(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 4: Sparse_Recovery."""
    ws = wb.create_sheet("Sparse_Recovery")

    row = _write_title(ws, 1, 1, "Sparse Interferer Recovery")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F4_sparse_recovery.png", f"A{row}")
    row += 22

    recovery_df = all_results.get("recovery_df", None)
    if recovery_df is not None and isinstance(recovery_df, pd.DataFrame) and len(recovery_df) > 0:
        row = _write_title(ws, row, 1, "Recovery Probability Data")
        row += 1
        row = _write_dataframe(ws, recovery_df, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_baseline_mismatch(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 5: Baseline_Mismatch."""
    ws = wb.create_sheet("Baseline_Mismatch")

    row = _write_title(ws, 1, 1, "Baseline Mismatch Analysis")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F5_baseline_mismatch.png", f"A{row}")
    row += 22

    # Mismatch metrics
    mismatch_metrics = all_results.get("mismatch_metrics", None)
    if mismatch_metrics is not None and isinstance(mismatch_metrics, dict) and len(mismatch_metrics) > 0:
        row = _write_title(ws, row, 1, "Mismatch Metrics")
        row += 1
        for key, val in mismatch_metrics.items():
            if isinstance(val, float):
                display_val = round(val, 6)
            else:
                display_val = val
            _write_label_value(ws, row, str(key), display_val)
            row += 1
        row += 1
    else:
        _write_label_value(ws, row, "Mismatch Metrics", "Data not available")
        row += 2

    # Scatter data subset
    scatter_df = all_results.get("mismatch_scatter", None)
    if scatter_df is not None and isinstance(scatter_df, pd.DataFrame) and len(scatter_df) > 0:
        row = _write_title(ws, row, 1, "Scatter Data (first 200 rows)")
        row += 1
        subset = scatter_df.head(200)
        row = _write_dataframe(ws, subset, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_scheduling(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 6: Scheduling."""
    ws = wb.create_sheet("Scheduling")

    row = _write_title(ws, 1, 1, "Scheduling Comparison")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F6_scheduler_comparison.png", f"A{row}")
    row += 22

    sched_df = all_results.get("sched_results", None)
    if sched_df is not None and isinstance(sched_df, pd.DataFrame) and len(sched_df) > 0:
        # Aggregated summary: groupby scheduler and regime
        agg_cols = {}
        if "p99" in sched_df.columns:
            agg_cols["p99"] = "mean"
        if "cvar99" in sched_df.columns:
            agg_cols["cvar99"] = "mean"
        if "mean" in sched_df.columns:
            agg_cols["mean_latency"] = ("mean", "mean")

        group_keys = ["scheduler"]
        if "regime" in sched_df.columns:
            group_keys.append("regime")

        # Build aggregation safely
        metrics_for_agg = [m for m in ["p99", "cvar99", "mean", "p95"] if m in sched_df.columns]
        if metrics_for_agg:
            summary = sched_df.groupby(group_keys)[metrics_for_agg].mean().reset_index()

            row = _write_title(ws, row, 1, "Aggregated: Mean p99 and CVaR99 by Scheduler / Regime")
            row += 1
            row = _write_dataframe(ws, summary, row, include_index=False)
            row += 1

        # Also write per-regime breakdown for SIT-DPP vs Random
        if "regime" in sched_df.columns and "scheduler" in sched_df.columns:
            row = _write_title(ws, row, 1, "SIT-DPP vs Random Detail")
            row += 1
            filtered = sched_df[sched_df["scheduler"].isin(["sit_dpp", "random"])]
            if len(filtered) > 0:
                detail = filtered.groupby(["scheduler", "regime"])[metrics_for_agg].agg(
                    ["mean", "std"]
                ).reset_index()
                detail.columns = ["_".join(col).strip("_") for col in detail.columns]
                row = _write_dataframe(ws, detail, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_worst_case(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 7: Worst_Case."""
    ws = wb.create_sheet("Worst_Case")

    row = _write_title(ws, 1, 1, "Worst-Case Blowup Avoidance")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F7_worst_case.png", f"A{row}")
    row += 22

    # Worst-case summary dict
    wc_summary = all_results.get("worst_case_summary", None)
    if wc_summary is not None and isinstance(wc_summary, dict) and len(wc_summary) > 0:
        row = _write_title(ws, row, 1, "Worst-Case Summary Metrics")
        row += 1
        for key, val in wc_summary.items():
            if isinstance(val, float):
                display_val = round(val, 4)
            else:
                display_val = val
            _write_label_value(ws, row, str(key), display_val)
            row += 1
        row += 1
    else:
        _write_label_value(ws, row, "Worst-Case Summary", "Data not available")
        row += 2

    # Worst-case DataFrame
    wc_df = all_results.get("worst_case_df", None)
    if wc_df is not None and isinstance(wc_df, pd.DataFrame) and len(wc_df) > 0:
        row = _write_title(ws, row, 1, "Worst-Case Condition Results")
        row += 1
        row = _write_dataframe(ws, wc_df, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_ablations(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 8: Ablations."""
    ws = wb.create_sheet("Ablations")

    row = _write_title(ws, 1, 1, "Ablation Study")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F8_ablations.png", f"A{row}")
    row += 22

    ablation_df = all_results.get("ablation_df", None)
    if ablation_df is not None and isinstance(ablation_df, pd.DataFrame) and len(ablation_df) > 0:
        row = _write_title(ws, row, 1, "Ablation Variants")
        row += 1
        row = _write_dataframe(ws, ablation_df, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_qa_checks(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 9: QA_Checks."""
    ws = wb.create_sheet("QA_Checks")

    row = _write_title(ws, 1, 1, "QA Check Results")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    _try_embed_image(ws, f"{figures_dir}/F9_qa_summary.png", f"A{row}")
    row += 22

    qa_results = all_results.get("qa_results", None)
    if qa_results is not None and isinstance(qa_results, dict) and len(qa_results) > 0:
        row = _write_title(ws, row, 1, "Individual Test Results")
        row += 1

        headers = ["Test Name", "Status", "Message"]
        for ci, h in enumerate(headers, 1):
            ws.cell(row=row, column=ci, value=h)
        _style_header_row(ws, row, len(headers))
        row += 1

        for test_name, result in qa_results.items():
            passed = result.get("passed", None)
            message = result.get("message", "")

            if passed is True:
                status = "PASS"
                fill = _PASS_FILL
            elif passed is False:
                status = "FAIL"
                fill = _FAIL_FILL
            else:
                status = "N/A"
                fill = _WARN_FILL

            ws.cell(row=row, column=1, value=test_name).font = _VALUE_FONT
            status_cell = ws.cell(row=row, column=2, value=status)
            status_cell.font = Font(name="Calibri", bold=True, size=10)
            status_cell.fill = fill
            status_cell.alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=3, value=str(message)[:200]).font = _VALUE_FONT
            row += 1

        row += 1
        all_pass = all(v.get("passed", True) for v in qa_results.values())
        _write_label_value(ws, row, "Overall Verdict", "ALL PASSED" if all_pass else "SOME FAILED")
        verdict_cell = ws.cell(row=row, column=2)
        verdict_cell.fill = _PASS_FILL if all_pass else _FAIL_FILL
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_figure_manifest(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 10: Figure_Manifest."""
    ws = wb.create_sheet("Figure_Manifest")

    row = _write_title(ws, 1, 1, "Figure Manifest")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")

    headers = ["Figure ID", "Description", "Output Path (PNG)", "Output Path (PDF)", "Script Reference"]
    for ci, h in enumerate(headers, 1):
        ws.cell(row=row, column=ci, value=h)
    _style_header_row(ws, row, len(headers))
    row += 1

    manifest = [
        ("F1", "IRBS Drift Bias Demonstration",
         "F1_irbs_bias_demo", "sit.analysis.figures.plot_f1_irbs_bias_demo"),
        ("F2", "Tail Explosion Under High-Risk Spectator",
         "F2_phenomenon_ladder", "sit.analysis.figures.plot_f2_phenomenon"),
        ("F3", "Interference Tomography Heatmap with CI",
         "F3_tomography_heatmap", "sit.analysis.figures.plot_f3_tomography_heatmap"),
        ("F4", "Sparse Interferer Recovery vs Sampling",
         "F4_sparse_recovery", "sit.analysis.figures.plot_f4_sparse_recovery"),
        ("F5", "Naive Model Mismatch (Underpredicts Tail Risk)",
         "F5_baseline_mismatch", "sit.analysis.figures.plot_f5_baseline_mismatch"),
        ("F6", "Scheduler Comparison Across Regimes",
         "F6_scheduler_comparison", "sit.analysis.figures.plot_f6_scheduler_comparison"),
        ("F7", "Worst-Case Blowup Avoidance",
         "F7_worst_case", "sit.analysis.figures.plot_f7_worst_case"),
        ("F8", "Ablation Study -- Component Contributions",
         "F8_ablations", "sit.analysis.figures.plot_f8_ablations"),
        ("F9", "QA Check Summary",
         "F9_qa_summary", "sit.analysis.figures.plot_f9_qa_summary"),
    ]

    for fig_id, desc, basename, script_ref in manifest:
        png_path = f"{figures_dir}/{basename}.png"
        pdf_path = f"{figures_dir}/{basename}.pdf"
        exists_str = ""
        if Path(png_path).exists():
            exists_str = " [exists]"

        ws.cell(row=row, column=1, value=fig_id).font = _LABEL_FONT
        ws.cell(row=row, column=2, value=desc).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=png_path + exists_str).font = _VALUE_FONT
        ws.cell(row=row, column=4, value=pdf_path).font = _VALUE_FONT
        ws.cell(row=row, column=5, value=script_ref).font = _VALUE_FONT
        row += 1

    _auto_column_widths(ws)


def _build_how_to_recompute(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 11: How_to_Recompute."""
    ws = wb.create_sheet("How_to_Recompute")

    row = _write_title(ws, 1, 1, "How to Recompute These Results")
    row += 1
    row += 1

    _write_label_value(ws, row, "Quick run (pipeline validation, ~2-5 min):", "")
    row += 1
    cmd_cell = ws.cell(row=row, column=1,
                       value="python -m sit.experiments.run_all --config sit/experiments/configs/quick.yaml")
    cmd_cell.font = Font(name="Consolas", size=10)
    row += 2

    _write_label_value(ws, row, "Full run (complete results):", "")
    row += 1
    cmd_cell = ws.cell(row=row, column=1,
                       value="python -m sit.experiments.run_all --config sit/experiments/configs/full.yaml")
    cmd_cell.font = Font(name="Consolas", size=10)
    row += 2

    row = _write_title(ws, row, 1, "Notes")
    row += 1
    notes = [
        "Both commands are deterministic (seeded RNG). Identical results across runs.",
        "Quick mode uses a reduced grid for fast pipeline validation.",
        "Full mode runs the complete factorial grid and all analyses.",
        "The workbook, figures, and tables are regenerated on each run.",
        "Output directories are created automatically.",
        "Requirements: numpy, pandas, openpyxl, matplotlib, scipy, pyyaml.",
    ]
    for note in notes:
        ws.cell(row=row, column=1, value=note).font = _VALUE_FONT
        row += 1

    _auto_column_widths(ws)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_workbook(config: Dict, all_results: Dict) -> str:
    """Create the SIT_Results.xlsx workbook.

    Parameters
    ----------
    config : Dict
        Experiment configuration (from YAML).
    all_results : Dict
        Dictionary of all experimental results produced by the pipeline.

    Returns
    -------
    str
        Absolute path to the saved workbook file.
    """
    workbook_dir = config.get("output", {}).get("workbook_dir", "results/workbook")
    Path(workbook_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(workbook_dir) / "SIT_Results.xlsx")

    wb = openpyxl.Workbook()

    # Build every sheet
    _build_overview(wb, config, all_results)
    _build_irbs_drift_bias(wb, config, all_results)
    _build_tomography(wb, config, all_results)
    _build_sparse_recovery(wb, config, all_results)
    _build_baseline_mismatch(wb, config, all_results)
    _build_scheduling(wb, config, all_results)
    _build_worst_case(wb, config, all_results)
    _build_ablations(wb, config, all_results)
    _build_qa_checks(wb, config, all_results)
    _build_figure_manifest(wb, config, all_results)
    _build_how_to_recompute(wb, config, all_results)

    wb.save(output_path)
    return output_path

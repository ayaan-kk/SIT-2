"""Generate the SIT Results Excel workbook with editable charts.

Creates SIT_Results.xlsx with sheets covering every experimental result.
All charts are native Excel charts (editable in Excel/Google Sheets).

Sheets:
  Overview, IRBS_Drift_Bias, Tomography, Sparse_Recovery, Baseline_Mismatch,
  Scheduling, Worst_Case, Ablations, QA_Checks, Channel_Decomposition,
  Sensitivity, Cross_Validation, Figure_Manifest, How_to_Recompute
"""

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
from openpyxl.chart import (
    BarChart, LineChart, ScatterChart, Reference, Series,
)
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.series import DataPoint
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_HEADER_FONT = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="2d6a4f", end_color="2d6a4f", fill_type="solid")
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)

_SUBHEADER_FONT = Font(name="Calibri", bold=True, size=10)
_SUBHEADER_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")

_LABEL_FONT = Font(name="Calibri", bold=True, size=10)
_VALUE_FONT = Font(name="Calibri", size=10)
_PASS_FILL = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")
_FAIL_FILL = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
_WARN_FILL = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")
_TITLE_FONT = Font(name="Calibri", bold=True, size=14, color="2d6a4f")
_SECTION_FONT = Font(name="Calibri", bold=True, size=12, color="2F5496")
_THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)

# Chart colour palette (hex, no leading #)
_CHART_COLORS = [
    "2d6a4f", "c1121f", "1d3557", "e76f51", "7209b7",
    "6c757d", "264653", "e9c46a", "f4a261", "2a9d8f",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_header_row(ws, row: int, max_col: int):
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGNMENT
        cell.border = _THIN_BORDER


def _style_subheader_row(ws, row: int, max_col: int):
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = _SUBHEADER_FONT
        cell.fill = _SUBHEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = _THIN_BORDER


def _write_title(ws, row: int, col: int, title: str) -> int:
    cell = ws.cell(row=row, column=col, value=title)
    cell.font = _TITLE_FONT
    return row + 1


def _write_section(ws, row: int, col: int, title: str) -> int:
    cell = ws.cell(row=row, column=col, value=title)
    cell.font = _SECTION_FONT
    return row + 1


def _write_label_value(ws, row: int, label: str, value, col_label: int = 1, col_value: int = 2):
    ws.cell(row=row, column=col_label, value=label).font = _LABEL_FONT
    c = ws.cell(row=row, column=col_value, value=value)
    c.font = _VALUE_FONT
    return row + 1


def _write_dataframe(ws, df: pd.DataFrame, start_row: int, start_col: int = 1,
                     include_index: bool = True, header_style: bool = True) -> int:
    """Write a DataFrame to the worksheet and return the next available row."""
    row_offset = start_row
    for r_idx, row in enumerate(dataframe_to_rows(df, index=include_index, header=True)):
        if row_offset == start_row and r_idx == 0 and include_index:
            if all(v is None for v in row):
                continue
        for c_idx, value in enumerate(row, start=start_col):
            cell = ws.cell(row=row_offset, column=c_idx)
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            cell.value = value
            cell.font = _VALUE_FONT
            cell.border = _THIN_BORDER
        if header_style and row_offset == start_row:
            max_col = start_col + len(row) - 1
            _style_subheader_row(ws, row_offset, max_col)
        row_offset += 1
    return row_offset


def _auto_column_widths(ws, min_width: int = 10, max_width: int = 40):
    for col_cells in ws.columns:
        max_len = min_width
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)) + 2, max_width))
        ws.column_dimensions[col_letter].width = max_len


def _not_available_message(ws, row: int, col: int = 1) -> int:
    cell = ws.cell(row=row, column=col, value="Data not available")
    cell.font = Font(name="Calibri", italic=True, size=10, color="999999")
    return row + 2


def _safe_float(val) -> float:
    """Convert to float or return 0."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _make_bar_chart(title: str, x_title: str, y_title: str,
                    width: int = 18, height: int = 12) -> BarChart:
    c = BarChart()
    c.title = title
    c.x_axis.title = x_title
    c.y_axis.title = y_title
    c.width = width
    c.height = height
    c.style = 10
    c.legend.position = 'b'
    return c


def _make_line_chart(title: str, x_title: str, y_title: str,
                     width: int = 18, height: int = 12) -> LineChart:
    c = LineChart()
    c.title = title
    c.x_axis.title = x_title
    c.y_axis.title = y_title
    c.width = width
    c.height = height
    c.style = 10
    c.legend.position = 'b'
    return c


def _make_scatter_chart(title: str, x_title: str, y_title: str,
                        width: int = 16, height: int = 12) -> ScatterChart:
    c = ScatterChart()
    c.title = title
    c.x_axis.title = x_title
    c.y_axis.title = y_title
    c.width = width
    c.height = height
    c.style = 10
    c.legend.position = 'b'
    return c


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _build_overview(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 1: Overview."""
    ws = wb.active
    ws.title = "Overview"

    row = _write_title(ws, 1, 1, "SIT: Spectator Interference Tomography — Results Overview")
    row += 1

    exp_name = config.get("experiment", {}).get("name", "N/A")
    exp_mode = config.get("experiment", {}).get("mode", "N/A")
    row = _write_label_value(ws, row, "Experiment", exp_name)
    row = _write_label_value(ws, row, "Mode", exp_mode)
    row = _write_label_value(ws, row, "Conditions", all_results.get("n_conditions", "N/A"))
    row = _write_label_value(ws, row, "Total trials", all_results.get("trial_count", "N/A"))
    row = _write_label_value(ws, row, "Total samples", all_results.get("sample_count", "N/A"))
    row += 1

    # Headline table: SIT-DPP vs Random per regime
    sched_df = all_results.get("sched_results")
    if sched_df is not None and isinstance(sched_df, pd.DataFrame) and len(sched_df) > 0:
        row = _write_section(ws, row, 1, "Headline: SIT-DPP vs Random")
        row += 1

        headers = ["Regime", "Metric", "SIT-DPP Mean", "Random Mean", "Reduction %"]
        for ci, h in enumerate(headers, 1):
            ws.cell(row=row, column=ci, value=h)
        _style_header_row(ws, row, len(headers))
        data_start_row = row + 1
        row += 1

        for regime in ["structured", "adversarial", "benign"]:
            regime_data = sched_df[sched_df["regime"] == regime] if "regime" in sched_df.columns else pd.DataFrame()
            if len(regime_data) == 0:
                continue
            for metric in ["p99", "cvar99"]:
                if metric not in regime_data.columns:
                    continue
                sit_vals = regime_data[regime_data["scheduler"] == "sit_dpp"][metric].values
                rand_vals = regime_data[regime_data["scheduler"] == "random"][metric].values
                if len(sit_vals) > 0 and len(rand_vals) > 0:
                    sit_mean = float(np.mean(sit_vals))
                    rand_mean = float(np.mean(rand_vals))
                    reduction = (rand_mean - sit_mean) / rand_mean * 100 if rand_mean > 0 else 0.0
                    ws.cell(row=row, column=1, value=regime).font = _VALUE_FONT
                    ws.cell(row=row, column=2, value=metric).font = _VALUE_FONT
                    ws.cell(row=row, column=3, value=round(sit_mean, 2)).font = _VALUE_FONT
                    ws.cell(row=row, column=4, value=round(rand_mean, 2)).font = _VALUE_FONT
                    c = ws.cell(row=row, column=5, value=round(reduction, 1))
                    c.font = _VALUE_FONT
                    c.number_format = '0.0"%"'
                    row += 1
        data_end_row = row - 1

        # Bar chart: SIT vs Random reduction %
        if data_end_row >= data_start_row:
            chart = _make_bar_chart(
                "SIT-DPP Reduction vs Random (%)", "Regime / Metric", "Reduction %"
            )
            cats = Reference(ws, min_col=1, max_col=2, min_row=data_start_row, max_row=data_end_row)
            vals = Reference(ws, min_col=5, min_row=data_start_row - 1, max_row=data_end_row)
            chart.add_data(vals, titles_from_data=True)
            chart.set_categories(cats)
            chart.series[0].graphicalProperties.solidFill = _CHART_COLORS[0]
            row += 1
            ws.add_chart(chart, f"A{row}")
            row += 16

    # QA summary
    qa = all_results.get("qa_results")
    if qa is not None and isinstance(qa, dict) and len(qa) > 0:
        row += 1
        row = _write_section(ws, row, 1, "QA Summary")
        row += 1
        all_pass = all(v.get("passed", True) for v in qa.values())
        row = _write_label_value(ws, row, "Overall",
                                 "ALL PASSED" if all_pass else "SOME FAILED")
        ws.cell(row=row - 1, column=2).fill = _PASS_FILL if all_pass else _FAIL_FILL
        row = _write_label_value(ws, row, "Total checks", len(qa))
        passed_count = sum(1 for v in qa.values() if v.get("passed"))
        row = _write_label_value(ws, row, "Passed", passed_count)

    _auto_column_widths(ws)


def _build_irbs_drift_bias(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 2: IRBS_Drift_Bias with bar chart."""
    ws = wb.create_sheet("IRBS_Drift_Bias")
    row = _write_title(ws, 1, 1, "IRBS Drift Bias Demonstration")
    row += 1

    bias_df = all_results.get("bias_df")
    if bias_df is not None and isinstance(bias_df, pd.DataFrame) and len(bias_df) > 0:
        # Summary stats
        true_tau = float(bias_df["true_tau"].iloc[0]) if "true_tau" in bias_df.columns else 0.0
        naive_bias = float(bias_df["naive_tau"].mean() - true_tau) if "naive_tau" in bias_df.columns else 0.0
        irbs_bias = float(bias_df["irbs_tau"].mean() - true_tau) if "irbs_tau" in bias_df.columns else 0.0

        row = _write_label_value(ws, row, "True treatment effect (tau)", round(true_tau, 2))
        row = _write_label_value(ws, row, "Naive mean bias", round(naive_bias, 2))
        row = _write_label_value(ws, row, "IRBS mean bias", round(irbs_bias, 2))
        row = _write_label_value(ws, row, "Bias reduction", round(abs(naive_bias) - abs(irbs_bias), 2))
        row += 1

        # Write comparison data table for chart
        row = _write_section(ws, row, 1, "Comparison Data")
        row += 1
        ws.cell(row=row, column=1, value="Protocol"); ws.cell(row=row, column=2, value="Mean Estimate")
        ws.cell(row=row, column=3, value="Abs Bias"); ws.cell(row=row, column=4, value="Std Dev")
        _style_header_row(ws, row, 4)
        chart_header_row = row
        row += 1
        ws.cell(row=row, column=1, value="Naive").font = _VALUE_FONT
        ws.cell(row=row, column=2, value=round(float(bias_df["naive_tau"].mean()), 2)).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=round(abs(naive_bias), 2)).font = _VALUE_FONT
        ws.cell(row=row, column=4, value=round(float(bias_df["naive_tau"].std()), 2)).font = _VALUE_FONT
        row += 1
        ws.cell(row=row, column=1, value="IRBS").font = _VALUE_FONT
        ws.cell(row=row, column=2, value=round(float(bias_df["irbs_tau"].mean()), 2)).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=round(abs(irbs_bias), 2)).font = _VALUE_FONT
        ws.cell(row=row, column=4, value=round(float(bias_df["irbs_tau"].std()), 2)).font = _VALUE_FONT
        data_end = row
        row += 1

        # Bar chart: Abs Bias comparison
        chart = _make_bar_chart("IRBS vs Naive: Absolute Bias", "Protocol", "Absolute Bias (us)")
        cats = Reference(ws, min_col=1, min_row=chart_header_row + 1, max_row=data_end)
        vals = Reference(ws, min_col=3, min_row=chart_header_row, max_row=data_end)
        chart.add_data(vals, titles_from_data=True)
        chart.set_categories(cats)
        chart.series[0].graphicalProperties.solidFill = _CHART_COLORS[1]
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 16

        # Raw data table
        row += 1
        row = _write_section(ws, row, 1, "Raw Repeat Data")
        row += 1
        row = _write_dataframe(ws, bias_df, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_tomography(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 3: Tomography with bar chart of top interferers."""
    ws = wb.create_sheet("Tomography")
    row = _write_title(ws, 1, 1, "Interference Tomography Matrix")
    row += 1

    tomo_mean = all_results.get("tomo_mean")
    if tomo_mean is not None and isinstance(tomo_mean, pd.DataFrame) and len(tomo_mean) > 0:
        # Full matrix
        row = _write_section(ws, row, 1, "Mean Interference Matrix (delta-p99, us)")
        row += 1
        matrix_start = row
        row = _write_dataframe(ws, tomo_mean, row, include_index=True)
        row += 1

        # Top interferers table for chart
        row = _write_section(ws, row, 1, "Top Interferer Pairs")
        row += 1
        ws.cell(row=row, column=1, value="Pair"); ws.cell(row=row, column=2, value="Delta-p99 (us)")
        _style_header_row(ws, row, 2)
        chart_header = row
        row += 1

        flat = tomo_mean.stack().astype(float)
        top10 = flat.nlargest(min(10, len(flat)))
        for (t, s), v in top10.items():
            ws.cell(row=row, column=1, value=f"{t} + {s}").font = _VALUE_FONT
            ws.cell(row=row, column=2, value=round(float(v), 1)).font = _VALUE_FONT
            row += 1
        data_end = row - 1

        # Horizontal bar chart of top interferers
        chart = BarChart()
        chart.type = "bar"
        chart.title = "Top Interferer Pairs (Delta-p99)"
        chart.x_axis.title = "Delta-p99 (us)"
        chart.y_axis.title = "Target + Spectator Pair"
        chart.width = 20
        chart.height = 14
        chart.style = 10
        cats = Reference(ws, min_col=1, min_row=chart_header + 1, max_row=data_end)
        vals = Reference(ws, min_col=2, min_row=chart_header, max_row=data_end)
        chart.add_data(vals, titles_from_data=True)
        chart.set_categories(cats)
        chart.series[0].graphicalProperties.solidFill = _CHART_COLORS[1]
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 18

        # Sparsity stats
        sparsity_df = all_results.get("sparsity_df")
        if sparsity_df is not None and isinstance(sparsity_df, pd.DataFrame) and len(sparsity_df) > 0:
            row += 1
            row = _write_section(ws, row, 1, "Sparsity Statistics")
            row += 1
            row = _write_dataframe(ws, sparsity_df, row, include_index=False)

        # CI matrices
        for key, label in [("tomo_ci_lower", "CI Lower"), ("tomo_ci_upper", "CI Upper"),
                            ("tomo_stderr", "Standard Error")]:
            mat = all_results.get(key)
            if mat is not None and isinstance(mat, pd.DataFrame):
                row += 1
                row = _write_section(ws, row, 1, f"{label} Matrix")
                row += 1
                row = _write_dataframe(ws, mat, row, include_index=True)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_sparse_recovery(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 4: Sparse_Recovery with line chart."""
    ws = wb.create_sheet("Sparse_Recovery")
    row = _write_title(ws, 1, 1, "Sparse Interferer Recovery")
    row += 1

    recovery_df = all_results.get("recovery_df")
    if recovery_df is not None and isinstance(recovery_df, pd.DataFrame) and len(recovery_df) > 0:
        row = _write_section(ws, row, 1, "Recovery Probability Data")
        row += 1
        data_start = row
        row = _write_dataframe(ws, recovery_df, row, include_index=False)
        row += 1

        # Pivot for chart: trials vs recovery prob by k
        k_values = sorted(recovery_df["k"].unique())
        trial_values = sorted(recovery_df["trials_per_spectator"].unique())

        # Write chart-friendly table
        row = _write_section(ws, row, 1, "Recovery by k (for chart)")
        row += 1
        ws.cell(row=row, column=1, value="Trials per Spectator")
        for ki, k in enumerate(k_values):
            ws.cell(row=row, column=2 + ki, value=f"k={int(k)}")
        _style_header_row(ws, row, 1 + len(k_values))
        chart_header = row
        row += 1

        for t in trial_values:
            ws.cell(row=row, column=1, value=int(t)).font = _VALUE_FONT
            for ki, k in enumerate(k_values):
                subset = recovery_df[(recovery_df["trials_per_spectator"] == t) & (recovery_df["k"] == k)]
                val = float(subset["recovery_probability"].iloc[0]) if len(subset) > 0 else 0
                ws.cell(row=row, column=2 + ki, value=round(val, 3)).font = _VALUE_FONT
            row += 1
        data_end = row - 1

        # Line chart
        chart = _make_line_chart("Sparse Recovery Curve", "Trials per Spectator", "Recovery Probability")
        cats = Reference(ws, min_col=1, min_row=chart_header + 1, max_row=data_end)
        for ki in range(len(k_values)):
            vals = Reference(ws, min_col=2 + ki, min_row=chart_header, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
        chart.set_categories(cats)
        for i, s in enumerate(chart.series):
            s.graphicalProperties.line.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            s.graphicalProperties.line.width = 25000
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 16
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_baseline_mismatch(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 5: Baseline_Mismatch with scatter chart."""
    ws = wb.create_sheet("Baseline_Mismatch")
    row = _write_title(ws, 1, 1, "Baseline Mismatch Analysis")
    row += 1

    mismatch_metrics = all_results.get("mismatch_metrics")
    if mismatch_metrics is not None and isinstance(mismatch_metrics, dict) and len(mismatch_metrics) > 0:
        row = _write_section(ws, row, 1, "Mismatch Metrics")
        row += 1
        for key, val in mismatch_metrics.items():
            display_val = round(val, 4) if isinstance(val, float) else val
            row = _write_label_value(ws, row, str(key), display_val)
        row += 1

    scatter_df = all_results.get("mismatch_scatter")
    if scatter_df is not None and isinstance(scatter_df, pd.DataFrame) and len(scatter_df) > 0:
        # Write observed vs predicted for scatter chart
        row = _write_section(ws, row, 1, "Observed vs Predicted (first 500 rows)")
        row += 1
        ws.cell(row=row, column=1, value="Observed"); ws.cell(row=row, column=2, value="Predicted")
        _style_header_row(ws, row, 2)
        chart_header = row
        row += 1

        subset = scatter_df.head(500)
        for _, r in subset.iterrows():
            ws.cell(row=row, column=1, value=round(_safe_float(r.get("observed")), 2)).font = _VALUE_FONT
            ws.cell(row=row, column=2, value=round(_safe_float(r.get("predicted")), 2)).font = _VALUE_FONT
            row += 1
        data_end = row - 1

        # Scatter chart
        chart = _make_scatter_chart("Observed vs Predicted Interference", "Observed (us)", "Predicted (us)")
        xvalues = Reference(ws, min_col=1, min_row=chart_header + 1, max_row=data_end)
        yvalues = Reference(ws, min_col=2, min_row=chart_header + 1, max_row=data_end)
        series = Series(yvalues, xvalues, title="Conditions")
        chart.series.append(series)
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 16
    else:
        if mismatch_metrics is None:
            row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_scheduling(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 6: Scheduling with grouped bar charts."""
    ws = wb.create_sheet("Scheduling")
    row = _write_title(ws, 1, 1, "Scheduling Comparison")
    row += 1

    sched_df = all_results.get("sched_results")
    if sched_df is not None and isinstance(sched_df, pd.DataFrame) and len(sched_df) > 0:
        metrics_for_agg = [m for m in ["p99", "cvar99", "mean", "p95"] if m in sched_df.columns]

        # Overall comparison table
        if metrics_for_agg:
            summary = sched_df.groupby("scheduler")[metrics_for_agg].mean().reset_index()
            row = _write_section(ws, row, 1, "Overall Mean by Scheduler")
            row += 1
            table_start = row
            row = _write_dataframe(ws, summary, row, include_index=False)
            table_end = row - 1
            row += 1

            # Bar chart: p99 and cvar99 by scheduler
            n_scheds = len(summary)
            chart = _make_bar_chart("Scheduler Comparison: p99 and CVaR99", "Scheduler", "Latency (us)")
            chart.grouping = "clustered"
            cats = Reference(ws, min_col=1, min_row=table_start + 1, max_row=table_end)

            for mi, metric in enumerate(["p99", "cvar99"]):
                if metric in summary.columns:
                    col_idx = list(summary.columns).index(metric) + 1
                    vals = Reference(ws, min_col=col_idx, min_row=table_start, max_row=table_end)
                    chart.add_data(vals, titles_from_data=True)
            chart.set_categories(cats)
            for i, s in enumerate(chart.series):
                s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            ws.add_chart(chart, f"A{row}")
            row += 16

        # Per-regime breakdown
        if "regime" in sched_df.columns:
            row += 1
            regime_summary = sched_df.groupby(["scheduler", "regime"])[metrics_for_agg].mean().reset_index()
            row = _write_section(ws, row, 1, "Per-Regime Breakdown")
            row += 1
            regime_table_start = row
            row = _write_dataframe(ws, regime_summary, row, include_index=False)
            row += 1

            # Chart per regime
            for regime in sorted(sched_df["regime"].unique()):
                regime_data = regime_summary[regime_summary["regime"] == regime].copy()
                if len(regime_data) == 0:
                    continue

                # Write mini-table for this regime chart
                row += 1
                row = _write_section(ws, row, 1, f"Chart Data: {regime.capitalize()} Regime")
                row += 1
                ws.cell(row=row, column=1, value="Scheduler")
                ws.cell(row=row, column=2, value="p99")
                ws.cell(row=row, column=3, value="cvar99")
                _style_header_row(ws, row, 3)
                mini_start = row
                row += 1
                for _, r in regime_data.iterrows():
                    ws.cell(row=row, column=1, value=str(r.get("scheduler", ""))).font = _VALUE_FONT
                    ws.cell(row=row, column=2, value=round(_safe_float(r.get("p99")), 1)).font = _VALUE_FONT
                    ws.cell(row=row, column=3, value=round(_safe_float(r.get("cvar99")), 1)).font = _VALUE_FONT
                    row += 1
                mini_end = row - 1

                chart = _make_bar_chart(
                    f"Scheduler Comparison: {regime.capitalize()} Regime",
                    "Scheduler", "Latency (us)"
                )
                chart.grouping = "clustered"
                cats = Reference(ws, min_col=1, min_row=mini_start + 1, max_row=mini_end)
                for col in [2, 3]:
                    vals = Reference(ws, min_col=col, min_row=mini_start, max_row=mini_end)
                    chart.add_data(vals, titles_from_data=True)
                chart.set_categories(cats)
                for i, s in enumerate(chart.series):
                    s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
                ws.add_chart(chart, f"A{row + 1}")
                row += 17

        # Full raw data (first 2000 rows)
        row += 1
        row = _write_section(ws, row, 1, "Raw Scheduling Data (first 2000)")
        row += 1
        row = _write_dataframe(ws, sched_df.head(2000), row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_worst_case(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 7: Worst_Case."""
    ws = wb.create_sheet("Worst_Case")
    row = _write_title(ws, 1, 1, "Worst-Case Blowup Avoidance")
    row += 1

    wc_summary = all_results.get("worst_case_summary")
    if wc_summary is not None and isinstance(wc_summary, dict) and len(wc_summary) > 0:
        row = _write_section(ws, row, 1, "Summary Metrics")
        row += 1
        for key, val in wc_summary.items():
            display_val = round(val, 2) if isinstance(val, float) else val
            row = _write_label_value(ws, row, str(key), display_val)
        row += 1

        # Bar chart: baseline vs SIT worst-case
        row = _write_section(ws, row, 1, "Chart Data")
        row += 1
        ws.cell(row=row, column=1, value="Scheduler")
        ws.cell(row=row, column=2, value="Worst-Case p99 Mean")
        ws.cell(row=row, column=3, value="Max Blowup")
        _style_header_row(ws, row, 3)
        chart_h = row
        row += 1
        ws.cell(row=row, column=1, value="Random").font = _VALUE_FONT
        ws.cell(row=row, column=2, value=round(_safe_float(wc_summary.get("baseline_worst_mean")), 1)).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=round(_safe_float(wc_summary.get("max_blowup_baseline")), 1)).font = _VALUE_FONT
        row += 1
        ws.cell(row=row, column=1, value="SIT-DPP").font = _VALUE_FONT
        ws.cell(row=row, column=2, value=round(_safe_float(wc_summary.get("sit_worst_mean")), 1)).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=round(_safe_float(wc_summary.get("max_blowup_sit")), 1)).font = _VALUE_FONT
        data_end = row
        row += 1

        chart = _make_bar_chart("Worst-Case Analysis: Random vs SIT-DPP", "Scheduler", "Latency (us)")
        chart.grouping = "clustered"
        cats = Reference(ws, min_col=1, min_row=chart_h + 1, max_row=data_end)
        for col in [2, 3]:
            vals = Reference(ws, min_col=col, min_row=chart_h, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
        chart.set_categories(cats)
        for i, s in enumerate(chart.series):
            s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
        ws.add_chart(chart, f"A{row + 1}")
        row += 17
    else:
        row = _not_available_message(ws, row)

    wc_df = all_results.get("worst_case_df")
    if wc_df is not None and isinstance(wc_df, pd.DataFrame) and len(wc_df) > 0:
        row += 1
        row = _write_section(ws, row, 1, "Worst-Case Condition Details")
        row += 1
        row = _write_dataframe(ws, wc_df, row, include_index=False)

    _auto_column_widths(ws)


def _build_ablations(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 8: Ablations with bar chart."""
    ws = wb.create_sheet("Ablations")
    row = _write_title(ws, 1, 1, "Ablation Study")
    row += 1

    ablation_df = all_results.get("ablation_df")
    if ablation_df is not None and isinstance(ablation_df, pd.DataFrame) and len(ablation_df) > 0:
        row = _write_section(ws, row, 1, "Ablation Variants")
        row += 1

        # Write data table
        ws.cell(row=row, column=1, value="Variant")
        ws.cell(row=row, column=2, value="p99")
        ws.cell(row=row, column=3, value="cvar99")
        ws.cell(row=row, column=4, value="mean")
        ws.cell(row=row, column=5, value="Description")
        _style_header_row(ws, row, 5)
        chart_h = row
        row += 1

        for _, r in ablation_df.iterrows():
            ws.cell(row=row, column=1, value=str(r.get("variant", ""))).font = _VALUE_FONT
            ws.cell(row=row, column=2, value=round(_safe_float(r.get("p99")), 1)).font = _VALUE_FONT
            ws.cell(row=row, column=3, value=round(_safe_float(r.get("cvar99")), 1)).font = _VALUE_FONT
            ws.cell(row=row, column=4, value=round(_safe_float(r.get("mean")), 1)).font = _VALUE_FONT
            ws.cell(row=row, column=5, value=str(r.get("description", ""))).font = _VALUE_FONT
            row += 1
        data_end = row - 1

        # Bar chart: p99 and cvar99 per variant
        chart = _make_bar_chart("Ablation Study: p99 and CVaR99", "Variant", "Latency (us)")
        chart.grouping = "clustered"
        cats = Reference(ws, min_col=1, min_row=chart_h + 1, max_row=data_end)
        for col in [2, 3]:
            vals = Reference(ws, min_col=col, min_row=chart_h, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
        chart.set_categories(cats)
        for i, s in enumerate(chart.series):
            s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 16
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_qa_checks(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 9: QA_Checks."""
    ws = wb.create_sheet("QA_Checks")
    row = _write_title(ws, 1, 1, "QA Check Results")
    row += 1

    qa_results = all_results.get("qa_results")
    if qa_results is not None and isinstance(qa_results, dict) and len(qa_results) > 0:
        headers = ["Test Name", "Status", "Message"]
        for ci, h in enumerate(headers, 1):
            ws.cell(row=row, column=ci, value=h)
        _style_header_row(ws, row, len(headers))
        row += 1

        for test_name, result in qa_results.items():
            passed = result.get("passed")
            message = result.get("message", "")
            if passed is True:
                status, fill = "PASS", _PASS_FILL
            elif passed is False:
                status, fill = "FAIL", _FAIL_FILL
            else:
                status, fill = "N/A", _WARN_FILL

            ws.cell(row=row, column=1, value=test_name).font = _VALUE_FONT
            sc = ws.cell(row=row, column=2, value=status)
            sc.font = Font(name="Calibri", bold=True, size=10)
            sc.fill = fill
            sc.alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=3, value=str(message)[:200]).font = _VALUE_FONT
            row += 1

        row += 1
        all_pass = all(v.get("passed", True) for v in qa_results.values())
        row = _write_label_value(ws, row, "Overall Verdict",
                                 "ALL PASSED" if all_pass else "SOME FAILED")
        ws.cell(row=row - 1, column=2).fill = _PASS_FILL if all_pass else _FAIL_FILL
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_channel_decomposition(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 10: Channel_Decomposition with stacked bar chart."""
    ws = wb.create_sheet("Channel_Decomp")
    row = _write_title(ws, 1, 1, "Channel Decomposition Analysis")
    row += 1

    channel_df = all_results.get("channel_df")
    if channel_df is not None and isinstance(channel_df, pd.DataFrame) and len(channel_df) > 0:
        # Aggregate: mean per-channel severity by channel
        if "channel" in channel_df.columns and "per_channel_severity" in channel_df.columns:
            ch_agg = channel_df.groupby("channel")["per_channel_severity"].agg(["sum", "mean"]).reset_index()
            ch_agg.columns = ["channel", "total_severity", "mean_severity"]
            ch_agg = ch_agg.sort_values("total_severity", ascending=False)

            row = _write_section(ws, row, 1, "Channel Contribution Summary")
            row += 1
            ws.cell(row=row, column=1, value="Channel")
            ws.cell(row=row, column=2, value="Total Severity")
            ws.cell(row=row, column=3, value="Mean Severity")
            ws.cell(row=row, column=4, value="Share (%)")
            _style_header_row(ws, row, 4)
            chart_h = row
            row += 1

            grand_total = ch_agg["total_severity"].sum()
            for _, r in ch_agg.iterrows():
                ws.cell(row=row, column=1, value=str(r["channel"])).font = _VALUE_FONT
                ws.cell(row=row, column=2, value=round(float(r["total_severity"]), 2)).font = _VALUE_FONT
                ws.cell(row=row, column=3, value=round(float(r["mean_severity"]), 4)).font = _VALUE_FONT
                share = float(r["total_severity"]) / grand_total * 100 if grand_total > 0 else 0
                ws.cell(row=row, column=4, value=round(share, 1)).font = _VALUE_FONT
                row += 1
            data_end = row - 1

            # Bar chart: channel contributions
            chart = _make_bar_chart("Interference by Channel", "Channel", "Total Severity")
            cats = Reference(ws, min_col=1, min_row=chart_h + 1, max_row=data_end)
            vals = Reference(ws, min_col=2, min_row=chart_h, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
            chart.set_categories(cats)
            for i, s in enumerate(chart.series):
                s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            row += 1
            ws.add_chart(chart, f"A{row}")
            row += 16

            # Stacked bar: per target-spectator pair (top 15)
            pair_agg = channel_df.groupby(["target", "spectator", "channel"])[
                "per_channel_severity"].mean().reset_index()
            pair_totals = pair_agg.groupby(["target", "spectator"])[
                "per_channel_severity"].sum().reset_index()
            pair_totals.columns = ["target", "spectator", "total"]
            pair_totals = pair_totals.sort_values("total", ascending=False).head(15)
            top_pairs = list(zip(pair_totals["target"], pair_totals["spectator"]))

            channels_list = sorted(pair_agg["channel"].unique())

            row += 1
            row = _write_section(ws, row, 1, "Top 15 Pairs: Per-Channel Breakdown")
            row += 1
            ws.cell(row=row, column=1, value="Pair")
            for ci, ch in enumerate(channels_list):
                ws.cell(row=row, column=2 + ci, value=ch)
            _style_header_row(ws, row, 1 + len(channels_list))
            stacked_h = row
            row += 1

            for t, s in top_pairs:
                ws.cell(row=row, column=1, value=f"{t} + {s}").font = _VALUE_FONT
                for ci, ch in enumerate(channels_list):
                    sub = pair_agg[(pair_agg["target"] == t) & (pair_agg["spectator"] == s) & (pair_agg["channel"] == ch)]
                    val = float(sub["per_channel_severity"].iloc[0]) if len(sub) > 0 else 0
                    ws.cell(row=row, column=2 + ci, value=round(val, 4)).font = _VALUE_FONT
                row += 1
            stacked_end = row - 1

            # Stacked bar chart
            chart2 = BarChart()
            chart2.type = "col"
            chart2.grouping = "stacked"
            chart2.title = "Per-Channel Interference (Top 15 Pairs)"
            chart2.x_axis.title = "Target + Spectator"
            chart2.y_axis.title = "Severity"
            chart2.width = 22
            chart2.height = 14
            chart2.style = 10
            chart2.legend.position = 'b'

            cats2 = Reference(ws, min_col=1, min_row=stacked_h + 1, max_row=stacked_end)
            for ci in range(len(channels_list)):
                vals2 = Reference(ws, min_col=2 + ci, min_row=stacked_h, max_row=stacked_end)
                chart2.add_data(vals2, titles_from_data=True)
            chart2.set_categories(cats2)
            for i, s in enumerate(chart2.series):
                s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            row += 1
            ws.add_chart(chart2, f"A{row}")
            row += 18

        # Full raw data (first 1000)
        row += 1
        row = _write_section(ws, row, 1, "Raw Channel Data (first 1000)")
        row += 1
        row = _write_dataframe(ws, channel_df.head(1000), row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_sensitivity(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 11: Sensitivity analysis with line charts."""
    ws = wb.create_sheet("Sensitivity")
    row = _write_title(ws, 1, 1, "Sensitivity Analysis")
    row += 1

    sensitivity_df = all_results.get("sensitivity_df")
    if sensitivity_df is not None and isinstance(sensitivity_df, pd.DataFrame) and len(sensitivity_df) > 0:
        row = _write_section(ws, row, 1, "Full Sensitivity Data")
        row += 1
        row = _write_dataframe(ws, sensitivity_df, row, include_index=False)
        row += 1

        # Chart: p99 reduction by load
        load_data = sensitivity_df[sensitivity_df["parameter"] == "load"].copy()
        if len(load_data) > 0:
            row += 1
            row = _write_section(ws, row, 1, "p99 Reduction vs Load (Chart Data)")
            row += 1
            ws.cell(row=row, column=1, value="Load")
            ws.cell(row=row, column=2, value="p99 Reduction %")
            ws.cell(row=row, column=3, value="CVaR99 Reduction %")
            _style_header_row(ws, row, 3)
            ch_start = row
            row += 1

            # Average across regimes for a clean chart
            load_avg = load_data.groupby("value")[["p99_reduction_pct", "cvar99_reduction_pct"]].mean().reset_index()
            load_avg = load_avg.sort_values("value")
            for _, r in load_avg.iterrows():
                ws.cell(row=row, column=1, value=round(float(r["value"]), 2)).font = _VALUE_FONT
                ws.cell(row=row, column=2, value=round(float(r["p99_reduction_pct"]), 1)).font = _VALUE_FONT
                ws.cell(row=row, column=3, value=round(float(r["cvar99_reduction_pct"]), 1)).font = _VALUE_FONT
                row += 1
            ch_end = row - 1

            chart = _make_line_chart("SIT-DPP Improvement vs Load", "Load Level", "Reduction %")
            cats = Reference(ws, min_col=1, min_row=ch_start + 1, max_row=ch_end)
            for col in [2, 3]:
                vals = Reference(ws, min_col=col, min_row=ch_start, max_row=ch_end)
                chart.add_data(vals, titles_from_data=True)
            chart.set_categories(cats)
            for i, s in enumerate(chart.series):
                s.graphicalProperties.line.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
                s.graphicalProperties.line.width = 25000
            row += 1
            ws.add_chart(chart, f"A{row}")
            row += 16

        # Chart: p99 reduction by distance
        dist_data = sensitivity_df[sensitivity_df["parameter"] == "distance"].copy()
        if len(dist_data) > 0:
            row += 1
            row = _write_section(ws, row, 1, "p99 Reduction vs Distance (Chart Data)")
            row += 1
            ws.cell(row=row, column=1, value="Distance")
            ws.cell(row=row, column=2, value="p99 Reduction %")
            ws.cell(row=row, column=3, value="CVaR99 Reduction %")
            _style_header_row(ws, row, 3)
            ch_start = row
            row += 1

            dist_avg = dist_data.groupby("value")[["p99_reduction_pct", "cvar99_reduction_pct"]].mean().reset_index()
            for _, r in dist_avg.iterrows():
                ws.cell(row=row, column=1, value=str(r["value"])).font = _VALUE_FONT
                ws.cell(row=row, column=2, value=round(float(r["p99_reduction_pct"]), 1)).font = _VALUE_FONT
                ws.cell(row=row, column=3, value=round(float(r["cvar99_reduction_pct"]), 1)).font = _VALUE_FONT
                row += 1
            ch_end = row - 1

            chart2 = _make_bar_chart("SIT-DPP Improvement vs Distance", "Distance", "Reduction %")
            cats2 = Reference(ws, min_col=1, min_row=ch_start + 1, max_row=ch_end)
            for col in [2, 3]:
                vals2 = Reference(ws, min_col=col, min_row=ch_start, max_row=ch_end)
                chart2.add_data(vals2, titles_from_data=True)
            chart2.set_categories(cats2)
            for i, s in enumerate(chart2.series):
                s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            row += 1
            ws.add_chart(chart2, f"A{row}")
            row += 16

        # Chart: by device
        dev_data = sensitivity_df[sensitivity_df["parameter"] == "device"].copy()
        if len(dev_data) > 0:
            row += 1
            row = _write_section(ws, row, 1, "p99 Reduction vs Device (Chart Data)")
            row += 1
            ws.cell(row=row, column=1, value="Device")
            ws.cell(row=row, column=2, value="p99 Reduction %")
            ws.cell(row=row, column=3, value="CVaR99 Reduction %")
            _style_header_row(ws, row, 3)
            ch_start = row
            row += 1

            for _, r in dev_data.iterrows():
                ws.cell(row=row, column=1, value=str(r["value"])).font = _VALUE_FONT
                ws.cell(row=row, column=2, value=round(float(r["p99_reduction_pct"]), 1)).font = _VALUE_FONT
                ws.cell(row=row, column=3, value=round(float(r["cvar99_reduction_pct"]), 1)).font = _VALUE_FONT
                row += 1
            ch_end = row - 1

            chart3 = _make_bar_chart("SIT-DPP Improvement vs Device", "Device", "Reduction %")
            cats3 = Reference(ws, min_col=1, min_row=ch_start + 1, max_row=ch_end)
            for col in [2, 3]:
                vals3 = Reference(ws, min_col=col, min_row=ch_start, max_row=ch_end)
                chart3.add_data(vals3, titles_from_data=True)
            chart3.set_categories(cats3)
            for i, s in enumerate(chart3.series):
                s.graphicalProperties.solidFill = _CHART_COLORS[i % len(_CHART_COLORS)]
            row += 1
            ws.add_chart(chart3, f"A{row}")
            row += 16
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_cross_validation(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 12: Cross_Validation with scatter chart."""
    ws = wb.create_sheet("Cross_Validation")
    row = _write_title(ws, 1, 1, "Tomography Cross-Validation")
    row += 1

    cv_df = all_results.get("cv_df")
    cv_mae = all_results.get("cv_mae")
    cv_corr = all_results.get("cv_correlation")

    if cv_mae is not None:
        row = _write_label_value(ws, row, "Mean Absolute Error (MAE)", round(float(cv_mae), 2))
    if cv_corr is not None:
        row = _write_label_value(ws, row, "Correlation (r)", round(float(cv_corr), 4))
    row += 1

    if cv_df is not None and isinstance(cv_df, pd.DataFrame) and len(cv_df) > 0:
        # Scatter: predicted vs observed
        row = _write_section(ws, row, 1, "Predicted vs Observed (Chart Data)")
        row += 1
        ws.cell(row=row, column=1, value="Train Prediction")
        ws.cell(row=row, column=2, value="Test Observed")
        _style_header_row(ws, row, 2)
        chart_h = row
        row += 1

        subset = cv_df.head(500)
        for _, r in subset.iterrows():
            ws.cell(row=row, column=1, value=round(_safe_float(r.get("train_prediction")), 2)).font = _VALUE_FONT
            ws.cell(row=row, column=2, value=round(_safe_float(r.get("test_observed")), 2)).font = _VALUE_FONT
            row += 1
        data_end = row - 1

        chart = _make_scatter_chart(
            f"Cross-Validation: Predicted vs Observed (r={cv_corr:.3f})" if cv_corr else "Cross-Validation",
            "Train Prediction (us)", "Test Observed (us)"
        )
        xv = Reference(ws, min_col=1, min_row=chart_h + 1, max_row=data_end)
        yv = Reference(ws, min_col=2, min_row=chart_h + 1, max_row=data_end)
        series = Series(yv, xv, title="CV Points")
        chart.series.append(series)
        row += 1
        ws.add_chart(chart, f"A{row}")
        row += 16

        # Full CV data
        row += 1
        row = _write_section(ws, row, 1, "Full Cross-Validation Data")
        row += 1
        row = _write_dataframe(ws, cv_df, row, include_index=False)
    else:
        row = _not_available_message(ws, row)

    _auto_column_widths(ws)


def _build_figure_manifest(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 13: Figure_Manifest."""
    ws = wb.create_sheet("Figure_Manifest")
    row = _write_title(ws, 1, 1, "Figure Manifest")
    row += 1

    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")
    headers = ["Figure ID", "Description", "Output Path (PNG)", "Output Path (PDF)"]
    for ci, h in enumerate(headers, 1):
        ws.cell(row=row, column=ci, value=h)
    _style_header_row(ws, row, len(headers))
    row += 1

    manifest = [
        ("F1", "IRBS Drift Bias Demonstration", "F1_irbs_bias_demo"),
        ("F2", "Tail Explosion Under High-Risk Spectator", "F2_phenomenon_ladder"),
        ("F3", "Interference Tomography Heatmap with CI", "F3_tomography_heatmap"),
        ("F4", "Sparse Interferer Recovery vs Sampling", "F4_sparse_recovery"),
        ("F5", "Naive Model Mismatch", "F5_baseline_mismatch"),
        ("F6", "Scheduler Comparison Across Regimes", "F6_scheduler_comparison"),
        ("F7", "Worst-Case Blowup Avoidance", "F7_worst_case"),
        ("F8", "Ablation Study", "F8_ablations"),
        ("F9", "QA Check Summary", "F9_qa_summary"),
        ("F10", "Channel Decomposition", "F10_channel_decomposition"),
        ("F11", "Sensitivity Analysis", "F11_sensitivity"),
    ]

    for fig_id, desc, basename in manifest:
        png_path = f"{figures_dir}/{basename}.png"
        pdf_path = f"{figures_dir}/{basename}.pdf"
        exists = " [exists]" if Path(png_path).exists() else ""
        ws.cell(row=row, column=1, value=fig_id).font = _LABEL_FONT
        ws.cell(row=row, column=2, value=desc).font = _VALUE_FONT
        ws.cell(row=row, column=3, value=png_path + exists).font = _VALUE_FONT
        ws.cell(row=row, column=4, value=pdf_path).font = _VALUE_FONT
        row += 1

    _auto_column_widths(ws)


def _build_how_to_recompute(wb: openpyxl.Workbook, config: Dict, all_results: Dict):
    """Sheet 14: How_to_Recompute."""
    ws = wb.create_sheet("How_to_Recompute")
    row = _write_title(ws, 1, 1, "How to Recompute These Results")
    row += 2

    row = _write_label_value(ws, row, "Quick run (~2-5 min):", "")
    cmd = ws.cell(row=row, column=1,
                  value="python -m sit.experiments.run_all --config sit/experiments/configs/quick.yaml")
    cmd.font = Font(name="Consolas", size=10)
    row += 2
    row = _write_label_value(ws, row, "Full run (~12 min):", "")
    cmd = ws.cell(row=row, column=1,
                  value="python -m sit.experiments.run_all --config sit/experiments/configs/full.yaml")
    cmd.font = Font(name="Consolas", size=10)
    row += 2

    row = _write_section(ws, row, 1, "Notes")
    row += 1
    notes = [
        "Both commands are deterministic (seeded RNG). Identical results across runs.",
        "Quick mode uses a reduced grid for fast pipeline validation.",
        "Full mode runs the complete factorial grid and all analyses.",
        "The workbook, figures, and tables are regenerated on each run.",
        "Output directories are created automatically.",
        "Requirements: numpy, pandas, openpyxl, matplotlib, pyyaml.",
        "All charts in this workbook are native Excel charts — fully editable.",
    ]
    for note in notes:
        ws.cell(row=row, column=1, value=note).font = _VALUE_FONT
        row += 1

    _auto_column_widths(ws)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_workbook(config: Dict, all_results: Dict) -> str:
    """Create the SIT_Results.xlsx workbook with editable charts.

    Returns the absolute path to the saved workbook file.
    """
    workbook_dir = config.get("output", {}).get("workbook_dir", "results/workbook")
    Path(workbook_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(workbook_dir) / "SIT_Results.xlsx")

    wb = openpyxl.Workbook()

    _build_overview(wb, config, all_results)
    _build_irbs_drift_bias(wb, config, all_results)
    _build_tomography(wb, config, all_results)
    _build_sparse_recovery(wb, config, all_results)
    _build_baseline_mismatch(wb, config, all_results)
    _build_scheduling(wb, config, all_results)
    _build_worst_case(wb, config, all_results)
    _build_ablations(wb, config, all_results)
    _build_qa_checks(wb, config, all_results)
    _build_channel_decomposition(wb, config, all_results)
    _build_sensitivity(wb, config, all_results)
    _build_cross_validation(wb, config, all_results)
    _build_figure_manifest(wb, config, all_results)
    _build_how_to_recompute(wb, config, all_results)

    wb.save(output_path)
    return output_path

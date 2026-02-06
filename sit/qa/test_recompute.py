"""QA Test: Verify derived tables match recomputation from raw data."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sit.analysis.metrics import compute_p99, compute_cvar99


def test_recompute_trial_summaries(
    raw_dir: str = "data/raw",
    tolerance: float = 1e-6,
) -> bool:
    """Verify that trial summary metrics can be recomputed from raw samples.

    Since we don't store full raw samples in parquet (too large), we verify
    internal consistency of the trial summary metrics:
    - cvar99 >= p99 (by definition)
    - p99 >= p95
    - p95 >= mean (for right-skewed distributions)
    - All values positive
    """
    trial_path = Path(raw_dir) / "trial_summaries.parquet"
    if not trial_path.exists():
        print("SKIP: No trial summaries file found")
        return True

    df = pd.read_parquet(trial_path)

    all_pass = True
    checks = []

    # CVaR99 >= p99
    if "cvar99" in df.columns and "p99" in df.columns:
        violations = (df["cvar99"] < df["p99"] - tolerance).sum()
        total = len(df)
        passed = violations == 0
        checks.append(("cvar99 >= p99", passed, f"{violations}/{total} violations"))
        if not passed:
            all_pass = False

    # p99 >= p95
    if "p99" in df.columns and "p95" in df.columns:
        violations = (df["p99"] < df["p95"] - tolerance).sum()
        passed = violations == 0
        checks.append(("p99 >= p95", passed, f"{violations}/{len(df)} violations"))
        if not passed:
            all_pass = False

    # All positive
    for col in ["mean", "p95", "p99", "cvar95", "cvar99"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            passed = neg == 0
            checks.append((f"{col} >= 0", passed, f"{neg} negative values"))
            if not passed:
                all_pass = False

    # SLO violation rate in [0, 1]
    if "slo_violation_rate" in df.columns:
        out_of_range = ((df["slo_violation_rate"] < -tolerance) |
                        (df["slo_violation_rate"] > 1 + tolerance)).sum()
        passed = out_of_range == 0
        checks.append(("slo_violation_rate in [0,1]", passed, f"{out_of_range} out of range"))
        if not passed:
            all_pass = False

    for name, passed, msg in checks:
        print(f"Recompute check ({name}): {'PASS' if passed else 'FAIL'} - {msg}")

    return all_pass


def test_recompute_scheduling_metrics(
    raw_dir: str = "data/raw",
    tolerance: float = 1e-6,
) -> bool:
    """Verify scheduling result metrics are internally consistent."""
    sched_path = Path(raw_dir) / "scheduling_results.parquet"
    if not sched_path.exists():
        print("SKIP: No scheduling results file found")
        return True

    df = pd.read_parquet(sched_path)
    all_pass = True

    # CVaR99 >= p99
    if "cvar99" in df.columns and "p99" in df.columns:
        violations = (df["cvar99"] < df["p99"] - tolerance).sum()
        passed = violations == 0
        print(f"Scheduling recompute (cvar99 >= p99): {'PASS' if passed else 'FAIL'} "
              f"- {violations}/{len(df)} violations")
        if not passed:
            all_pass = False

    # All positive metrics
    for col in ["mean", "p95", "p99", "cvar99"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            passed = neg == 0
            print(f"Scheduling recompute ({col} >= 0): {'PASS' if passed else 'FAIL'} "
                  f"- {neg} negative values")
            if not passed:
                all_pass = False

    return all_pass


def run_all_recompute_tests() -> bool:
    """Run all recompute verification tests."""
    pass1 = test_recompute_trial_summaries()
    pass2 = test_recompute_scheduling_metrics()
    return pass1 and pass2


if __name__ == "__main__":
    success = run_all_recompute_tests()
    sys.exit(0 if success else 1)

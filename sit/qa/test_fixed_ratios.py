"""QA Test: Detect suspicious fixed-ratio artifacts.

If CVaR or p99 is an exact fixed multiple of mean across many rows,
this indicates a bug (e.g., CVaR computed as 1.5 * p99 rather than from samples).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sit.simulator.validation import check_no_fixed_ratio


def test_fixed_ratios_from_file(filepath: str, min_rows: int = 20, tolerance: float = 0.001) -> bool:
    """Test a results file for fixed-ratio artifacts.

    Args:
        filepath: path to CSV or parquet with columns mean, p99, cvar99
        min_rows: minimum rows required for test
        tolerance: CV threshold below which ratios are suspiciously constant

    Returns:
        True if test passes (no fixed ratios detected).
    """
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    required = {"mean", "p99", "cvar99"}
    if not required.issubset(df.columns):
        print(f"SKIP: {filepath} missing columns {required - set(df.columns)}")
        return True

    passed, msg = check_no_fixed_ratio(
        df["mean"].values,
        df["p99"].values,
        df["cvar99"].values,
        tolerance=tolerance,
        min_rows=min_rows,
    )

    print(f"Fixed ratio test on {filepath}: {'PASS' if passed else 'FAIL'} - {msg}")
    return passed


def test_fixed_ratios_from_dataframe(df: pd.DataFrame, name: str = "data") -> bool:
    """Test a DataFrame for fixed-ratio artifacts."""
    required = {"mean", "p99", "cvar99"}
    if not required.issubset(df.columns):
        print(f"SKIP: {name} missing columns {required - set(df.columns)}")
        return True

    passed, msg = check_no_fixed_ratio(
        df["mean"].values,
        df["p99"].values,
        df["cvar99"].values,
    )

    print(f"Fixed ratio test on {name}: {'PASS' if passed else 'FAIL'} - {msg}")
    return passed


def run_all_fixed_ratio_tests() -> bool:
    """Run fixed-ratio tests on all available result files."""
    all_pass = True
    data_dir = Path("data/raw")
    derived_dir = Path("data/derived")

    # Check scheduling results
    sched_path = data_dir / "scheduling_results.parquet"
    if sched_path.exists():
        if not test_fixed_ratios_from_file(str(sched_path)):
            all_pass = False

    # Check trial summaries
    trial_path = data_dir / "trial_summaries.parquet"
    if trial_path.exists():
        if not test_fixed_ratios_from_file(str(trial_path)):
            all_pass = False

    return all_pass


if __name__ == "__main__":
    success = run_all_fixed_ratio_tests()
    sys.exit(0 if success else 1)

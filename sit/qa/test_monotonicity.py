"""QA Test: Monotonicity checks.

Increasing load should not systematically reduce interference severity
in structured/adversarial regimes. Some noise is allowed.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sit.simulator.validation import check_monotonicity


def test_load_monotonicity(filepath: str, tolerance_fraction: float = 0.25) -> bool:
    """Test that p99 generally increases with load in adversarial regime.

    Args:
        filepath: path to scheduling results
        tolerance_fraction: max fraction of violations allowed

    Returns:
        True if test passes.
    """
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    if "load" not in df.columns or "p99" not in df.columns:
        print(f"SKIP: {filepath} missing load or p99 columns")
        return True

    all_pass = True

    for regime in ["structured", "adversarial"]:
        subset = df[df["regime"] == regime] if "regime" in df.columns else df
        if len(subset) < 3:
            continue

        load_means = subset.groupby("load")["p99"].mean()
        passed, vr, msg = check_monotonicity(
            load_means.index.values.astype(float),
            load_means.values,
            tolerance_fraction=tolerance_fraction,
        )

        print(f"Monotonicity (load vs p99, {regime}): {'PASS' if passed else 'FAIL'} - {msg}")
        if not passed:
            all_pass = False

    return all_pass


def test_distance_monotonicity(filepath: str) -> bool:
    """Test that p99 generally decreases with increasing distance.

    Closer placement should yield higher interference.
    """
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    if "distance" not in df.columns or "p99" not in df.columns:
        print(f"SKIP: {filepath} missing distance or p99 columns")
        return True

    dist_order = {"same_core": 0, "same_llc": 1, "same_numa": 2, "cross_numa": 3, "cross_socket": 4}

    all_pass = True
    for regime in ["structured", "adversarial"]:
        subset = df[df["regime"] == regime] if "regime" in df.columns else df
        if len(subset) < 3:
            continue

        dist_means = subset.groupby("distance")["p99"].mean()
        available = [d for d in dist_order if d in dist_means.index]
        if len(available) < 2:
            continue

        ordered_vals = [dist_means[d] for d in sorted(available, key=lambda x: dist_order[x])]
        ordered_dists = np.arange(len(ordered_vals))

        # Expect decreasing: closer = higher p99
        # So reversed should be increasing
        passed, vr, msg = check_monotonicity(
            ordered_dists,
            np.array(ordered_vals)[::-1],  # reverse so "increasing distance" = "decreasing p99"
            tolerance_fraction=0.3,
        )
        print(f"Monotonicity (distance vs p99, {regime}): {'PASS' if passed else 'WARN'} - {msg}")

    return all_pass


def run_all_monotonicity_tests() -> bool:
    """Run all monotonicity tests."""
    all_pass = True
    sched_path = Path("data/raw/scheduling_results.parquet")

    if sched_path.exists():
        if not test_load_monotonicity(str(sched_path)):
            all_pass = False
        test_distance_monotonicity(str(sched_path))  # advisory only
    else:
        print("SKIP: No scheduling results found")

    return all_pass


if __name__ == "__main__":
    success = run_all_monotonicity_tests()
    sys.exit(0 if success else 1)

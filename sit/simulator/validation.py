"""Validation utilities for simulator outputs."""

import numpy as np
from typing import List, Tuple


def check_latency_positive(samples: np.ndarray) -> bool:
    """All latency samples must be positive."""
    return bool(np.all(samples > 0))


def check_no_fixed_ratio(
    means: np.ndarray,
    p99s: np.ndarray,
    cvars: np.ndarray,
    tolerance: float = 0.001,
    min_rows: int = 20,
) -> Tuple[bool, str]:
    """Check that p99/mean and cvar/mean ratios are NOT suspiciously constant.

    Returns (passed, message).
    """
    if len(means) < min_rows:
        return True, "Not enough rows to check"

    mask = means > 0
    if np.sum(mask) < min_rows:
        return True, "Not enough positive means"

    ratio_p99 = p99s[mask] / means[mask]
    ratio_cvar = cvars[mask] / means[mask]

    cv_p99 = np.std(ratio_p99) / np.mean(ratio_p99) if np.mean(ratio_p99) > 0 else 0
    cv_cvar = np.std(ratio_cvar) / np.mean(ratio_cvar) if np.mean(ratio_cvar) > 0 else 0

    if cv_p99 < tolerance:
        return False, f"FIXED RATIO: p99/mean CV={cv_p99:.6f} < {tolerance}"
    if cv_cvar < tolerance:
        return False, f"FIXED RATIO: cvar/mean CV={cv_cvar:.6f} < {tolerance}"

    return True, f"OK: p99/mean CV={cv_p99:.4f}, cvar/mean CV={cv_cvar:.4f}"


def check_monotonicity(
    loads: np.ndarray,
    severities: np.ndarray,
    tolerance_fraction: float = 0.2,
) -> Tuple[bool, float, str]:
    """Check that severity generally increases with load.

    Allows some noise but flags systematic inversions.

    Returns (passed, violation_rate, message).
    """
    if len(loads) < 3:
        return True, 0.0, "Not enough points"

    # Sort by load
    order = np.argsort(loads)
    sorted_sev = severities[order]

    # Count violations: pairs where higher load has lower severity
    violations = 0
    total_pairs = 0
    for i in range(len(sorted_sev) - 1):
        total_pairs += 1
        if sorted_sev[i + 1] < sorted_sev[i] - 1e-10:
            violations += 1

    violation_rate = violations / max(total_pairs, 1)
    passed = violation_rate <= tolerance_fraction
    msg = f"Monotonicity: {violations}/{total_pairs} violations ({violation_rate:.2%})"

    return passed, violation_rate, msg

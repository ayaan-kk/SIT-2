"""QA Test: Verify seed reproducibility.

Same seed must yield identical outputs for the simulator.
"""

import numpy as np
import sys

from sit.simulator.workloads import get_targets, get_spectators
from sit.simulator.device_profiles import get_device_profiles
from sit.simulator.latency_generator import generate_trial_samples


def test_seed_reproducibility(n_checks: int = 5) -> bool:
    """Verify that identical seeds produce identical latency samples.

    Runs the simulator twice with the same seed for multiple conditions
    and verifies exact match.
    """
    targets = get_targets()
    spectators = get_spectators()
    devices = get_device_profiles()

    target = list(targets.values())[0]
    spectator = list(spectators.values())[0]
    device = list(devices.values())[0]

    all_pass = True
    conditions = [
        (0.3, "same_core", "benign"),
        (0.5, "same_numa", "structured"),
        (0.8, "cross_socket", "adversarial"),
        (0.1, "same_llc", "benign"),
        (0.9, "same_core", "adversarial"),
    ]

    for i, (load, dist, regime) in enumerate(conditions[:n_checks]):
        seed = 12345 + i * 1000

        # Run 1
        samples1 = generate_trial_samples(
            target, device, load, dist, regime, 200,
            np.random.default_rng(seed),
            spectator=spectator,
        )

        # Run 2
        samples2 = generate_trial_samples(
            target, device, load, dist, regime, 200,
            np.random.default_rng(seed),
            spectator=spectator,
        )

        match = np.allclose(samples1, samples2, rtol=0, atol=0)
        print(f"Seed repro (seed={seed}, load={load}, dist={dist}, regime={regime}): "
              f"{'PASS' if match else 'FAIL'}")

        if not match:
            all_pass = False
            # Show first difference
            diff_idx = np.where(~np.isclose(samples1, samples2, rtol=0, atol=0))[0]
            if len(diff_idx) > 0:
                idx = diff_idx[0]
                print(f"  First diff at index {idx}: {samples1[idx]} vs {samples2[idx]}")

    # Also test control (no spectator)
    seed = 99999
    s1 = generate_trial_samples(
        target, device, 0.5, "same_numa", "structured", 100,
        np.random.default_rng(seed),
    )
    s2 = generate_trial_samples(
        target, device, 0.5, "same_numa", "structured", 100,
        np.random.default_rng(seed),
    )
    match = np.allclose(s1, s2, rtol=0, atol=0)
    print(f"Seed repro (control, no spectator): {'PASS' if match else 'FAIL'}")
    if not match:
        all_pass = False

    return all_pass


def run_all_seed_tests() -> bool:
    """Run all seed reproducibility tests."""
    return test_seed_reproducibility()


if __name__ == "__main__":
    success = run_all_seed_tests()
    sys.exit(0 if success else 1)

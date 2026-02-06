"""Drift model for SIT simulator.

Drift represents time-varying multiplicative factors on latency:
  Y_t = mu + tau * Z_t + g(t) + eps_t

g(t) consists of:
- Slow thermal ramp component
- DVFS-like piecewise changes
- Stochastic background daemon bursts
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .device_profiles import DeviceProfile


@dataclass
class DriftState:
    """Mutable state for drift simulation across trials."""
    thermal_level: float = 1.0
    dvfs_offset: float = 0.0
    trial_index: int = 0


def compute_drift_multiplier(
    trial_index: int,
    total_trials: int,
    device: DeviceProfile,
    rng: np.random.Generator,
    drift_state: Optional[DriftState] = None,
) -> float:
    """Compute the drift multiplier g(t) for a given trial.

    The drift multiplier is multiplicative on latency:
      latency *= (1 + g(t))

    Components:
    1. Thermal ramp: slow monotonic increase over experiment window
    2. DVFS steps: random discrete jumps in frequency/voltage state
    3. Background daemon bursts: random bursty events

    Args:
        trial_index: current trial number (0-indexed)
        total_trials: total number of trials in experiment
        device: device profile with drift parameters
        rng: numpy random generator
        drift_state: mutable state tracking drift accumulation

    Returns:
        Multiplicative drift factor (1.0 = no drift).
    """
    if drift_state is None:
        drift_state = DriftState()

    # Fraction of experiment elapsed
    t_frac = trial_index / max(total_trials - 1, 1)

    # 1. Thermal ramp: gradual increase, bounded by ceiling
    thermal_component = 1.0 + (device.thermal_ceiling - 1.0) * (
        1.0 - np.exp(-device.thermal_ramp_rate * trial_index * 10)
    )
    thermal_component = min(thermal_component, device.thermal_ceiling)

    # 2. DVFS step changes: random discrete jumps
    if rng.random() < device.dvfs_step_prob:
        # Step up or down with slight upward bias
        step = rng.normal(0.3, 1.0) * device.dvfs_step_magnitude
        drift_state.dvfs_offset += step
    # Clamp DVFS offset
    drift_state.dvfs_offset = np.clip(drift_state.dvfs_offset, -0.15, 0.3)
    dvfs_component = 1.0 + drift_state.dvfs_offset

    # 3. Background daemon bursts: random events that temporarily spike latency
    if rng.random() < device.background_burst_prob:
        burst_component = device.background_burst_severity
    else:
        burst_component = 1.0

    # Combine multiplicatively
    drift_multiplier = thermal_component * dvfs_component * burst_component

    drift_state.trial_index = trial_index

    return drift_multiplier


def compute_drift_series(
    total_trials: int,
    device: DeviceProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute drift multiplier for every trial in a sequence.

    Returns:
        Array of shape (total_trials,) with drift multipliers.
    """
    drift_state = DriftState()
    multipliers = np.empty(total_trials)
    for i in range(total_trials):
        multipliers[i] = compute_drift_multiplier(
            i, total_trials, device, rng, drift_state
        )
    return multipliers


def inject_synthetic_drift(
    total_trials: int,
    ramp_rate: float = 0.005,
    step_at_fraction: float = 0.5,
    step_magnitude: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a synthetic drift pattern for bias demonstration.

    This creates a known drift function for the IRBS bias demo:
    - Linear ramp
    - Discrete step at step_at_fraction
    - Small random noise

    Args:
        total_trials: number of trials
        ramp_rate: slope of linear ramp per trial
        step_at_fraction: where the step occurs (0-1)
        step_magnitude: size of the step
        rng: random generator for noise

    Returns:
        Array of drift multipliers.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = np.arange(total_trials, dtype=np.float64)
    t_frac = t / max(total_trials - 1, 1)

    # Linear ramp
    ramp = 1.0 + ramp_rate * t

    # Step function
    step = np.where(t_frac >= step_at_fraction, step_magnitude, 0.0)

    # Small noise
    noise = rng.normal(0, 0.01, size=total_trials)

    return ramp + step + noise

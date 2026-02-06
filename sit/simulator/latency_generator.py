"""Latency generator for SIT simulator.

Generates raw latency samples for a given condition, incorporating:
- Baseline distribution (lognormal)
- Drift (time-varying multiplicative factor)
- Structured interference (channel-mediated slowdown)
- Tail spikes (heavy-tailed events driven by interference structure)

Primary entry point: generate_trial_samples()
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .workloads import Workload
from .device_profiles import DeviceProfile
from .drift import compute_drift_multiplier, DriftState
from .interference_channels import (
    compute_interference_severity,
    DISTANCES,
)


def generate_baseline_latencies(
    n_samples: int,
    workload: Workload,
    device: DeviceProfile,
    load: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate baseline latency samples (no interference).

    Uses lognormal distribution scaled by device and load.
    """
    base_mu = np.log(workload.base_latency_us * device.baseline_latency_scale)
    # Load increases both mean and variance
    load_mu_shift = 0.3 * load ** 1.5
    load_sigma_shift = 0.1 * load

    mu = base_mu + load_mu_shift
    sigma = workload.latency_shape + load_sigma_shift

    samples = rng.lognormal(mu, sigma, size=n_samples)

    # Workload burstiness: some samples get burst multiplier
    burst_mask = rng.random(n_samples) < workload.burstiness * load
    samples[burst_mask] *= workload.burst_multiplier

    return samples


def generate_spike_latencies(
    n_spikes: int,
    base_latency: float,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate heavy-tailed spike latencies.

    Spikes are drawn from a Pareto distribution to create structured tail events.
    The severity of spikes depends on interference structure.
    """
    # Pareto shape: lower alpha = heavier tail
    # severity modulates both the scale and tail heaviness
    alpha = max(1.1, 3.0 - severity * 0.8)  # heavier tail with more severity
    scale = base_latency * (2.0 + severity * 3.0)

    spikes = (rng.pareto(alpha, size=n_spikes) + 1.0) * scale
    return spikes


def generate_trial_samples(
    target: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_samples: int,
    rng: np.random.Generator,
    spectator: Optional[Workload] = None,
    drift_multiplier: float = 1.0,
    apply_static_partition: bool = False,
    triton_batching: bool = False,
) -> np.ndarray:
    """Generate latency samples for a single trial.

    Args:
        target: target workload
        device: device profile
        load: load level (0-1)
        distance: placement distance key
        regime: "benign", "structured", or "adversarial"
        n_samples: number of latency samples per trial
        rng: numpy random generator
        spectator: spectator workload (None for control trial)
        drift_multiplier: drift factor from drift model
        apply_static_partition: simulate cache/resource partitioning
        triton_batching: simulate Triton-style batching effects

    Returns:
        Array of latency samples in microseconds.
    """
    # 1. Generate baseline latencies
    samples = generate_baseline_latencies(n_samples, target, device, load, rng)

    # 2. Apply drift
    samples *= drift_multiplier

    # 3. Apply interference if spectator present
    if spectator is not None:
        severity, per_channel, spike_prob = compute_interference_severity(
            target, spectator, device, distance, load, regime
        )

        # Static partition reduces interference but adds overhead
        if apply_static_partition:
            severity *= 0.3  # partitioning reduces interference
            spike_prob *= 0.4
            samples *= 1.05  # but adds partitioning overhead

        # Multiplicative slowdown from interference
        # Not uniform: each sample gets slightly different interference
        interference_noise = rng.normal(1.0, 0.1 * severity, size=n_samples)
        interference_factor = 1.0 + severity * np.maximum(interference_noise, 0.5)
        samples *= interference_factor

        # 4. Structured tail spikes
        # Each sample independently may become a spike
        spike_mask = rng.random(n_samples) < spike_prob
        n_spikes = int(np.sum(spike_mask))
        if n_spikes > 0:
            median_latency = float(np.median(samples))
            spike_values = generate_spike_latencies(
                n_spikes, median_latency, severity, rng
            )
            # Spikes replace the original values (they are catastrophic events)
            samples[spike_mask] = np.maximum(samples[spike_mask], spike_values)

    # 5. Triton batching effects (application-level)
    if triton_batching:
        # Batching reduces mean via amortization but can increase tail
        # due to head-of-line blocking under interference
        batch_reduction = 0.85  # 15% mean reduction from batching
        samples *= batch_reduction
        # But queuing delay adds tail: some requests wait for batch fill
        queue_delay_mask = rng.random(n_samples) < 0.1 * load
        queue_delays = rng.exponential(
            target.base_latency_us * device.baseline_latency_scale * 0.5 * load,
            size=int(np.sum(queue_delay_mask)),
        )
        samples[queue_delay_mask] += queue_delays

    # Ensure all latencies are positive
    samples = np.maximum(samples, 1.0)

    return samples


def generate_control_treatment_pair(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    load: float,
    distance: str,
    regime: str,
    n_samples: int,
    rng: np.random.Generator,
    drift_multiplier: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate matched control and treatment trial samples.

    Returns:
        (control_samples, treatment_samples)
    """
    # Use child RNGs to ensure independence
    ctrl_rng = np.random.default_rng(rng.integers(0, 2**63))
    treat_rng = np.random.default_rng(rng.integers(0, 2**63))

    control = generate_trial_samples(
        target=target,
        device=device,
        load=load,
        distance=distance,
        regime=regime,
        n_samples=n_samples,
        rng=ctrl_rng,
        spectator=None,
        drift_multiplier=drift_multiplier,
    )

    treatment = generate_trial_samples(
        target=target,
        device=device,
        load=load,
        distance=distance,
        regime=regime,
        n_samples=n_samples,
        rng=treat_rng,
        spectator=spectator,
        drift_multiplier=drift_multiplier,
    )

    return control, treatment

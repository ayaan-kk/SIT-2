"""Latency generator for SIT simulator.

Generates raw latency samples for a given condition, incorporating:
- Baseline distribution (lognormal)
- Drift (time-varying multiplicative factor)
- Structured interference (channel-mediated slowdown)
- Tail spikes (heavy-tailed events driven by interference structure)
- Temporal autocorrelation within trial (AR(1) cache warming/cooling)
- Per-channel static partition effectiveness model
- Triton batching with load-dependent amortization and HOL blocking

Primary entry point: generate_trial_samples()
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .workloads import Workload, CH_IDX, NUM_CHANNELS
from .device_profiles import DeviceProfile
from .drift import compute_drift_multiplier, DriftState
from .interference_channels import (
    compute_interference_severity,
    DISTANCES,
    DISTANCE_ATTENUATION,
)

# ---------------------------------------------------------------------------
# Static partition effectiveness per channel
# ---------------------------------------------------------------------------
# Some channels can be effectively partitioned (hardware mechanisms exist);
# others are inherently shared or global and cannot be isolated.
# Value of 1.0 = perfectly partitionable; 0.0 = not partitionable at all.
PARTITION_EFFECTIVENESS = np.array([
    0.85,  # LLC: Intel CAT / AMD L3 partitioning -- highly effective
    0.15,  # MEM_BW: memory bandwidth is a shared fabric, barely partitionable
    0.40,  # TLB: can help via page coloring, partial effect
    0.80,  # PREFETCH: tied to LLC partitioning, largely effective
    0.30,  # NUMA: topology-based, partition has limited effect
    0.05,  # THERMAL: heat is physics, not partitionable
    0.05,  # OS_FAULTS: OS-wide scheduling, not partitionable
])

# Throughput penalty range for static partition (uniform draw per trial)
PARTITION_THROUGHPUT_PENALTY_MIN = 0.15  # 15% throughput loss (minimum)
PARTITION_THROUGHPUT_PENALTY_MAX = 0.25  # 25% throughput loss (maximum)

# ---------------------------------------------------------------------------
# Spike channel characterisation
# ---------------------------------------------------------------------------
# Per-channel Pareto alpha floor (lower = heavier tail).
# Channels that are fundamentally bounded (e.g., OS scheduling) get higher
# alpha (lighter tail).  Hardware saturation channels get lower alpha.
CHANNEL_SPIKE_ALPHA = np.array([
    1.15,  # LLC: heavy tail (cache miss cascades)
    1.20,  # MEM_BW: heavy tail (queue build-up)
    1.50,  # TLB: moderate tail
    1.30,  # PREFETCH: moderate-heavy tail
    1.40,  # NUMA: moderate tail (bounded by topology)
    1.80,  # THERMAL: lighter tail (throttling is bounded)
    2.00,  # OS_FAULTS: lightest tail (scheduling delays are bounded)
])

# ---------------------------------------------------------------------------
# Temporal autocorrelation parameters
# ---------------------------------------------------------------------------
AR1_RHO_MIN = 0.10   # minimum autocorrelation coefficient
AR1_RHO_MAX = 0.30   # maximum autocorrelation coefficient


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
    per_channel_severity: Optional[np.ndarray] = None,
    distance: Optional[str] = None,
) -> np.ndarray:
    """Generate heavy-tailed spike latencies.

    Spikes are drawn from a Pareto distribution to create structured tail
    events.  The Pareto shape (alpha) depends on *which* channels dominate
    the interference, not a single generic severity number:

    - LLC + MEM_BW saturated at close distance  =>  heavy Pareto (alpha near 1.1)
    - OS_FAULTS dominant                         =>  moderate Pareto (alpha near 2.0)
    - Mixed / moderate                           =>  intermediate

    Args:
        n_spikes: number of spike values to generate
        base_latency: reference latency (e.g. median) for scaling
        severity: scalar severity (backward-compatible path)
        rng: numpy Generator
        per_channel_severity: (optional) per-channel severity vector for
            structured alpha selection
        distance: (optional) placement distance key for proximity effect
    """
    if n_spikes == 0:
        return np.empty(0)

    # --- Determine Pareto alpha from channel structure ---
    if per_channel_severity is not None and np.sum(per_channel_severity) > 1e-10:
        # Weight the channel-specific alpha floors by each channel's share of
        # total severity to get a composite alpha.
        weights = per_channel_severity / np.sum(per_channel_severity)
        alpha_base = float(np.dot(weights, CHANNEL_SPIKE_ALPHA))

        # Close placement with LLC+MEM_BW dominance pushes alpha even lower
        if distance is not None and distance in ("same_core", "same_llc"):
            llc_membw_share = weights[CH_IDX["LLC"]] + weights[CH_IDX["MEM_BW"]]
            # Up to 0.15 extra tail heaviness from proximity + cache/BW saturation
            alpha_base -= 0.15 * llc_membw_share

        alpha = max(1.05, alpha_base - severity * 0.3)
    else:
        # Backward-compatible fallback: generic severity-driven alpha
        alpha = max(1.1, 3.0 - severity * 0.8)

    scale = base_latency * (2.0 + severity * 3.0)

    spikes = (rng.pareto(alpha, size=n_spikes) + 1.0) * scale
    return spikes


def _apply_temporal_autocorrelation(
    samples: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply AR(1) temporal autocorrelation to model cache warming/cooling.

    Within a single trial, consecutive samples are mildly correlated:
      x[t] = rho * x[t-1] + sqrt(1 - rho^2) * eps[t]

    This is applied as a multiplicative modulation so the marginal
    distribution is preserved (mean and variance stay the same).
    """
    n = len(samples)
    if n < 2:
        return samples

    # Draw rho uniformly from [AR1_RHO_MIN, AR1_RHO_MAX]
    rho = rng.uniform(AR1_RHO_MIN, AR1_RHO_MAX)

    # Generate AR(1) process in log-space so multiplicative modulation
    # corresponds to additive AR(1) on log-latency.
    log_samples = np.log(np.maximum(samples, 1e-6))
    mean_log = np.mean(log_samples)
    centered = log_samples - mean_log

    # Apply AR(1) filter: propagate autocorrelation through centered series
    ar_series = np.empty(n)
    ar_series[0] = centered[0]
    innovation_scale = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        ar_series[t] = rho * ar_series[t - 1] + innovation_scale * centered[t]

    # Rescale to preserve original variance
    orig_std = np.std(centered)
    ar_std = np.std(ar_series)
    if ar_std > 1e-10 and orig_std > 1e-10:
        ar_series *= (orig_std / ar_std)

    # Convert back
    return np.exp(ar_series + mean_log)


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

        # ------------------------------------------------------------------
        # Static partition: per-channel effectiveness model
        # ------------------------------------------------------------------
        if apply_static_partition:
            # Reduce severity per channel according to partition effectiveness.
            # Channels that are partitionable (LLC, PREFETCH) see large
            # reductions; non-partitionable channels (MEM_BW, THERMAL,
            # OS_FAULTS) retain most of their interference.
            partition_reduction = 1.0 - PARTITION_EFFECTIVENESS  # residual fraction
            per_channel_partitioned = per_channel * partition_reduction
            severity = float(np.sum(per_channel_partitioned))

            # Spike probability is also reduced, but only proportional to
            # the partitionable fraction of the dominant spike channels.
            avg_spike_ch_effectiveness = float(
                (PARTITION_EFFECTIVENESS[CH_IDX["LLC"]]
                 + PARTITION_EFFECTIVENESS[CH_IDX["MEM_BW"]]) / 2.0
            )
            spike_prob *= (1.0 - avg_spike_ch_effectiveness)

            # Throughput penalty: partitioning reduces maximum achievable
            # throughput by 15-25%, modeled as increased baseline latency
            # under load.  The penalty intensifies with load because the
            # reduced partition cannot absorb bursts.
            throughput_penalty = rng.uniform(
                PARTITION_THROUGHPUT_PENALTY_MIN,
                PARTITION_THROUGHPUT_PENALTY_MAX,
            )
            # Penalty is stronger at high load (quadratic blend)
            effective_penalty = 1.0 + throughput_penalty * (0.3 + 0.7 * load ** 2)
            samples *= effective_penalty

            # Use partitioned severity for downstream interference calc
            per_channel = per_channel_partitioned

        # Multiplicative slowdown from interference
        # Not uniform: each sample gets slightly different interference
        interference_noise = rng.normal(1.0, 0.1 * severity, size=n_samples)
        interference_factor = 1.0 + severity * np.maximum(interference_noise, 0.5)
        samples *= interference_factor

        # ------------------------------------------------------------------
        # 4. Structured tail spikes (channel-aware)
        # ------------------------------------------------------------------
        # Each sample independently may become a spike
        spike_mask = rng.random(n_samples) < spike_prob
        n_spikes = int(np.sum(spike_mask))
        if n_spikes > 0:
            median_latency = float(np.median(samples))
            spike_values = generate_spike_latencies(
                n_spikes,
                median_latency,
                severity,
                rng,
                per_channel_severity=per_channel,
                distance=distance,
            )
            # Spikes replace the original values (they are catastrophic events)
            samples[spike_mask] = np.maximum(samples[spike_mask], spike_values)

    # ------------------------------------------------------------------
    # 5. Triton batching effects (application-level, load-aware)
    # ------------------------------------------------------------------
    if triton_batching:
        # Amortization benefit: stronger at moderate load, weaker at extremes.
        # At very low load batches are small (less amortization); at moderate
        # load batches are full and efficient; at extreme load the benefit
        # remains but queuing penalties dominate.
        batch_efficiency = 0.75 + 0.25 * (1.0 - abs(load - 0.5) * 2.0)
        # Mean reduction: 15-25% depending on batch efficiency
        batch_reduction = 0.75 + 0.25 * (1.0 - batch_efficiency)
        samples *= batch_reduction

        # --- Head-of-line (HOL) blocking under high interference + load ---
        # When interference is present AND load is high, requests queue up
        # behind slow batch members.  This inflates the tail significantly:
        # the p99 can *increase* under Triton in adverse conditions.
        #
        # severity is computed above when spectator is not None; otherwise
        # there is no interference-driven HOL blocking.
        severity_for_hol = severity if spectator is not None else 0.0

        # HOL probability and magnitude scale with load and interference
        hol_probability = min(0.30, 0.05 * load + 0.12 * load * severity_for_hol)
        hol_mask = rng.random(n_samples) < hol_probability

        if int(np.sum(hol_mask)) > 0:
            # HOL delay: exponential with scale proportional to base latency,
            # load, and interference severity.  Under high interference this
            # can be several times the base latency.
            hol_scale = (
                target.base_latency_us
                * device.baseline_latency_scale
                * (0.3 + 0.7 * load)
                * (1.0 + 1.5 * severity_for_hol)
            )
            hol_delays = rng.exponential(hol_scale, size=int(np.sum(hol_mask)))
            samples[hol_mask] += hol_delays

    # ------------------------------------------------------------------
    # 6. Temporal autocorrelation (AR(1) within trial)
    # ------------------------------------------------------------------
    # Model cache warming/cooling effects: consecutive samples within a
    # trial are mildly autocorrelated (rho in [0.10, 0.30]).  This makes
    # within-trial p99 less noisy and more realistic.
    samples = _apply_temporal_autocorrelation(samples, rng)

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

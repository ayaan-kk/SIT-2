"""Interference channel model for SIT simulator.

Interference between target T and spectator S is mediated through explicit channels:
  channels = [LLC, MEM_BW, TLB, PREFETCH, NUMA, THERMAL, OS_FAULTS]

Interference severity depends on:
- Channel overlap (cosine similarity of pressure vectors)
- Per-channel saturation (combined pressure vs device capacity)
- Load level
- Placement distance (closer = worse)
- Regime (benign/structured/adversarial)
"""

import numpy as np
from typing import Tuple

from .workloads import Workload, NUM_CHANNELS
from .device_profiles import DeviceProfile

# Distance levels and their multipliers
DISTANCES = {
    "same_core": 0,
    "same_llc": 1,
    "same_numa": 2,
    "cross_numa": 3,
    "cross_socket": 4,
}

# Distance attenuation: closer = more interference
# These are NOT linear; same_core is drastically worse
DISTANCE_ATTENUATION = {
    "same_core": 1.0,
    "same_llc": 0.7,
    "same_numa": 0.4,
    "cross_numa": 0.2,
    "cross_socket": 0.1,
}

# Per-channel distance sensitivity (some channels are more distance-sensitive)
# LLC and PREFETCH are very distance-sensitive; THERMAL and OS less so
CHANNEL_DISTANCE_SENSITIVITY = np.array([
    0.9,   # LLC: very distance-sensitive
    0.7,   # MEM_BW: moderately distance-sensitive
    0.6,   # TLB: moderately distance-sensitive
    0.85,  # PREFETCH: very distance-sensitive
    0.95,  # NUMA: extremely distance-sensitive
    0.3,   # THERMAL: less distance-sensitive (heat spreads)
    0.2,   # OS_FAULTS: least distance-sensitive (OS-wide)
])

# Regime multipliers
REGIME_MULTIPLIERS = {
    "benign": 0.5,
    "structured": 1.0,
    "adversarial": 2.0,
}


def compute_channel_overlap(
    target: Workload,
    spectator: Workload,
) -> np.ndarray:
    """Compute per-channel overlap between target and spectator.

    Returns vector in R^C where each element is the product of pressures
    (representing resource contention in that channel).
    """
    return target.channel_pressure * spectator.channel_pressure


def compute_cosine_similarity(
    target: Workload,
    spectator: Workload,
) -> float:
    """Compute cosine similarity of resource vectors."""
    a = target.resource_vector
    b = spectator.resource_vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_interference_severity(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    distance: str,
    load: float,
    regime: str,
) -> Tuple[float, np.ndarray, float]:
    """Compute total interference severity and spike probability.

    Returns:
        (total_severity, per_channel_severity, spike_probability)

    total_severity: multiplicative slowdown factor on latency (>= 0)
    per_channel_severity: per-channel interference contributions
    spike_probability: probability of a tail spike event
    """
    # Per-channel overlap
    overlap = compute_channel_overlap(target, spectator)

    # Per-channel saturation: how much combined pressure exceeds capacity
    combined_pressure = (target.channel_pressure + spectator.channel_pressure * load)
    saturation = np.maximum(0, combined_pressure - device.channel_capacities)
    # Scale by overlap so interference only occurs where both workloads compete
    per_channel = overlap * saturation

    # Distance attenuation (per-channel, some channels more sensitive)
    dist_atten = DISTANCE_ATTENUATION[distance]
    # Per-channel: blend between global attenuation and channel-specific
    channel_atten = dist_atten ** CHANNEL_DISTANCE_SENSITIVITY
    per_channel *= channel_atten

    # Load scaling: superlinear beyond 0.5
    load_factor = load * (1.0 + load ** 2)

    # Regime multiplier
    regime_mult = REGIME_MULTIPLIERS[regime]

    # Total severity
    per_channel_severity = per_channel * load_factor * regime_mult
    total_severity = float(np.sum(per_channel_severity))

    # Spike probability: driven by high-overlap channels (LLC + MEM_BW especially)
    # This creates structured tail spikes, not random noise
    llc_membw_overlap = overlap[0] + overlap[1]  # LLC + MEM_BW
    spike_base = 0.01 + 0.15 * llc_membw_overlap
    spike_load = load ** 1.5
    spike_distance = dist_atten ** 0.5
    spike_regime = {"benign": 0.3, "structured": 1.0, "adversarial": 2.5}[regime]

    spike_probability = min(0.6, spike_base * spike_load * spike_distance * spike_regime)

    # Add channel noise contribution
    noise_contribution = float(np.sum(device.channel_noise * overlap))
    total_severity += noise_contribution

    return total_severity, per_channel_severity, spike_probability

"""Interference channel model for SIT simulator.

Interference between target T and spectator S is mediated through explicit channels:
  channels = [LLC, MEM_BW, TLB, PREFETCH, NUMA, THERMAL, OS_FAULTS]

Interference severity depends on:
- Channel overlap (cosine similarity of pressure vectors)
- Per-channel saturation (combined pressure vs device capacity)
- Nonlinear saturation: super-linear growth when combined pressure exceeds capacity
- Cross-channel interaction effects (synergy amplification)
- Load level
- Placement distance (closer = worse)
- Regime (benign/structured/adversarial)

Spike probability is driven by channel-specific threshold crossings, not
generic overlap, producing structured tails that naive models cannot predict.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .workloads import Workload, NUM_CHANNELS, CHANNELS, CH_IDX
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

# ---------------------------------------------------------------------------
# Nonlinear saturation parameters
# ---------------------------------------------------------------------------
# Steepness of the logistic saturation curve.  Higher k = sharper transition
# from linear to super-linear growth once combined pressure exceeds capacity.
SATURATION_STEEPNESS = 8.0      # k in logistic
SATURATION_AMPLITUDE = 1.5      # maximum extra multiplier at full saturation

# ---------------------------------------------------------------------------
# Channel interaction (synergy) matrix
# ---------------------------------------------------------------------------
# I[c1, c2] > 0 means the pair (c1, c2) amplifies each other's interference.
# The matrix is symmetric; only upper-triangle values are specified, then
# mirrored.  All other pairs default to zero (no synergy).
_INTERACTION_MATRIX = np.zeros((NUM_CHANNELS, NUM_CHANNELS), dtype=np.float64)

# LLC + MEM_BW: evicting cache lines forces memory traffic, compounding both
_INTERACTION_MATRIX[CH_IDX["LLC"], CH_IDX["MEM_BW"]] = 0.30
# TLB + PREFETCH: TLB misses stall the prefetcher, cascading into more misses
_INTERACTION_MATRIX[CH_IDX["TLB"], CH_IDX["PREFETCH"]] = 0.20
# LLC + PREFETCH: cache contention defeats prefetch benefit
_INTERACTION_MATRIX[CH_IDX["LLC"], CH_IDX["PREFETCH"]] = 0.15
# MEM_BW + NUMA: bandwidth saturation on remote node is especially bad
_INTERACTION_MATRIX[CH_IDX["MEM_BW"], CH_IDX["NUMA"]] = 0.18
# THERMAL + MEM_BW: thermal throttling worsens under sustained bandwidth load
_INTERACTION_MATRIX[CH_IDX["THERMAL"], CH_IDX["MEM_BW"]] = 0.10
# OS_FAULTS + TLB: page faults thrash the TLB
_INTERACTION_MATRIX[CH_IDX["OS_FAULTS"], CH_IDX["TLB"]] = 0.12

# Make symmetric
INTERACTION_MATRIX = _INTERACTION_MATRIX + _INTERACTION_MATRIX.T

# ---------------------------------------------------------------------------
# Channel-pair spike thresholds
# ---------------------------------------------------------------------------
# Each entry defines a pair of channels and the combined-pressure threshold
# above which spike probability receives a large boost.  This makes spikes
# *structurally* determined by specific resource conflicts rather than generic
# overlap, so naive smooth predictors systematically miss them.
#
# Format: (channel_a, channel_b, threshold, spike_boost)
#   spike_boost is added to spike probability when both channels' combined
#   pressure exceeds threshold.
SPIKE_TRIGGERS = [
    (CH_IDX["LLC"],       CH_IDX["MEM_BW"],    0.55, 0.18),  # cache + BW saturation
    (CH_IDX["LLC"],       CH_IDX["PREFETCH"],   0.50, 0.12),  # cache + prefetch saturation
    (CH_IDX["TLB"],       CH_IDX["PREFETCH"],   0.55, 0.10),  # TLB + prefetch cascade
    (CH_IDX["MEM_BW"],    CH_IDX["NUMA"],       0.60, 0.14),  # bandwidth on remote NUMA
    (CH_IDX["THERMAL"],   CH_IDX["MEM_BW"],     0.65, 0.08),  # thermal throttle + BW
    (CH_IDX["OS_FAULTS"], CH_IDX["TLB"],        0.50, 0.10),  # page faults + TLB thrash
]


# ===================================================================
# Core functions
# ===================================================================

def _logistic_saturation(combined: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Compute per-channel nonlinear saturation factor.

    When combined pressure exceeds capacity, interference grows super-linearly
    following a logistic (S-curve) response.  Returns a multiplier >= 1.0.

    saturation_factor = 1 + A / (1 + exp(-k * (combined - capacity)))

    where A = SATURATION_AMPLITUDE, k = SATURATION_STEEPNESS.  When combined
    is well below capacity the factor is ~1; when well above it approaches
    1 + A.
    """
    x = combined - capacity
    logistic = 1.0 / (1.0 + np.exp(-SATURATION_STEEPNESS * x))
    return 1.0 + SATURATION_AMPLITUDE * logistic


def _compute_interaction_boost(per_channel_severity: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute cross-channel synergy amplification.

    Returns:
        (total_boost, interaction_detail)
        total_boost: scalar to add to total severity
        interaction_detail: vector of per-pair interaction contributions
            (flattened upper-triangle, for decomposition)
    """
    # For each pair (i,j) with I[i,j]>0, boost = I[i,j] * s_i * s_j
    # This is a bilinear form: s^T I s
    total_boost = float(per_channel_severity @ INTERACTION_MATRIX @ per_channel_severity)

    # Per-pair detail (upper triangle only) for decomposition
    n_pairs = NUM_CHANNELS * (NUM_CHANNELS - 1) // 2
    interaction_detail = np.zeros(n_pairs)
    idx = 0
    for i in range(NUM_CHANNELS):
        for j in range(i + 1, NUM_CHANNELS):
            interaction_detail[idx] = (
                INTERACTION_MATRIX[i, j]
                * per_channel_severity[i]
                * per_channel_severity[j]
            )
            idx += 1
    return total_boost, interaction_detail


def _compute_threshold_spike_probability(
    combined_pressure: np.ndarray,
    capacity: np.ndarray,
    dist_atten: float,
    load: float,
    regime: str,
) -> float:
    """Compute spike probability from channel-specific threshold crossings.

    Instead of relying on generic overlap, spikes are triggered when specific
    *pairs* of channels exceed critical thresholds simultaneously.  This
    creates structured tail events that smooth models cannot predict.
    """
    # Normalised pressure relative to capacity (can exceed 1.0)
    relative_pressure = combined_pressure / np.maximum(capacity, 1e-6)

    spike_accum = 0.0
    for ch_a, ch_b, threshold, boost in SPIKE_TRIGGERS:
        # Both channels must be above threshold for the trigger to fire
        pair_pressure = min(relative_pressure[ch_a], relative_pressure[ch_b])
        if pair_pressure > threshold:
            # Boost scales with how far above threshold the pair is
            excess = pair_pressure - threshold
            spike_accum += boost * (1.0 + 2.0 * excess)

    # Scale by load, distance, regime (same physics as before)
    spike_load = load ** 1.5
    spike_distance = dist_atten ** 0.5
    spike_regime = {"benign": 0.3, "structured": 1.0, "adversarial": 2.5}[regime]

    # Small baseline ensures non-zero probability even without threshold crossings
    spike_base = 0.01

    spike_probability = min(0.6, (spike_base + spike_accum) * spike_load * spike_distance * spike_regime)
    return spike_probability


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

    # --- Nonlinear saturation (NEW) ---
    # When combined pressure exceeds capacity the interference grows
    # super-linearly following a logistic curve.
    saturation_mult = _logistic_saturation(combined_pressure, device.channel_capacities)
    saturation = np.maximum(0, combined_pressure - device.channel_capacities)
    # Apply nonlinear amplification to the excess
    saturation *= saturation_mult

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

    # --- Cross-channel interaction boost (NEW) ---
    interaction_boost, _ = _compute_interaction_boost(per_channel_severity)
    total_severity += interaction_boost

    # --- Channel-threshold spike probability (NEW) ---
    spike_probability = _compute_threshold_spike_probability(
        combined_pressure, device.channel_capacities, dist_atten, load, regime
    )

    # Add channel noise contribution
    noise_contribution = float(np.sum(device.channel_noise * overlap))
    total_severity += noise_contribution

    return total_severity, per_channel_severity, spike_probability


# ===================================================================
# Channel decomposition (for analysis figures)
# ===================================================================

def compute_channel_decomposition(
    target: Workload,
    spectator: Workload,
    device: DeviceProfile,
    distance: str,
    load: float,
    regime: str,
) -> pd.DataFrame:
    """Return a DataFrame decomposing total interference into per-channel
    and cross-channel interaction contributions.

    This is used for the channel decomposition analysis figure.

    Columns:
        channel        - channel name (or "interaction:<pair>" for synergy terms)
        overlap        - raw overlap for that channel
        saturation     - excess pressure above capacity
        saturation_nl  - nonlinear saturation multiplier
        distance_atten - distance attenuation applied
        severity       - final per-channel severity after load/regime scaling
        spike_contrib  - contribution to spike probability from threshold triggers
        pct_total      - percentage of total severity

    Returns:
        DataFrame with one row per channel plus rows for non-zero interaction
        pairs and a summary row.
    """
    overlap = compute_channel_overlap(target, spectator)
    combined_pressure = target.channel_pressure + spectator.channel_pressure * load

    saturation_mult = _logistic_saturation(combined_pressure, device.channel_capacities)
    saturation_raw = np.maximum(0, combined_pressure - device.channel_capacities)
    saturation_nl = saturation_raw * saturation_mult

    dist_atten = DISTANCE_ATTENUATION[distance]
    channel_atten = dist_atten ** CHANNEL_DISTANCE_SENSITIVITY

    load_factor = load * (1.0 + load ** 2)
    regime_mult = REGIME_MULTIPLIERS[regime]

    per_channel_severity = overlap * saturation_nl * channel_atten * load_factor * regime_mult

    # Channel noise contribution per channel
    noise_per_ch = device.channel_noise * overlap

    # Per-channel total (severity + noise)
    per_channel_total = per_channel_severity + noise_per_ch

    total_severity = float(np.sum(per_channel_total))

    # Interaction terms
    interaction_boost, interaction_detail = _compute_interaction_boost(per_channel_severity)
    total_severity += interaction_boost

    # Spike decomposition
    relative_pressure = combined_pressure / np.maximum(device.channel_capacities, 1e-6)
    spike_load = load ** 1.5
    spike_distance = dist_atten ** 0.5
    spike_regime = {"benign": 0.3, "structured": 1.0, "adversarial": 2.5}[regime]

    # Build rows
    rows = []
    for i, ch_name in enumerate(CHANNELS):
        # Compute spike contrib for triggers involving this channel
        ch_spike = 0.0
        for ch_a, ch_b, threshold, boost in SPIKE_TRIGGERS:
            if ch_a == i or ch_b == i:
                pair_pressure = min(relative_pressure[ch_a], relative_pressure[ch_b])
                if pair_pressure > threshold:
                    excess = pair_pressure - threshold
                    ch_spike += 0.5 * boost * (1.0 + 2.0 * excess)  # half credit each

        rows.append({
            "channel": ch_name,
            "overlap": float(overlap[i]),
            "combined_pressure": float(combined_pressure[i]),
            "capacity": float(device.channel_capacities[i]),
            "saturation_raw": float(saturation_raw[i]),
            "saturation_nl_mult": float(saturation_mult[i]),
            "distance_atten": float(channel_atten[i]),
            "severity": float(per_channel_total[i]),
            "spike_contrib": float(ch_spike * spike_load * spike_distance * spike_regime),
            "pct_total": float(per_channel_total[i] / total_severity * 100) if total_severity > 0 else 0.0,
        })

    # Interaction pair rows
    idx = 0
    for i in range(NUM_CHANNELS):
        for j in range(i + 1, NUM_CHANNELS):
            val = interaction_detail[idx]
            idx += 1
            if abs(val) > 1e-8:
                pair_name = f"interaction:{CHANNELS[i]}+{CHANNELS[j]}"
                rows.append({
                    "channel": pair_name,
                    "overlap": 0.0,
                    "combined_pressure": 0.0,
                    "capacity": 0.0,
                    "saturation_raw": 0.0,
                    "saturation_nl_mult": 0.0,
                    "distance_atten": 0.0,
                    "severity": float(val),
                    "spike_contrib": 0.0,
                    "pct_total": float(val / total_severity * 100) if total_severity > 0 else 0.0,
                })

    # Summary row
    total_spike = _compute_threshold_spike_probability(
        combined_pressure, device.channel_capacities, dist_atten, load, regime
    )
    rows.append({
        "channel": "TOTAL",
        "overlap": float(np.sum(overlap)),
        "combined_pressure": float(np.sum(combined_pressure)),
        "capacity": float(np.sum(device.channel_capacities)),
        "saturation_raw": float(np.sum(saturation_raw)),
        "saturation_nl_mult": 0.0,
        "distance_atten": 0.0,
        "severity": float(total_severity),
        "spike_contrib": float(total_spike),
        "pct_total": 100.0,
    })

    return pd.DataFrame(rows)

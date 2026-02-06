"""Workload definitions for SIT simulator.

Each workload has:
- channel_pressure: vector in R^C for interference channels
  channels = [LLC, MEM_BW, TLB, PREFETCH, NUMA, THERMAL, OS_FAULTS]
- resource_vector: for kernel similarity in scheduling
- baseline latency parameters
- burstiness characteristics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Channel indices
CHANNELS = ["LLC", "MEM_BW", "TLB", "PREFETCH", "NUMA", "THERMAL", "OS_FAULTS"]
NUM_CHANNELS = len(CHANNELS)
CH_IDX = {name: i for i, name in enumerate(CHANNELS)}


@dataclass
class Workload:
    """A workload definition (target or spectator)."""
    name: str
    wtype: str  # "target" or "spectator"
    channel_pressure: np.ndarray  # shape (NUM_CHANNELS,)
    resource_vector: np.ndarray   # shape (NUM_CHANNELS,) for kernel similarity
    base_latency_us: float        # baseline median latency in microseconds
    latency_shape: float          # shape parameter for lognormal
    burstiness: float             # 0-1, probability of burst mode
    burst_multiplier: float       # latency multiplier during burst
    description: str = ""

    def __post_init__(self):
        self.channel_pressure = np.asarray(self.channel_pressure, dtype=np.float64)
        self.resource_vector = np.asarray(self.resource_vector, dtype=np.float64)
        assert self.channel_pressure.shape == (NUM_CHANNELS,)
        assert self.resource_vector.shape == (NUM_CHANNELS,)


def get_targets() -> Dict[str, Workload]:
    """Return latency-sensitive target workloads."""
    targets = {}

    targets["rpc_microservice"] = Workload(
        name="rpc_microservice",
        wtype="target",
        # Moderate LLC, low mem BW, low TLB, some prefetch, low NUMA, low thermal, low OS
        channel_pressure=np.array([0.4, 0.2, 0.15, 0.3, 0.1, 0.05, 0.1]),
        resource_vector=np.array([0.4, 0.2, 0.15, 0.3, 0.1, 0.05, 0.1]),
        base_latency_us=200.0,
        latency_shape=0.3,
        burstiness=0.05,
        burst_multiplier=2.5,
        description="RPC-like microservice with moderate cache footprint",
    )

    targets["inference_request"] = Workload(
        name="inference_request",
        wtype="target",
        # High LLC, high mem BW, moderate TLB, high prefetch, moderate NUMA, moderate thermal, low OS
        channel_pressure=np.array([0.7, 0.8, 0.3, 0.6, 0.4, 0.3, 0.05]),
        resource_vector=np.array([0.7, 0.8, 0.3, 0.6, 0.4, 0.3, 0.05]),
        base_latency_us=5000.0,
        latency_shape=0.4,
        burstiness=0.08,
        burst_multiplier=3.0,
        description="Online inference request (e.g., ML model serving)",
    )

    targets["realtime_control"] = Workload(
        name="realtime_control",
        wtype="target",
        # Low LLC, low mem BW, low TLB, low prefetch, low NUMA, sensitive to thermal, moderate OS
        channel_pressure=np.array([0.15, 0.1, 0.1, 0.1, 0.05, 0.2, 0.25]),
        resource_vector=np.array([0.15, 0.1, 0.1, 0.1, 0.05, 0.2, 0.25]),
        base_latency_us=50.0,
        latency_shape=0.2,
        burstiness=0.02,
        burst_multiplier=5.0,
        description="Realtime control loop with tight deadline",
    )

    targets["kv_lookup"] = Workload(
        name="kv_lookup",
        wtype="target",
        # High LLC (working set fits), low mem BW, moderate TLB, moderate prefetch, low NUMA, low thermal, low OS
        channel_pressure=np.array([0.8, 0.15, 0.35, 0.4, 0.1, 0.05, 0.05]),
        resource_vector=np.array([0.8, 0.15, 0.35, 0.4, 0.1, 0.05, 0.05]),
        base_latency_us=30.0,
        latency_shape=0.25,
        burstiness=0.03,
        burst_multiplier=4.0,
        description="Key-value lookup with large cache working set",
    )

    targets["streaming_frame"] = Workload(
        name="streaming_frame",
        wtype="target",
        # Moderate LLC, high mem BW, moderate TLB, high prefetch, moderate NUMA, moderate thermal, moderate OS
        channel_pressure=np.array([0.35, 0.7, 0.25, 0.5, 0.3, 0.2, 0.15]),
        resource_vector=np.array([0.35, 0.7, 0.25, 0.5, 0.3, 0.2, 0.15]),
        base_latency_us=1500.0,
        latency_shape=0.35,
        burstiness=0.06,
        burst_multiplier=2.0,
        description="Streaming frame pipeline with bandwidth demand",
    )

    return targets


def get_spectators() -> Dict[str, Workload]:
    """Return spectator (interferer) workloads."""
    spectators = {}

    spectators["cache_thrash"] = Workload(
        name="cache_thrash",
        wtype="spectator",
        channel_pressure=np.array([0.95, 0.3, 0.2, 0.1, 0.05, 0.1, 0.05]),
        resource_vector=np.array([0.95, 0.3, 0.2, 0.1, 0.05, 0.1, 0.05]),
        base_latency_us=100.0,
        latency_shape=0.5,
        burstiness=0.1,
        burst_multiplier=2.0,
        description="Cache thrashing workload saturating LLC",
    )

    spectators["membw_saturator"] = Workload(
        name="membw_saturator",
        wtype="spectator",
        channel_pressure=np.array([0.3, 0.95, 0.15, 0.2, 0.3, 0.15, 0.05]),
        resource_vector=np.array([0.3, 0.95, 0.15, 0.2, 0.3, 0.15, 0.05]),
        base_latency_us=200.0,
        latency_shape=0.4,
        burstiness=0.08,
        burst_multiplier=2.5,
        description="Memory bandwidth saturator via streaming access",
    )

    spectators["tlb_stress"] = Workload(
        name="tlb_stress",
        wtype="spectator",
        channel_pressure=np.array([0.2, 0.2, 0.95, 0.1, 0.15, 0.05, 0.15]),
        resource_vector=np.array([0.2, 0.2, 0.95, 0.1, 0.15, 0.05, 0.15]),
        base_latency_us=150.0,
        latency_shape=0.45,
        burstiness=0.07,
        burst_multiplier=3.0,
        description="TLB stress via large scattered page access",
    )

    spectators["numa_remote"] = Workload(
        name="numa_remote",
        wtype="spectator",
        channel_pressure=np.array([0.15, 0.4, 0.2, 0.1, 0.95, 0.1, 0.1]),
        resource_vector=np.array([0.15, 0.4, 0.2, 0.1, 0.95, 0.1, 0.1]),
        base_latency_us=300.0,
        latency_shape=0.5,
        burstiness=0.06,
        burst_multiplier=2.0,
        description="NUMA remote memory stress",
    )

    spectators["prefetch_adversary"] = Workload(
        name="prefetch_adversary",
        wtype="spectator",
        channel_pressure=np.array([0.4, 0.3, 0.15, 0.95, 0.1, 0.05, 0.05]),
        resource_vector=np.array([0.4, 0.3, 0.15, 0.95, 0.1, 0.05, 0.05]),
        base_latency_us=120.0,
        latency_shape=0.35,
        burstiness=0.05,
        burst_multiplier=2.5,
        description="Prefetch adversary with irregular access patterns",
    )

    spectators["io_burst"] = Workload(
        name="io_burst",
        wtype="spectator",
        channel_pressure=np.array([0.1, 0.25, 0.1, 0.05, 0.1, 0.05, 0.7]),
        resource_vector=np.array([0.1, 0.25, 0.1, 0.05, 0.1, 0.05, 0.7]),
        base_latency_us=500.0,
        latency_shape=0.6,
        burstiness=0.2,
        burst_multiplier=4.0,
        description="IO burst workload causing interrupt and OS scheduling pressure",
    )

    spectators["pagefault_heavy"] = Workload(
        name="pagefault_heavy",
        wtype="spectator",
        channel_pressure=np.array([0.15, 0.3, 0.5, 0.1, 0.2, 0.05, 0.85]),
        resource_vector=np.array([0.15, 0.3, 0.5, 0.1, 0.2, 0.05, 0.85]),
        base_latency_us=400.0,
        latency_shape=0.55,
        burstiness=0.15,
        burst_multiplier=3.5,
        description="Page fault heavy workload causing TLB and OS pressure",
    )

    spectators["thermal_stress"] = Workload(
        name="thermal_stress",
        wtype="spectator",
        channel_pressure=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.95, 0.1]),
        resource_vector=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.95, 0.1]),
        base_latency_us=80.0,
        latency_shape=0.3,
        burstiness=0.04,
        burst_multiplier=2.0,
        description="CPU thermal stress via sustained compute",
    )

    # Additional spectators for diversity
    spectators["mixed_cache_membw"] = Workload(
        name="mixed_cache_membw",
        wtype="spectator",
        channel_pressure=np.array([0.7, 0.75, 0.2, 0.3, 0.15, 0.2, 0.1]),
        resource_vector=np.array([0.7, 0.75, 0.2, 0.3, 0.15, 0.2, 0.1]),
        base_latency_us=180.0,
        latency_shape=0.45,
        burstiness=0.1,
        burst_multiplier=2.5,
        description="Mixed cache and memory bandwidth pressure",
    )

    spectators["light_background"] = Workload(
        name="light_background",
        wtype="spectator",
        channel_pressure=np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.15]),
        resource_vector=np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.15]),
        base_latency_us=50.0,
        latency_shape=0.2,
        burstiness=0.02,
        burst_multiplier=1.5,
        description="Light background daemon with minimal interference",
    )

    return spectators


def get_all_workloads():
    """Return (targets_dict, spectators_dict)."""
    return get_targets(), get_spectators()

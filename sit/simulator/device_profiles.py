"""Device profiles for SIT simulator.

Each device defines:
- channel_capacities: how much each resource channel can absorb before saturation
- channel_noise: baseline noise per channel
- dvfs_drift_params: parameters for DVFS and thermal drift
- baseline_latency_scale: multiplier on workload base latency
- name and description
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from .workloads import NUM_CHANNELS


@dataclass
class DeviceProfile:
    """A simulated device profile."""
    name: str
    device_id: str
    channel_capacities: np.ndarray   # shape (NUM_CHANNELS,), higher = more headroom
    channel_noise: np.ndarray        # shape (NUM_CHANNELS,), baseline noise per channel
    baseline_latency_scale: float    # multiplier on workload base latency
    thermal_ramp_rate: float         # thermal drift rate (multiplicative per timestep)
    thermal_ceiling: float           # max thermal multiplier
    dvfs_step_prob: float            # probability of a DVFS step change per trial
    dvfs_step_magnitude: float       # magnitude of DVFS step
    background_burst_prob: float     # probability of background daemon burst
    background_burst_severity: float # severity multiplier of background burst
    description: str = ""

    def __post_init__(self):
        self.channel_capacities = np.asarray(self.channel_capacities, dtype=np.float64)
        self.channel_noise = np.asarray(self.channel_noise, dtype=np.float64)
        assert self.channel_capacities.shape == (NUM_CHANNELS,)
        assert self.channel_noise.shape == (NUM_CHANNELS,)


def get_device_profiles() -> Dict[str, DeviceProfile]:
    """Return 10 simulated device profiles."""
    profiles = {}

    # 1. Low-power embedded
    profiles["embedded_a"] = DeviceProfile(
        name="Embedded-A",
        device_id="embedded_a",
        channel_capacities=np.array([0.3, 0.2, 0.25, 0.2, 0.1, 0.4, 0.3]),
        channel_noise=np.array([0.08, 0.06, 0.07, 0.05, 0.03, 0.1, 0.08]),
        baseline_latency_scale=2.5,
        thermal_ramp_rate=0.003,
        thermal_ceiling=1.4,
        dvfs_step_prob=0.08,
        dvfs_step_magnitude=0.15,
        background_burst_prob=0.03,
        background_burst_severity=1.8,
        description="Low-power embedded SoC (simulated)",
    )

    # 2. Low-power embedded variant
    profiles["embedded_b"] = DeviceProfile(
        name="Embedded-B",
        device_id="embedded_b",
        channel_capacities=np.array([0.35, 0.25, 0.3, 0.25, 0.15, 0.35, 0.25]),
        channel_noise=np.array([0.07, 0.05, 0.06, 0.05, 0.04, 0.09, 0.07]),
        baseline_latency_scale=2.2,
        thermal_ramp_rate=0.0025,
        thermal_ceiling=1.35,
        dvfs_step_prob=0.06,
        dvfs_step_magnitude=0.12,
        background_burst_prob=0.025,
        background_burst_severity=1.6,
        description="Low-power embedded SoC variant (simulated)",
    )

    # 3. Laptop CPU
    profiles["laptop_a"] = DeviceProfile(
        name="Laptop-A",
        device_id="laptop_a",
        channel_capacities=np.array([0.5, 0.45, 0.4, 0.45, 0.2, 0.5, 0.4]),
        channel_noise=np.array([0.06, 0.05, 0.05, 0.04, 0.03, 0.08, 0.06]),
        baseline_latency_scale=1.5,
        thermal_ramp_rate=0.002,
        thermal_ceiling=1.3,
        dvfs_step_prob=0.1,
        dvfs_step_magnitude=0.1,
        background_burst_prob=0.05,
        background_burst_severity=1.5,
        description="Laptop-class CPU (simulated)",
    )

    # 4. Laptop CPU variant
    profiles["laptop_b"] = DeviceProfile(
        name="Laptop-B",
        device_id="laptop_b",
        channel_capacities=np.array([0.55, 0.5, 0.45, 0.5, 0.25, 0.45, 0.35]),
        channel_noise=np.array([0.05, 0.04, 0.05, 0.04, 0.03, 0.07, 0.05]),
        baseline_latency_scale=1.4,
        thermal_ramp_rate=0.0018,
        thermal_ceiling=1.25,
        dvfs_step_prob=0.08,
        dvfs_step_magnitude=0.08,
        background_burst_prob=0.04,
        background_burst_severity=1.4,
        description="Laptop-class CPU variant (simulated)",
    )

    # 5. Desktop workstation
    profiles["workstation_a"] = DeviceProfile(
        name="Workstation-A",
        device_id="workstation_a",
        channel_capacities=np.array([0.7, 0.65, 0.6, 0.65, 0.4, 0.6, 0.5]),
        channel_noise=np.array([0.04, 0.03, 0.04, 0.03, 0.03, 0.05, 0.04]),
        baseline_latency_scale=1.0,
        thermal_ramp_rate=0.001,
        thermal_ceiling=1.15,
        dvfs_step_prob=0.05,
        dvfs_step_magnitude=0.06,
        background_burst_prob=0.03,
        background_burst_severity=1.3,
        description="Desktop workstation (simulated)",
    )

    # 6. Workstation variant
    profiles["workstation_b"] = DeviceProfile(
        name="Workstation-B",
        device_id="workstation_b",
        channel_capacities=np.array([0.75, 0.7, 0.55, 0.6, 0.45, 0.65, 0.55]),
        channel_noise=np.array([0.04, 0.03, 0.04, 0.03, 0.02, 0.04, 0.04]),
        baseline_latency_scale=0.95,
        thermal_ramp_rate=0.0008,
        thermal_ceiling=1.12,
        dvfs_step_prob=0.04,
        dvfs_step_magnitude=0.05,
        background_burst_prob=0.025,
        background_burst_severity=1.25,
        description="Desktop workstation variant (simulated)",
    )

    # 7. Server (high core count)
    profiles["server_a"] = DeviceProfile(
        name="Server-A",
        device_id="server_a",
        channel_capacities=np.array([0.85, 0.8, 0.7, 0.75, 0.7, 0.7, 0.6]),
        channel_noise=np.array([0.03, 0.025, 0.03, 0.025, 0.03, 0.04, 0.03]),
        baseline_latency_scale=0.8,
        thermal_ramp_rate=0.0005,
        thermal_ceiling=1.1,
        dvfs_step_prob=0.03,
        dvfs_step_magnitude=0.04,
        background_burst_prob=0.02,
        background_burst_severity=1.2,
        description="High core-count server (simulated)",
    )

    # 8. Server variant
    profiles["server_b"] = DeviceProfile(
        name="Server-B",
        device_id="server_b",
        channel_capacities=np.array([0.9, 0.85, 0.75, 0.8, 0.75, 0.75, 0.65]),
        channel_noise=np.array([0.025, 0.02, 0.025, 0.02, 0.025, 0.03, 0.025]),
        baseline_latency_scale=0.75,
        thermal_ramp_rate=0.0004,
        thermal_ceiling=1.08,
        dvfs_step_prob=0.02,
        dvfs_step_magnitude=0.03,
        background_burst_prob=0.015,
        background_burst_severity=1.15,
        description="High core-count server variant (simulated)",
    )

    # 9. ARM server
    profiles["arm_server_a"] = DeviceProfile(
        name="ARM-Server-A",
        device_id="arm_server_a",
        channel_capacities=np.array([0.65, 0.7, 0.6, 0.55, 0.6, 0.8, 0.55]),
        channel_noise=np.array([0.035, 0.03, 0.035, 0.03, 0.03, 0.03, 0.035]),
        baseline_latency_scale=0.9,
        thermal_ramp_rate=0.0006,
        thermal_ceiling=1.12,
        dvfs_step_prob=0.04,
        dvfs_step_magnitude=0.05,
        background_burst_prob=0.02,
        background_burst_severity=1.2,
        description="ARM-based server (simulated)",
    )

    # 10. ARM server variant
    profiles["arm_server_b"] = DeviceProfile(
        name="ARM-Server-B",
        device_id="arm_server_b",
        channel_capacities=np.array([0.7, 0.75, 0.65, 0.6, 0.65, 0.75, 0.5]),
        channel_noise=np.array([0.03, 0.025, 0.03, 0.025, 0.025, 0.03, 0.03]),
        baseline_latency_scale=0.85,
        thermal_ramp_rate=0.0005,
        thermal_ceiling=1.1,
        dvfs_step_prob=0.035,
        dvfs_step_magnitude=0.04,
        background_burst_prob=0.018,
        background_burst_severity=1.18,
        description="ARM-based server variant (simulated)",
    )

    return profiles

"""SIT scheduling module.

Provides tail-risk-aware, diversity-promoting schedulers for co-tenant
placement, along with baseline policies for comparison.
"""

from sit.scheduling.utils import (
    build_similarity_kernel,
    logdet_objective,
    marginal_gain,
    compute_predicted_risk,
)
from sit.scheduling.sit_dpp import sit_dpp_schedule
from sit.scheduling.sit_slo import sit_slo_schedule

__all__ = [
    "build_similarity_kernel",
    "logdet_objective",
    "marginal_gain",
    "compute_predicted_risk",
    "sit_dpp_schedule",
    "sit_slo_schedule",
]

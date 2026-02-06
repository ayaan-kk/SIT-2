"""Tomography module: interference map reconstruction with uncertainty.

Builds structured interference matrices from IRBS experimental results,
provides bootstrap confidence intervals, sparsity analysis, and
UCB-based conservative ranking of interferers.
"""

from .build_matrix import build_tomography_matrix, build_ssi_matrix
from .bootstrap import bootstrap_cell, bootstrap_all_cells
from .sparsity_recovery import get_top_k_interferers, sparsity_stats, recovery_curve
from .ucb import compute_ucb, conservative_top_interferers

__all__ = [
    "build_tomography_matrix",
    "build_ssi_matrix",
    "bootstrap_cell",
    "bootstrap_all_cells",
    "get_top_k_interferers",
    "sparsity_stats",
    "recovery_curve",
    "compute_ucb",
    "conservative_top_interferers",
]

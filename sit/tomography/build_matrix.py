"""Build interference tomography matrices from IRBS results.

Provides two matrix-construction functions:

* :func:`build_tomography_matrix` -- bootstraps every (target, spectator)
  cell to produce mean-effect, CI-lower, and CI-upper DataFrames.
* :func:`build_ssi_matrix` -- computes a Structured Severity Index (SSI)
  that combines p99 and CVaR log-ratios into a single scalar per cell.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .bootstrap import bootstrap_all_cells


def build_tomography_matrix(
    irbs_results_dict: dict,
    targets: List[str],
    spectators: List[str],
    metric: str = "p99",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tomography matrices from IRBS experimental results.

    Bootstraps every (target, spectator) cell to obtain a mean treatment
    effect and 95 % confidence-interval bounds, then arranges the results
    into three aligned DataFrames.

    Args:
        irbs_results_dict: dict keyed by ``(target_name, spectator_name)``.
            Each value must be a dict with:

            * ``'trial_types'``: array of ``'control'`` / ``'treatment'``
            * ``'per_trial_summaries'``: dict mapping metric names to
              per-trial arrays

        targets: ordered list of target names (row labels).
        spectators: ordered list of spectator names (column labels).
        metric: metric to use for effect computation (default ``'p99'``).

    Returns:
        ``(mean_matrix, ci_lower_matrix, ci_upper_matrix)`` -- three
        :class:`pandas.DataFrame` objects with *targets* as rows and
        *spectators* as columns.  Values are the mean treatment effect,
        lower CI bound, and upper CI bound respectively.
    """
    # Bootstrap all cells to get (mean, ci_lo, ci_hi, std_err)
    bootstrapped = bootstrap_all_cells(irbs_results_dict, metric=metric)

    n_targets = len(targets)
    n_spectators = len(spectators)

    mean_data = np.zeros((n_targets, n_spectators))
    ci_lower_data = np.zeros((n_targets, n_spectators))
    ci_upper_data = np.zeros((n_targets, n_spectators))

    for i, t in enumerate(targets):
        for j, s in enumerate(spectators):
            key = (t, s)
            if key in bootstrapped:
                mean_val, ci_lo, ci_hi, _stderr = bootstrapped[key]
                mean_data[i, j] = mean_val
                ci_lower_data[i, j] = ci_lo
                ci_upper_data[i, j] = ci_hi

    mean_matrix = pd.DataFrame(mean_data, index=targets, columns=spectators)
    ci_lower_matrix = pd.DataFrame(ci_lower_data, index=targets, columns=spectators)
    ci_upper_matrix = pd.DataFrame(ci_upper_data, index=targets, columns=spectators)

    return mean_matrix, ci_lower_matrix, ci_upper_matrix


def build_ssi_matrix(
    irbs_results_dict: dict,
    targets: List[str],
    spectators: List[str],
    s: float = 2.0,
    w: float = 0.5,
) -> pd.DataFrame:
    """Compute a Structured Severity Index (SSI) matrix.

    SSI fuses p99 and CVaR treatment-vs-control ratios into a single
    severity score per cell:

    .. math::

        R_{p99}  &= \\max\\bigl(0,\\; \\ln(p99_{treat} / p99_{ctrl})\\bigr) \\\\
        R_{CVaR} &= \\max\\bigl(0,\\; \\ln(CVaR_{treat} / CVaR_{ctrl})\\bigr) \\\\
        SSI      &= s \\cdot \\ln\\bigl(1 + R_{p99} + w \\cdot R_{CVaR}\\bigr)

    Args:
        irbs_results_dict: dict keyed by ``(target_name, spectator_name)``.
            Each value must contain ``'trial_types'`` and
            ``'per_trial_summaries'`` with at least ``'p99'`` and
            ``'cvar'`` entries.
        targets: ordered list of target names (row labels).
        spectators: ordered list of spectator names (column labels).
        s: outer scaling factor (default 2.0).
        w: weight for the CVaR component (default 0.5).

    Returns:
        :class:`pandas.DataFrame` with SSI values, *targets* as rows and
        *spectators* as columns.
    """
    n_targets = len(targets)
    n_spectators = len(spectators)
    ssi_data = np.zeros((n_targets, n_spectators))

    for i, t in enumerate(targets):
        for j, sp in enumerate(spectators):
            key = (t, sp)
            if key not in irbs_results_dict:
                continue

            result = irbs_results_dict[key]
            trial_types = np.asarray(result["trial_types"])
            summaries = result["per_trial_summaries"]

            ctrl_mask = trial_types == "control"
            treat_mask = trial_types == "treatment"

            if not np.any(ctrl_mask) or not np.any(treat_mask):
                continue

            p99_values = np.asarray(summaries["p99"], dtype=np.float64)
            cvar_values = np.asarray(summaries["cvar"], dtype=np.float64)

            # Aggregate: mean of per-trial metric across control / treatment
            p99_ctrl = float(np.mean(p99_values[ctrl_mask]))
            p99_treat = float(np.mean(p99_values[treat_mask]))
            cvar_ctrl = float(np.mean(cvar_values[ctrl_mask]))
            cvar_treat = float(np.mean(cvar_values[treat_mask]))

            # Guard against zero / negative denominators
            if p99_ctrl <= 0 or cvar_ctrl <= 0:
                continue

            R_p99 = max(0.0, np.log(p99_treat / p99_ctrl))
            R_cvar = max(0.0, np.log(cvar_treat / cvar_ctrl))

            ssi_data[i, j] = s * np.log(1.0 + R_p99 + w * R_cvar)

    return pd.DataFrame(ssi_data, index=targets, columns=spectators)

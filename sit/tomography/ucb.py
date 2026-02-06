"""Upper Confidence Bound utilities for interference tomography.

Provides UCB-based conservative ranking of interferers.  By adding a
scaled standard-error term to each mean effect estimate we obtain a
pessimistic upper bound that accounts for estimation uncertainty --
useful for safety-critical scheduling decisions where *under*-estimating
interference is worse than *over*-estimating it.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_ucb(
    mean_matrix: pd.DataFrame,
    stderr_matrix: pd.DataFrame,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Compute the Upper Confidence Bound matrix.

    .. math::

        UCB_{ij} = \\text{mean}_{ij} + \\beta \\cdot \\text{stderr}_{ij}

    Args:
        mean_matrix: DataFrame of mean interference effects (targets x
            spectators).
        stderr_matrix: DataFrame of standard errors, same shape and
            index/columns as *mean_matrix*.
        beta: exploration / conservatism parameter (default 2.0).

    Returns:
        DataFrame of UCB values with identical index and columns.
    """
    ucb = mean_matrix + beta * stderr_matrix
    ucb.index = mean_matrix.index
    ucb.columns = mean_matrix.columns
    return ucb


def conservative_top_interferers(
    ucb_matrix: pd.DataFrame,
    target_name: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return the top-*k* interferers for *target_name* ranked by UCB.

    This provides a *conservative* (pessimistic) ranking: spectators
    whose true interference could plausibly be the highest are ranked
    first, even if their point estimate is not the largest.

    Args:
        ucb_matrix: DataFrame with UCB values (targets x spectators).
        target_name: row label of the target to query.
        k: number of top interferers to return (default 3).

    Returns:
        List of ``(spectator_name, ucb_value)`` sorted in descending
        UCB order.
    """
    row = ucb_matrix.loc[target_name]
    top_k = row.nlargest(k)
    return list(zip(top_k.index, top_k.values))

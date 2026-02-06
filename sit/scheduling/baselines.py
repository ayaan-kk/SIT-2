"""Baseline scheduling policies for SIT evaluation.

Each baseline takes the same signature so they can be swapped in
experiment scripts.  All return a list of selected spectator names.

Common Parameters
-----------------
target : str
    Name of the latency-sensitive target workload.
candidate_spectators : list of str
    Names of spectator workloads eligible for co-location.
n_slots : int
    Number of co-location slots to fill.
tomography_mean_matrix : pd.DataFrame
    Mean predicted interference (rows = targets, columns = spectators).
tomography_stderr_matrix : pd.DataFrame or None
    Standard-error counterpart of the mean matrix.
kernel_matrix : np.ndarray
    Pre-computed RBF similarity kernel (indices align with *workload_names*).
workload_names : list of str
    Ordered names corresponding to kernel_matrix rows/columns.
rng : np.random.Generator
    Random number generator for reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sit.scheduling.utils import compute_predicted_risk


# ---------------------------------------------------------------------------
# 1. Random placement
# ---------------------------------------------------------------------------

def random_placement(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Randomly select *n_slots* spectators without replacement.

    This is the simplest baseline: a scheduler with no awareness of
    interference whatsoever.
    """
    n_pick = min(n_slots, len(candidate_spectators))
    chosen = rng.choice(candidate_spectators, size=n_pick, replace=False)
    return list(chosen)


# ---------------------------------------------------------------------------
# 2. Mean-greedy placement
# ---------------------------------------------------------------------------

def mean_greedy(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Greedily select spectators with the lowest mean predicted interference.

    At each step the spectator that adds the least mean interference to
    the running total is chosen.  This policy ignores tail risk and
    diversity; it only optimises expected interference.
    """
    selected: List[str] = []
    remaining = list(candidate_spectators)

    for _ in range(min(n_slots, len(candidate_spectators))):
        best_name = None
        best_risk = np.inf

        for s in remaining:
            risk = compute_predicted_risk(
                target_name=target,
                cotenant_names=selected + [s],
                tomography_mean=tomography_mean_matrix,
                tomography_stderr=None,  # mean only
                beta=0.0,
            )
            if risk < best_risk:
                best_risk = risk
                best_name = s

        if best_name is None:
            break
        selected.append(best_name)
        remaining.remove(best_name)

    return selected


# ---------------------------------------------------------------------------
# 3. Similarity avoidance
# ---------------------------------------------------------------------------

def similarity_avoidance(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
    similarity_threshold: float = 0.8,
) -> List[str]:
    """Avoid spectators whose cosine similarity to the target exceeds a threshold.

    Spectators with similarity above *similarity_threshold* are dropped.
    From the remaining candidates, up to *n_slots* are selected in order
    of ascending similarity (least-similar first), with ties broken
    randomly.  If too few candidates survive the threshold, all survivors
    are returned.

    Cosine similarity is derived from the kernel matrix rows (which are
    already based on resource vectors).
    """
    if target not in workload_names:
        # Fall back to random if target not in kernel
        return random_placement(
            target, candidate_spectators, n_slots,
            tomography_mean_matrix, tomography_stderr_matrix,
            kernel_matrix, workload_names, rng,
        )

    target_idx = workload_names.index(target)
    target_vec = kernel_matrix[target_idx, :]  # similarity row

    # Compute cosine similarity using kernel rows as feature vectors
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        target_norm = 1.0

    candidate_sims: List[Tuple[str, float]] = []
    for s in candidate_spectators:
        if s not in workload_names:
            # Unknown spectator -- assign neutral similarity
            candidate_sims.append((s, 0.5))
            continue
        s_idx = workload_names.index(s)
        s_vec = kernel_matrix[s_idx, :]
        s_norm = np.linalg.norm(s_vec)
        if s_norm == 0:
            s_norm = 1.0
        cos_sim = np.dot(target_vec, s_vec) / (target_norm * s_norm)
        candidate_sims.append((s, float(cos_sim)))

    # Filter by threshold
    eligible = [(name, sim) for name, sim in candidate_sims
                if sim < similarity_threshold]

    # Sort ascending by similarity (least similar first); break ties randomly
    rng.shuffle(eligible)  # randomise before stable sort for tie-breaking
    eligible.sort(key=lambda x: x[1])

    selected = [name for name, _ in eligible[:n_slots]]
    return selected


# ---------------------------------------------------------------------------
# 4. Linux-proxy scheduler
# ---------------------------------------------------------------------------

def linux_proxy(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Simulate an OS-level scheduler that ignores micro-architectural channels.

    The Linux CFS scheduler has no knowledge of LLC contention, prefetch
    pollution, etc.  This proxy uses only a simplified CPU + memory
    utilisation metric (first two components of the resource vector, when
    available via the kernel matrix) to spread load.  In practice this
    performs similarly to random placement with a slight bias toward
    lighter workloads.

    The heuristic: rank candidates by the sum of their first two
    resource-vector-derived kernel features (a proxy for CPU + MEM_BW
    load) and pick the *n_slots* lightest.
    """
    # Build a simple load score per candidate
    scores: List[Tuple[str, float]] = []
    for s in candidate_spectators:
        if s in workload_names:
            s_idx = workload_names.index(s)
            # Use first two diagonal-adjacent kernel values as proxy for
            # CPU and memory load.  The kernel diagonal is always 1 for
            # RBF, so we use the raw resource-vector magnitude via the
            # kernel row sum as a rough proxy.
            row = kernel_matrix[s_idx, :]
            # Higher row sum => more similar to everything => likely
            # heavier resource consumer.  Use negative so lighter wins.
            load_proxy = float(row[0] + row[1]) if len(row) >= 2 else float(np.sum(row))
        else:
            load_proxy = 0.5  # neutral fallback

        scores.append((s, load_proxy))

    # Shuffle first for random tie-breaking, then stable sort ascending
    rng.shuffle(scores)
    scores.sort(key=lambda x: x[1])

    selected = [name for name, _ in scores[:n_slots]]
    return selected


# ---------------------------------------------------------------------------
# 5. Static partition
# ---------------------------------------------------------------------------

def static_partition(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Simulate static resource partitioning (e.g. Intel CAT / MBA).

    In this model the hardware partitions are assumed to eliminate
    *direct* LLC and memory-bandwidth contention.  The scheduler
    therefore allows **all** candidate spectators to be co-located
    (up to *n_slots*), relying on partitioning to contain interference.

    The returned list carries all candidates (capped at *n_slots*) and
    a metadata flag ``partitioning_applied=True`` can be checked by the
    caller via the presence of every candidate in the result.

    NOTE: partitioning does not eliminate TLB, prefetch, NUMA, thermal,
    or OS-fault channels -- these cross partition boundaries.  The
    downstream latency model should therefore still apply non-LLC/MEM_BW
    interference.
    """
    # Select all candidates up to n_slots (order does not matter, shuffle
    # for fairness).
    pool = list(candidate_spectators)
    rng.shuffle(pool)
    selected = pool[:n_slots]
    return selected


# ---------------------------------------------------------------------------
# 6. Triton baseline
# ---------------------------------------------------------------------------

def triton_baseline(
    target: str,
    candidate_spectators: List[str],
    n_slots: int,
    tomography_mean_matrix: pd.DataFrame,
    tomography_stderr_matrix: Optional[pd.DataFrame],
    kernel_matrix: np.ndarray,
    workload_names: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Simulate Triton Inference Server's batching-aware co-location policy.

    For inference-serving targets, Triton batches requests to improve
    throughput but does not consider micro-architectural interference.
    The placement heuristic is identical to *mean_greedy* (pick lowest
    mean interference), but the caller should note that Triton-style
    dynamic batching is applied on top.

    The returned list is the same as *mean_greedy* output.  Downstream
    evaluation should set ``triton_batching=True`` when computing
    latency to account for batching effects.
    """
    selected = mean_greedy(
        target=target,
        candidate_spectators=candidate_spectators,
        n_slots=n_slots,
        tomography_mean_matrix=tomography_mean_matrix,
        tomography_stderr_matrix=tomography_stderr_matrix,
        kernel_matrix=kernel_matrix,
        workload_names=workload_names,
        rng=rng,
    )
    return selected

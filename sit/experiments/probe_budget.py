"""Probe budget experiment: reconstruction error vs number of probes.

Evaluates how many (target, spectator) pair measurements are needed to
reconstruct a useful interference tomography map.  Compares four probe
selection strategies -- random, round-robin, UCB (variance-guided), and
DPP (diversity-promoting) -- across a range of probe budgets.

Key insight: DPP and UCB strategies should reach acceptable ranking
fidelity (NDCG, Kendall tau) with significantly fewer probes than
random or round-robin, demonstrating the value of *active* measurement
selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from itertools import product

from sit.simulator.workloads import get_targets, get_spectators
from sit.simulator.device_profiles import get_device_profiles
from sit.experiments.run_experiments import run_irbs_condition


# ------------------------------------------------------------------ #
#  Ranking and similarity helpers                                     #
# ------------------------------------------------------------------ #

def compute_ndcg(
    true_ranking: np.ndarray,
    estimated_ranking: np.ndarray,
    k: int,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at rank *k*.

    Both ``true_ranking`` and ``estimated_ranking`` are 1-D arrays of
    *relevance scores* (higher = more interfering) for the same ordered
    set of spectators.  The function ranks items by ``estimated_ranking``
    in descending order, then evaluates the quality of that ranking
    against the true relevance scores.

    .. math::

        DCG@k = \\sum_{i=1}^{k} \\frac{2^{rel_i} - 1}{\\log_2(i + 1)}

    NDCG@k = DCG@k / IDCG@k, where IDCG is the DCG of the ideal
    (true-score-sorted) ranking.

    Args:
        true_ranking: true relevance scores, shape ``(n_items,)``.
        estimated_ranking: estimated relevance scores, same shape.
        k: cut-off rank.

    Returns:
        NDCG@k in [0, 1].  Returns 1.0 when the estimated ranking
        perfectly matches the true top-*k*.
    """
    n = len(true_ranking)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)

    # Normalize relevance scores to [0, 1] to avoid overflow in 2^x
    true_min = np.min(true_ranking)
    true_max = np.max(true_ranking)
    if true_max - true_min > 1e-12:
        norm_true = (true_ranking - true_min) / (true_max - true_min)
    else:
        norm_true = np.zeros_like(true_ranking)

    # Rank items by estimated score (descending)
    est_order = np.argsort(-estimated_ranking)
    # Relevance of estimated top-k according to normalized true scores
    est_relevance = norm_true[est_order[:k]]

    # DCG of the estimated ranking
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum((2.0 ** est_relevance - 1.0) / discounts))

    # Ideal DCG: sort normalized true scores descending
    ideal_relevance = np.sort(norm_true)[::-1][:k]
    idcg = float(np.sum((2.0 ** ideal_relevance - 1.0) / discounts))

    if idcg < 1e-12:
        return 1.0  # all-zero relevance: any ranking is fine
    return dcg / idcg


def compute_kendall_tau(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
) -> float:
    """Compute Kendall's tau-b rank correlation coefficient.

    Counts concordant and discordant pairs between two score vectors
    and returns a correlation in [-1, 1].  A value of 1.0 means
    perfect agreement in pairwise ordering.

    Uses a simple O(n^2) implementation suitable for the small
    spectator-pool sizes in SIT experiments.

    Args:
        ranking_a: score vector A, shape ``(n,)``.
        ranking_b: score vector B, same shape.

    Returns:
        Kendall tau-b coefficient (float).
    """
    n = len(ranking_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff_a = ranking_a[i] - ranking_a[j]
            diff_b = ranking_b[i] - ranking_b[j]

            if abs(diff_a) < 1e-15:
                ties_a += 1
            if abs(diff_b) < 1e-15:
                ties_b += 1

            prod = diff_a * diff_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    denom = np.sqrt(
        (total_pairs - ties_a) * (total_pairs - ties_b)
    )
    if denom < 1e-12:
        return 0.0
    return float((concordant - discordant) / denom)


def _jaccard_at_k(true_scores: np.ndarray, est_scores: np.ndarray, k: int) -> float:
    """Jaccard similarity of the top-k sets."""
    k = min(k, len(true_scores))
    true_topk = set(np.argsort(-true_scores)[:k])
    est_topk = set(np.argsort(-est_scores)[:k])
    if len(true_topk) == 0:
        return 1.0
    return float(len(true_topk & est_topk) / len(true_topk | est_topk))


def _precision_at_k(true_scores: np.ndarray, est_scores: np.ndarray, k: int) -> float:
    """Precision@k: fraction of estimated top-k that are in true top-k."""
    k = min(k, len(true_scores))
    true_topk = set(np.argsort(-true_scores)[:k])
    est_topk = set(np.argsort(-est_scores)[:k])
    if k == 0:
        return 1.0
    return float(len(true_topk & est_topk) / k)


def _recall_at_k(true_scores: np.ndarray, est_scores: np.ndarray, k: int) -> float:
    """Recall@k: fraction of true top-k recovered by estimated top-k."""
    k = min(k, len(true_scores))
    true_topk = set(np.argsort(-true_scores)[:k])
    est_topk = set(np.argsort(-est_scores)[:k])
    if len(true_topk) == 0:
        return 1.0
    return float(len(true_topk & est_topk) / len(true_topk))


# ------------------------------------------------------------------ #
#  Probe selection strategies                                         #
# ------------------------------------------------------------------ #

def _select_random(
    all_pairs: List[Tuple[str, str]],
    n_probes: int,
    rng: np.random.Generator,
    **_kwargs,
) -> List[Tuple[str, str]]:
    """Select *n_probes* pairs uniformly at random."""
    n = min(n_probes, len(all_pairs))
    indices = rng.choice(len(all_pairs), size=n, replace=False)
    return [all_pairs[i] for i in indices]


def _select_round_robin(
    all_pairs: List[Tuple[str, str]],
    n_probes: int,
    rng: np.random.Generator,
    spectator_names: Optional[List[str]] = None,
    **_kwargs,
) -> List[Tuple[str, str]]:
    """Cycle through spectators, picking one pair per spectator per round."""
    if spectator_names is None:
        spectator_names = sorted(set(s for _, s in all_pairs))

    # Group pairs by spectator
    by_spec: Dict[str, List[Tuple[str, str]]] = {s: [] for s in spectator_names}
    for t, s in all_pairs:
        if s in by_spec:
            by_spec[s].append((t, s))

    # Shuffle within each spectator group for variety
    for s in spectator_names:
        arr = by_spec[s]
        rng.shuffle(arr)

    selected: List[Tuple[str, str]] = []
    cursors = {s: 0 for s in spectator_names}
    while len(selected) < min(n_probes, len(all_pairs)):
        for s in spectator_names:
            if len(selected) >= min(n_probes, len(all_pairs)):
                break
            pairs = by_spec[s]
            idx = cursors[s]
            if idx < len(pairs):
                selected.append(pairs[idx])
                cursors[s] = idx + 1
    return selected


def _select_ucb(
    all_pairs: List[Tuple[str, str]],
    n_probes: int,
    rng: np.random.Generator,
    variance_map: Optional[Dict[Tuple[str, str], float]] = None,
    **_kwargs,
) -> List[Tuple[str, str]]:
    """Select pairs with highest uncertainty (variance proxy) first.

    If no variance map is provided, assigns uniform initial variance so
    that each pair is explored equally; after each simulated probe, the
    selected pair's variance is halved (acquisition-style).
    """
    if variance_map is None:
        var_map = {p: 1.0 for p in all_pairs}
    else:
        var_map = dict(variance_map)

    selected: List[Tuple[str, str]] = []
    remaining = list(all_pairs)

    for _ in range(min(n_probes, len(all_pairs))):
        # Pick pair with highest variance (UCB-style: explore uncertain)
        # Add small random jitter for tie-breaking
        best_pair = max(
            remaining,
            key=lambda p: var_map.get(p, 0.0) + rng.uniform(0, 1e-8),
        )
        selected.append(best_pair)
        remaining.remove(best_pair)
        # After "measuring" this pair, reduce its variance
        var_map[best_pair] = var_map.get(best_pair, 1.0) * 0.5

    return selected


def _select_dpp(
    all_pairs: List[Tuple[str, str]],
    n_probes: int,
    rng: np.random.Generator,
    spectator_workloads: Optional[Dict[str, Any]] = None,
    **_kwargs,
) -> List[Tuple[str, str]]:
    """Select diverse pairs using a greedy DPP kernel on spectator resource vectors.

    Builds an RBF kernel over spectator resource vectors and greedily
    selects pairs that maximise the log-determinant of the selected
    sub-kernel (diversity objective).
    """
    if spectator_workloads is None or len(all_pairs) == 0:
        # Fall back to random if no spectator workloads provided
        return _select_random(all_pairs, n_probes, rng)

    # Build resource-vector matrix for all pairs
    pair_vectors: List[np.ndarray] = []
    for t, s in all_pairs:
        if s in spectator_workloads:
            vec = spectator_workloads[s].resource_vector.copy()
        else:
            vec = np.zeros(7)
        pair_vectors.append(vec)

    pair_vectors_arr = np.array(pair_vectors)
    n = len(all_pairs)

    # RBF kernel
    sigma = 1.0
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = pair_vectors_arr[i] - pair_vectors_arr[j]
            K[i, j] = np.exp(-np.dot(diff, diff) / sigma ** 2)

    epsilon = 1e-6
    selected_indices: List[int] = []
    remaining_indices = list(range(n))

    for _ in range(min(n_probes, n)):
        best_idx = -1
        best_gain = -np.inf

        for idx in remaining_indices:
            test_set = selected_indices + [idx]
            K_sub = K[np.ix_(test_set, test_set)] + epsilon * np.eye(len(test_set))
            gain = np.linalg.slogdet(K_sub)[1]
            # Tie-break with small random noise
            gain += rng.uniform(0, 1e-8)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [all_pairs[i] for i in selected_indices]


# ------------------------------------------------------------------ #
#  Core experiment runner                                             #
# ------------------------------------------------------------------ #

def run_probe_budget_experiment(
    n_samples: int = 300,
    n_repeats: int = 10,
    probe_counts: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run the probe budget experiment.

    For each probe-count budget and selection strategy:

    1. Build ground-truth tomography by running IRBS on *all*
       (target, spectator) pairs.
    2. Select a *budget*-limited subset of pairs using each strategy.
    3. Impute unmeasured cells as zero (no information) and construct
       a partial tomography.
    4. Compare the partial tomography against ground truth using
       L1/L2 reconstruction error, NDCG@k, Jaccard@k, Precision@k,
       Recall@k, and Kendall tau.

    Args:
        n_samples: samples per trial (default 300).
        n_repeats: independent repetitions per condition (default 10).
        probe_counts: list of probe budgets.  Defaults to
            ``[2, 4, 6, 8, 10, 12, 16, 20, 24, 32]``.
        seeds: base seeds for reproducibility.  Defaults to
            ``[42, 43, 44, 45]``.

    Returns:
        Dict with keys:

        ``"probe_df"``
            Per-repeat DataFrame with columns: n_probes, repeat,
            strategy, reconstruction_error_l1, reconstruction_error_l2,
            ranking_ndcg_k3, ranking_ndcg_k5, jaccard_k3, precision_k3,
            recall_k3, kendall_tau.

        ``"probe_summary_df"``
            Aggregated DataFrame: n_probes, strategy, mean_error_l1,
            mean_error_l2, mean_ndcg_k3, ci_lo, ci_hi.
    """
    if probe_counts is None:
        probe_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    if seeds is None:
        seeds = [42, 43, 44, 45]

    # Use a small set of targets and spectators for tractability
    all_targets = get_targets()
    all_spectators = get_spectators()
    devices = get_device_profiles()

    # Pick first 3 targets and first 5 spectators for the experiment
    target_names = list(all_targets.keys())[:3]
    spectator_names = list(all_spectators.keys())[:5]
    device = list(devices.values())[0]

    load = 0.7
    distance = "same_llc"
    regime = "structured"
    n_trials = 12  # per IRBS measurement

    # All possible (target, spectator) pairs
    all_pairs = [(t, s) for t in target_names for s in spectator_names]
    n_total_pairs = len(all_pairs)

    strategies = {
        "random": _select_random,
        "round_robin": _select_round_robin,
        "ucb": _select_ucb,
        "dpp": _select_dpp,
    }

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        rng_master = np.random.default_rng(seed)

        # ---- Build ground truth: full tomography for this seed ----
        ground_truth: Dict[Tuple[str, str], float] = {}
        ground_truth_variances: Dict[Tuple[str, str], float] = {}

        for t_name, s_name in all_pairs:
            target = all_targets[t_name]
            spectator = all_spectators[s_name]
            rng_gt = np.random.default_rng(
                rng_master.integers(0, 2**63)
            )
            result = run_irbs_condition(
                target, spectator, device, load, distance, regime,
                n_trials, n_samples, rng_gt,
            )
            ground_truth[(t_name, s_name)] = result["effects"]["delta_p99"]

            # Estimate variance from treatment vs control variability
            summary_df = result["summary_df"]
            ctrl_vals = summary_df.loc[
                summary_df["is_treatment"] == 0, "p99"
            ].values
            treat_vals = summary_df.loc[
                summary_df["is_treatment"] == 1, "p99"
            ].values
            if len(ctrl_vals) > 1 and len(treat_vals) > 1:
                ground_truth_variances[(t_name, s_name)] = float(
                    np.var(treat_vals) + np.var(ctrl_vals)
                )
            else:
                ground_truth_variances[(t_name, s_name)] = 1.0

        # Ground-truth score vector (one per spectator, averaged over targets)
        gt_per_spectator = np.zeros(len(spectator_names))
        for j, s_name in enumerate(spectator_names):
            vals = [ground_truth[(t, s_name)] for t in target_names
                    if (t, s_name) in ground_truth]
            gt_per_spectator[j] = np.mean(vals) if vals else 0.0

        # ---- Run each strategy x probe-count x repeat ----
        for n_probes in probe_counts:
            for rep in range(n_repeats):
                rng_rep = np.random.default_rng(
                    seed * 100000 + n_probes * 1000 + rep
                )

                for strategy_name, select_fn in strategies.items():
                    # Select pairs
                    selected_pairs = select_fn(
                        all_pairs=list(all_pairs),
                        n_probes=n_probes,
                        rng=np.random.default_rng(rng_rep.integers(0, 2**63)),
                        spectator_names=spectator_names,
                        variance_map=ground_truth_variances,
                        spectator_workloads=all_spectators,
                    )

                    # Build partial tomography (unmeasured = 0)
                    partial: Dict[Tuple[str, str], float] = {}
                    for pair in selected_pairs:
                        if pair in ground_truth:
                            # Add small measurement noise to simulate
                            # independent re-measurement
                            noise = rng_rep.normal(0, 0.01) * abs(
                                ground_truth[pair]
                            )
                            partial[pair] = ground_truth[pair] + noise

                    # Reconstruct: full vector (unmeasured pairs = 0)
                    gt_vector = np.array(
                        [ground_truth[p] for p in all_pairs]
                    )
                    est_vector = np.array(
                        [partial.get(p, 0.0) for p in all_pairs]
                    )

                    # Per-spectator estimated score (average over targets)
                    est_per_spectator = np.zeros(len(spectator_names))
                    for j, s_name in enumerate(spectator_names):
                        vals = [partial.get((t, s_name), 0.0)
                                for t in target_names]
                        est_per_spectator[j] = np.mean(vals) if vals else 0.0

                    # Reconstruction errors
                    l1_error = float(np.mean(np.abs(gt_vector - est_vector)))
                    l2_error = float(
                        np.sqrt(np.mean((gt_vector - est_vector) ** 2))
                    )

                    # Ranking metrics (on per-spectator scores)
                    ndcg_k3 = compute_ndcg(gt_per_spectator, est_per_spectator, k=3)
                    ndcg_k5 = compute_ndcg(gt_per_spectator, est_per_spectator, k=5)
                    jac_k3 = _jaccard_at_k(gt_per_spectator, est_per_spectator, k=3)
                    prec_k3 = _precision_at_k(gt_per_spectator, est_per_spectator, k=3)
                    rec_k3 = _recall_at_k(gt_per_spectator, est_per_spectator, k=3)
                    ktau = compute_kendall_tau(gt_per_spectator, est_per_spectator)

                    rows.append({
                        "n_probes": n_probes,
                        "repeat": rep + seed * n_repeats,
                        "strategy": strategy_name,
                        "reconstruction_error_l1": l1_error,
                        "reconstruction_error_l2": l2_error,
                        "ranking_ndcg_k3": ndcg_k3,
                        "ranking_ndcg_k5": ndcg_k5,
                        "jaccard_k3": jac_k3,
                        "precision_k3": prec_k3,
                        "recall_k3": rec_k3,
                        "kendall_tau": ktau,
                    })

    probe_df = pd.DataFrame(rows)

    # ---- Summary with bootstrap-style CIs on NDCG@3 ----
    summary_rows: List[Dict[str, Any]] = []
    for (n_probes, strategy), grp in probe_df.groupby(["n_probes", "strategy"]):
        vals = grp["ranking_ndcg_k3"].values
        mean_ndcg = float(np.mean(vals))
        # 95% CI via percentile bootstrap
        n_boot = 2000
        boot_rng = np.random.default_rng(42)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            sample = boot_rng.choice(vals, size=len(vals), replace=True)
            boot_means[b] = np.mean(sample)
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        summary_rows.append({
            "n_probes": n_probes,
            "strategy": strategy,
            "mean_error_l1": float(grp["reconstruction_error_l1"].mean()),
            "mean_error_l2": float(grp["reconstruction_error_l2"].mean()),
            "mean_ndcg_k3": mean_ndcg,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    probe_summary_df = pd.DataFrame(summary_rows)

    return {
        "probe_df": probe_df,
        "probe_summary_df": probe_summary_df,
    }

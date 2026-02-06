"""Main orchestrator: one-command reproducibility.

Usage:
    python -m sit.experiments.run_all --config sit/experiments/configs/quick.yaml
    python -m sit.experiments.run_all --config sit/experiments/configs/full.yaml

This single entry point:
1) Runs simulator experiments to produce raw traces
2) Computes IRBS effects
3) Builds tomography matrices with bootstrap uncertainty
4) Runs scheduling comparisons
5) Runs baseline mismatch experiments
6) Generates figures and tables
7) Generates workbook and report
8) Writes a figure manifest
"""

import argparse
import hashlib
import json
import time
import sys
import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sit.simulator.workloads import get_targets, get_spectators
from sit.simulator.device_profiles import get_device_profiles
from sit.simulator.drift import inject_synthetic_drift, compute_drift_series
from sit.simulator.latency_generator import generate_trial_samples
from sit.simulator.interference_channels import compute_cosine_similarity
from sit.simulator.validation import check_no_fixed_ratio, check_monotonicity
from sit.analysis.metrics import (
    compute_p99, compute_p95, compute_cvar99, compute_cvar95,
    compute_slo_violation_rate, compute_trial_summary, compute_ssi,
)
from sit.analysis.significance import bootstrap_mean_ci, bootstrap_difference_ci
from sit.experiments.run_experiments import (
    run_irbs_condition, run_naive_condition, run_scheduling_evaluation,
    run_drift_bias_demo, get_slo_threshold,
)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def phase1_irbs_and_tomography(config: Dict, all_results: Dict) -> Dict:
    """Phase 1: Run IRBS experiments and build tomography matrices."""
    print("\n" + "=" * 70)
    print("PHASE 1: IRBS Experiments & Tomography Construction")
    print("=" * 70)

    targets_dict = get_targets()
    spectators_dict = get_spectators()
    devices_dict = get_device_profiles()

    target_names = config["targets"]
    spectator_names = config["spectators"]
    device_names = config["devices"]
    distances = config["distances"]
    loads = config["loads"]
    regimes = config["regimes"]
    seeds = config["seeds"]
    n_trials = config["irbs"]["n_trials"]
    n_samples = config["irbs"]["n_samples"]

    # Build factorial grid
    total_conditions = (len(device_names) * len(target_names) * len(spectator_names) *
                        len(distances) * len(loads) * len(regimes) * len(seeds))
    print(f"Total conditions: {total_conditions}")
    print(f"Trials per condition: {n_trials}")
    print(f"Samples per trial: {n_samples}")
    print(f"Total trials: {total_conditions * n_trials}")

    irbs_results = {}  # (target, spectator, device, distance, load, regime, seed) -> result
    raw_records = []
    trial_count = 0
    sample_count = 0
    t0 = time.time()

    condition_idx = 0
    for seed in seeds:
        for d_name in device_names:
            device = devices_dict[d_name]
            for t_name in target_names:
                target = targets_dict[t_name]
                for s_name in spectator_names:
                    spectator = spectators_dict[s_name]
                    for dist in distances:
                        for load in loads:
                            for regime in regimes:
                                condition_idx += 1
                                if condition_idx % 100 == 0:
                                    elapsed = time.time() - t0
                                    rate = condition_idx / elapsed if elapsed > 0 else 0
                                    eta = (total_conditions - condition_idx) / rate if rate > 0 else 0
                                    print(f"  Condition {condition_idx}/{total_conditions} "
                                          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

                                rng = np.random.default_rng(seed + hash((d_name, t_name, s_name, dist)) % (2**31))

                                result = run_irbs_condition(
                                    target, spectator, device, load, dist, regime,
                                    n_trials, n_samples, rng,
                                )

                                key = (t_name, s_name, d_name, dist, load, regime, seed)
                                irbs_results[key] = result

                                trial_count += n_trials
                                sample_count += n_trials * n_samples

                                # Store summary record
                                for metric_name, metric_val in result["effects"].items():
                                    raw_records.append({
                                        "device": d_name,
                                        "target": t_name,
                                        "spectator": s_name,
                                        "distance": dist,
                                        "load": load,
                                        "regime": regime,
                                        "seed": seed,
                                        "metric": metric_name,
                                        "value": metric_val,
                                    })

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {condition_idx} conditions, {trial_count} trials, "
          f"{sample_count} samples in {elapsed:.1f}s")

    # Save raw records
    raw_df = pd.DataFrame(raw_records)
    Path(config["output"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(f"{config['output']['raw_dir']}/irbs_effects.parquet", index=False)

    # Also save trial-level summaries
    trial_records = []
    for key, result in irbs_results.items():
        t_name, s_name, d_name, dist, load, regime, seed = key
        for _, row in result["summary_df"].iterrows():
            rec = row.to_dict()
            rec.update({
                "device": d_name, "target": t_name, "spectator": s_name,
                "distance": dist, "load": load, "regime": regime, "seed": seed,
            })
            trial_records.append(rec)

    trial_df = pd.DataFrame(trial_records)
    trial_df.to_parquet(f"{config['output']['raw_dir']}/trial_summaries.parquet", index=False)

    all_results["irbs_results"] = irbs_results
    all_results["raw_effects_df"] = raw_df
    all_results["trial_df"] = trial_df
    all_results["trial_count"] = trial_count
    all_results["sample_count"] = sample_count
    all_results["n_conditions"] = condition_idx

    return all_results


def phase2_tomography_matrices(config: Dict, all_results: Dict) -> Dict:
    """Phase 2: Build tomography matrices with bootstrap uncertainty."""
    print("\n" + "=" * 70)
    print("PHASE 2: Tomography Matrix Construction")
    print("=" * 70)

    irbs_results = all_results["irbs_results"]
    target_names = config["targets"]
    spectator_names = config["spectators"]
    n_bootstrap = config["tomography"]["n_bootstrap"]

    # Build matrices for reference condition: first device, mid load, structured regime
    ref_device = config["devices"][0]
    ref_distance = config["distances"][len(config["distances"]) // 2]
    ref_load = config["loads"][len(config["loads"]) // 2]
    ref_regime = "structured"
    ref_seed = config["seeds"][0]

    # Mean effects matrix
    mean_matrix = pd.DataFrame(index=target_names, columns=spectator_names, dtype=float)
    ci_lower_matrix = pd.DataFrame(index=target_names, columns=spectator_names, dtype=float)
    ci_upper_matrix = pd.DataFrame(index=target_names, columns=spectator_names, dtype=float)
    stderr_matrix = pd.DataFrame(index=target_names, columns=spectator_names, dtype=float)
    ctrl_p99_values = {}

    for t_name in target_names:
        for s_name in spectator_names:
            # Aggregate across seeds for this condition
            effects_p99 = []
            ctrl_p99s = []
            for seed in config["seeds"]:
                key = (t_name, s_name, ref_device, ref_distance, ref_load, ref_regime, seed)
                if key in irbs_results:
                    effects_p99.append(irbs_results[key]["effects"]["delta_p99"])
                    ctrl_p99s.append(irbs_results[key]["effects"]["ctrl_p99"])

            if effects_p99:
                vals = np.array(effects_p99)
                rng_boot = np.random.default_rng(42 + hash((t_name, s_name)) % (2**31))
                mean_val, ci_lo, ci_hi = bootstrap_mean_ci(vals, n_bootstrap, rng=rng_boot)
                mean_matrix.loc[t_name, s_name] = mean_val
                ci_lower_matrix.loc[t_name, s_name] = ci_lo
                ci_upper_matrix.loc[t_name, s_name] = ci_hi
                stderr_matrix.loc[t_name, s_name] = np.std(vals) / np.sqrt(len(vals))
                ctrl_p99_values[t_name] = np.mean(ctrl_p99s)
            else:
                mean_matrix.loc[t_name, s_name] = 0.0
                ci_lower_matrix.loc[t_name, s_name] = 0.0
                ci_upper_matrix.loc[t_name, s_name] = 0.0
                stderr_matrix.loc[t_name, s_name] = 0.0

    # Sparsity statistics
    sparsity_rows = []
    for t_name in target_names:
        row_vals = mean_matrix.loc[t_name].astype(float).values
        total = np.sum(np.abs(row_vals))
        if total > 0:
            sorted_abs = np.sort(np.abs(row_vals))[::-1]
            top1_share = sorted_abs[0] / total if len(sorted_abs) > 0 else 0
            top3_share = np.sum(sorted_abs[:3]) / total if len(sorted_abs) >= 3 else 1.0
            top5_share = np.sum(sorted_abs[:5]) / total if len(sorted_abs) >= 5 else 1.0
        else:
            top1_share = top3_share = top5_share = 0.0

        sparsity_rows.append({
            "target": t_name,
            "top1_share": top1_share,
            "top3_share": top3_share,
            "top5_share": top5_share,
            "total_interference": total,
        })

    sparsity_df = pd.DataFrame(sparsity_rows)

    # Save
    Path(config["output"]["derived_dir"]).mkdir(parents=True, exist_ok=True)
    mean_matrix.to_csv(f"{config['output']['derived_dir']}/tomography_mean.csv")
    ci_lower_matrix.to_csv(f"{config['output']['derived_dir']}/tomography_ci_lower.csv")
    ci_upper_matrix.to_csv(f"{config['output']['derived_dir']}/tomography_ci_upper.csv")
    stderr_matrix.to_csv(f"{config['output']['derived_dir']}/tomography_stderr.csv")
    sparsity_df.to_csv(f"{config['output']['derived_dir']}/sparsity_stats.csv", index=False)

    print(f"Tomography matrix: {len(target_names)} x {len(spectator_names)}")
    print(f"Sparsity: mean top-3 share = {sparsity_df['top3_share'].mean():.2%}")

    all_results["tomo_mean"] = mean_matrix
    all_results["tomo_ci_lower"] = ci_lower_matrix
    all_results["tomo_ci_upper"] = ci_upper_matrix
    all_results["tomo_stderr"] = stderr_matrix
    all_results["sparsity_df"] = sparsity_df
    all_results["ctrl_p99_values"] = ctrl_p99_values

    return all_results


def phase3_sparse_recovery(config: Dict, all_results: Dict) -> Dict:
    """Phase 3: Sparse recovery curves."""
    print("\n" + "=" * 70)
    print("PHASE 3: Sparse Recovery Analysis")
    print("=" * 70)

    targets_dict = get_targets()
    spectators_dict = get_spectators()
    devices_dict = get_device_profiles()

    target_name = config["targets"][0]
    target = targets_dict[target_name]
    device = devices_dict[config["devices"][0]]
    distance = config["distances"][len(config["distances"]) // 2]
    load = config["loads"][len(config["loads"]) // 2]
    regime = "structured"

    trial_counts = config["tomography"]["recovery_trial_counts"]
    k_values = config["tomography"]["k_values"]
    n_repeats = config["tomography"]["recovery_repeats"]
    n_samples = config["irbs"]["n_samples"]
    max_trials = config["tomography"]["recovery_max_trials"]

    spectator_names = config["spectators"]
    spectators = [spectators_dict[s] for s in spectator_names]

    # Ground truth: run with max trials
    rng_truth = np.random.default_rng(42)
    truth_effects = {}
    for s_name in spectator_names:
        spec = spectators_dict[s_name]
        result = run_irbs_condition(
            target, spec, device, load, distance, regime,
            max_trials, n_samples, np.random.default_rng(rng_truth.integers(0, 2**63)),
        )
        truth_effects[s_name] = result["effects"]["delta_p99"]

    # True rankings
    sorted_truth = sorted(truth_effects.items(), key=lambda x: -x[1])
    true_top = {k: set(name for name, _ in sorted_truth[:k]) for k in k_values}

    recovery_rows = []
    for n_trial in trial_counts:
        for k in k_values:
            recoveries = 0
            for rep in range(n_repeats):
                rng_rep = np.random.default_rng(1000 + rep + n_trial * 100)
                rep_effects = {}
                for s_name in spectator_names:
                    spec = spectators_dict[s_name]
                    result = run_irbs_condition(
                        target, spec, device, load, distance, regime,
                        n_trial, n_samples, np.random.default_rng(rng_rep.integers(0, 2**63)),
                    )
                    rep_effects[s_name] = result["effects"]["delta_p99"]

                sorted_rep = sorted(rep_effects.items(), key=lambda x: -x[1])
                rep_top_k = set(name for name, _ in sorted_rep[:k])
                if rep_top_k == true_top[k]:
                    recoveries += 1

            recovery_rows.append({
                "trials_per_spectator": n_trial,
                "k": k,
                "recovery_probability": recoveries / n_repeats,
            })
            print(f"  trials={n_trial}, k={k}: recovery={recoveries}/{n_repeats}")

    recovery_df = pd.DataFrame(recovery_rows)
    recovery_df.to_csv(f"{config['output']['derived_dir']}/recovery_curve.csv", index=False)

    all_results["recovery_df"] = recovery_df
    return all_results


def phase4_scheduling(config: Dict, all_results: Dict) -> Dict:
    """Phase 4: Scheduling comparison."""
    print("\n" + "=" * 70)
    print("PHASE 4: Scheduling Evaluation")
    print("=" * 70)

    targets_dict = get_targets()
    spectators_dict = get_spectators()
    devices_dict = get_device_profiles()

    tomo_mean = all_results["tomo_mean"]
    tomo_stderr = all_results["tomo_stderr"]

    n_slots = config["scheduling"]["n_slots"]
    beta_ucb = config["scheduling"]["beta_ucb"]

    # Build similarity kernel
    all_workloads = list(spectators_dict.values())
    wl_names = [w.name for w in all_workloads]
    n_wl = len(all_workloads)
    K = np.zeros((n_wl, n_wl))
    sigma = 1.0
    for i in range(n_wl):
        for j in range(n_wl):
            diff = all_workloads[i].resource_vector - all_workloads[j].resource_vector
            K[i, j] = np.exp(-np.dot(diff, diff) / sigma ** 2)

    sched_results = []
    condition_count = 0

    for seed in config["seeds"]:
        for d_name in config["devices"]:
            device = devices_dict[d_name]
            for t_name in config["targets"]:
                target = targets_dict[t_name]
                for dist in config["distances"]:
                    for load in config["loads"]:
                        for regime in config["regimes"]:
                            condition_count += 1
                            rng = np.random.default_rng(seed + condition_count)
                            candidate_specs = config["spectators"]

                            schedulers = {}

                            # 1. Random
                            rng_rand = np.random.default_rng(rng.integers(0, 2**63))
                            rand_sel = list(rng_rand.choice(candidate_specs, size=min(n_slots, len(candidate_specs)), replace=False))
                            schedulers["random"] = (rand_sel, False, False)

                            # 2. Mean greedy
                            if t_name in tomo_mean.index:
                                spec_scores = []
                                for s in candidate_specs:
                                    if s in tomo_mean.columns:
                                        score = float(tomo_mean.loc[t_name, s])
                                    else:
                                        score = 0.0
                                    spec_scores.append((s, score))
                                spec_scores.sort(key=lambda x: x[1])
                                mean_greedy_sel = [s for s, _ in spec_scores[:n_slots]]
                            else:
                                mean_greedy_sel = rand_sel
                            schedulers["mean_greedy"] = (mean_greedy_sel, False, False)

                            # 3. Similarity avoidance
                            sim_scores = []
                            for s in candidate_specs:
                                if s in spectators_dict:
                                    sim = compute_cosine_similarity(target, spectators_dict[s])
                                    sim_scores.append((s, sim))
                            sim_scores.sort(key=lambda x: x[1])
                            sim_avoid_sel = [s for s, _ in sim_scores[:n_slots]]
                            schedulers["similarity_avoidance"] = (sim_avoid_sel, False, False)

                            # 4. Linux proxy (simplified: random from low-CPU workloads)
                            cpu_scores = [(s, spectators_dict[s].channel_pressure[5] + spectators_dict[s].channel_pressure[6])
                                         for s in candidate_specs if s in spectators_dict]
                            cpu_scores.sort(key=lambda x: x[1])
                            linux_sel = [s for s, _ in cpu_scores[:n_slots]]
                            schedulers["linux_proxy"] = (linux_sel, False, False)

                            # 5. Static partition
                            schedulers["static_partition"] = (rand_sel, True, False)

                            # 6. SIT-DPP (risk + diversity)
                            if t_name in tomo_mean.index:
                                sit_candidates = []
                                for s in candidate_specs:
                                    if s in tomo_mean.columns and s in wl_names:
                                        risk = float(tomo_mean.loc[t_name, s])
                                        s_idx = wl_names.index(s)
                                        sit_candidates.append((s, risk, s_idx))

                                # Greedy DPP selection
                                selected = []
                                selected_idx = []
                                remaining = list(sit_candidates)
                                epsilon = 1e-6
                                for _ in range(min(n_slots, len(remaining))):
                                    best_score = -np.inf
                                    best_item = None
                                    for s, risk, s_idx in remaining:
                                        # Diversity gain
                                        test_idx = selected_idx + [s_idx]
                                        K_sub = K[np.ix_(test_idx, test_idx)] + epsilon * np.eye(len(test_idx))
                                        div_gain = np.linalg.slogdet(K_sub)[1]
                                        # Risk penalty (lower is better)
                                        score = div_gain - 1.0 * risk
                                        if score > best_score:
                                            best_score = score
                                            best_item = (s, risk, s_idx)
                                    if best_item:
                                        selected.append(best_item[0])
                                        selected_idx.append(best_item[2])
                                        remaining = [x for x in remaining if x[0] != best_item[0]]
                                sit_dpp_sel = selected if selected else rand_sel
                            else:
                                sit_dpp_sel = rand_sel
                            schedulers["sit_dpp"] = (sit_dpp_sel, False, False)

                            # 7. SIT-UCB-DPP
                            if t_name in tomo_mean.index and t_name in tomo_stderr.index:
                                sit_ucb_candidates = []
                                for s in candidate_specs:
                                    if s in tomo_mean.columns and s in tomo_stderr.columns and s in wl_names:
                                        mean_risk = float(tomo_mean.loc[t_name, s])
                                        stderr = float(tomo_stderr.loc[t_name, s])
                                        ucb_risk = mean_risk + beta_ucb * stderr
                                        s_idx = wl_names.index(s)
                                        sit_ucb_candidates.append((s, ucb_risk, s_idx))

                                selected = []
                                selected_idx = []
                                remaining = list(sit_ucb_candidates)
                                for _ in range(min(n_slots, len(remaining))):
                                    best_score = -np.inf
                                    best_item = None
                                    for s, risk, s_idx in remaining:
                                        test_idx = selected_idx + [s_idx]
                                        K_sub = K[np.ix_(test_idx, test_idx)] + epsilon * np.eye(len(test_idx))
                                        div_gain = np.linalg.slogdet(K_sub)[1]
                                        score = div_gain - 1.0 * risk
                                        if score > best_score:
                                            best_score = score
                                            best_item = (s, risk, s_idx)
                                    if best_item:
                                        selected.append(best_item[0])
                                        selected_idx.append(best_item[2])
                                        remaining = [x for x in remaining if x[0] != best_item[0]]
                                sit_ucb_sel = selected if selected else rand_sel
                            else:
                                sit_ucb_sel = rand_sel
                            schedulers["sit_ucb_dpp"] = (sit_ucb_sel, False, False)

                            # Evaluate each scheduler
                            for sched_name, (sel, static_part, triton) in schedulers.items():
                                eval_rng = np.random.default_rng(rng.integers(0, 2**63))
                                result = run_scheduling_evaluation(
                                    target, spectators_dict, device, load, dist, regime,
                                    config["irbs"]["n_samples"], eval_rng,
                                    sel, sched_name,
                                    apply_static_partition=static_part,
                                    triton_batching=triton,
                                )
                                result["seed"] = seed
                                sched_results.append(result)

                            if condition_count % 50 == 0:
                                print(f"  Scheduling condition {condition_count}...")

    sched_df = pd.DataFrame(sched_results)
    sched_df.to_parquet(f"{config['output']['raw_dir']}/scheduling_results.parquet", index=False)

    print(f"Phase 4 complete: {len(sched_df)} scheduling evaluations")

    all_results["sched_results"] = sched_df
    return all_results


def phase5_baseline_mismatch(config: Dict, all_results: Dict) -> Dict:
    """Phase 5: Baseline mismatch analysis."""
    print("\n" + "=" * 70)
    print("PHASE 5: Baseline Mismatch Analysis")
    print("=" * 70)

    from sit.analysis.baseline_mismatch import NaivePredictor, compute_mismatch_metrics

    tomo_mean = all_results["tomo_mean"]
    ctrl_p99_values = all_results.get("ctrl_p99_values", {})

    targets_dict = get_targets()
    spectators_dict = get_spectators()

    predictor = NaivePredictor(c=1.5, q=1.2)

    scatter_rows = []
    for t_name in tomo_mean.index:
        if t_name not in targets_dict:
            continue
        for s_name in tomo_mean.columns:
            if s_name not in spectators_dict:
                continue
            observed = float(tomo_mean.loc[t_name, s_name])
            p99_ctrl = ctrl_p99_values.get(t_name, targets_dict[t_name].base_latency_us * 3.0)

            for load in config["loads"]:
                for dist in config["distances"]:
                    predicted_p99 = predictor.predict(
                        p99_ctrl, targets_dict[t_name], spectators_dict[s_name], load, dist
                    )
                    predicted_delta = predicted_p99 - p99_ctrl

                    scatter_rows.append({
                        "target": t_name,
                        "spectator": s_name,
                        "load": load,
                        "distance": dist,
                        "observed": observed,
                        "predicted": predicted_delta,
                    })

    scatter_df = pd.DataFrame(scatter_rows)

    if len(scatter_df) > 0:
        mismatch_metrics = compute_mismatch_metrics(
            scatter_df["observed"].values,
            scatter_df["predicted"].values,
        )
    else:
        mismatch_metrics = {}

    scatter_df.to_csv(f"{config['output']['derived_dir']}/mismatch_scatter.csv", index=False)
    print(f"Mismatch analysis: {len(scatter_df)} points")
    if mismatch_metrics:
        print(f"  Underprediction rate: {mismatch_metrics.get('underprediction_rate', 0):.1%}")
        print(f"  High-risk underprediction: {mismatch_metrics.get('underprediction_rate_high_risk', 0):.1%}")

    all_results["mismatch_scatter"] = scatter_df
    all_results["mismatch_metrics"] = mismatch_metrics
    return all_results


def phase6_drift_bias_demo(config: Dict, all_results: Dict) -> Dict:
    """Phase 6: IRBS drift bias demonstration."""
    print("\n" + "=" * 70)
    print("PHASE 6: IRBS Drift Bias Demonstration")
    print("=" * 70)

    targets_dict = get_targets()
    spectators_dict = get_spectators()
    devices_dict = get_device_profiles()

    target = targets_dict[config["targets"][0]]
    spectator = spectators_dict[config["spectators"][0]]
    device = devices_dict[config["devices"][0]]

    bias_df = run_drift_bias_demo(
        target, spectator, device,
        load=0.7,
        distance=config["distances"][len(config["distances"]) // 2],
        regime="structured",
        n_trials=config["irbs"]["n_trials"],
        n_samples=config["irbs"]["n_samples"],
        n_repeats=config["irbs"]["drift_bias_repeats"],
    )

    bias_df.to_csv(f"{config['output']['derived_dir']}/drift_bias_demo.csv", index=False)

    naive_bias = bias_df["naive_tau"].mean() - bias_df["true_tau"].iloc[0]
    irbs_bias = bias_df["irbs_tau"].mean() - bias_df["true_tau"].iloc[0]
    print(f"Naive bias: {naive_bias:.2f} μs")
    print(f"IRBS bias: {irbs_bias:.2f} μs")
    print(f"Bias reduction: {abs(naive_bias) - abs(irbs_bias):.2f} μs")

    all_results["bias_df"] = bias_df
    return all_results


def phase7_ablations(config: Dict, all_results: Dict) -> Dict:
    """Phase 7: Ablation study."""
    print("\n" + "=" * 70)
    print("PHASE 7: Ablation Study")
    print("=" * 70)

    sched_df = all_results["sched_results"]

    # Ablation variants from the scheduling results
    variants = []

    # Full SIT-DPP
    sit_data = sched_df[sched_df["scheduler"] == "sit_dpp"]
    if len(sit_data) > 0:
        variants.append({
            "variant": "sit_dpp",
            "p99": sit_data["p99"].mean(),
            "cvar99": sit_data["cvar99"].mean(),
            "mean": sit_data["mean"].mean(),
            "description": "Full SIT-DPP (risk + diversity)",
        })

    # SIT-UCB-DPP
    ucb_data = sched_df[sched_df["scheduler"] == "sit_ucb_dpp"]
    if len(ucb_data) > 0:
        variants.append({
            "variant": "sit_ucb_dpp",
            "p99": ucb_data["p99"].mean(),
            "cvar99": ucb_data["cvar99"].mean(),
            "mean": ucb_data["mean"].mean(),
            "description": "SIT with UCB uncertainty",
        })

    # No DPP (mean_greedy = risk only, no diversity)
    mg_data = sched_df[sched_df["scheduler"] == "mean_greedy"]
    if len(mg_data) > 0:
        variants.append({
            "variant": "no_dpp_risk_only",
            "p99": mg_data["p99"].mean(),
            "cvar99": mg_data["cvar99"].mean(),
            "mean": mg_data["mean"].mean(),
            "description": "Risk-only (no DPP diversity term)",
        })

    # No risk (similarity avoidance = diversity only, no risk)
    sa_data = sched_df[sched_df["scheduler"] == "similarity_avoidance"]
    if len(sa_data) > 0:
        variants.append({
            "variant": "no_risk_diversity_only",
            "p99": sa_data["p99"].mean(),
            "cvar99": sa_data["cvar99"].mean(),
            "mean": sa_data["mean"].mean(),
            "description": "Diversity-only (no risk term)",
        })

    # Random
    rand_data = sched_df[sched_df["scheduler"] == "random"]
    if len(rand_data) > 0:
        variants.append({
            "variant": "random_baseline",
            "p99": rand_data["p99"].mean(),
            "cvar99": rand_data["cvar99"].mean(),
            "mean": rand_data["mean"].mean(),
            "description": "Random placement",
        })

    # Static partition
    sp_data = sched_df[sched_df["scheduler"] == "static_partition"]
    if len(sp_data) > 0:
        variants.append({
            "variant": "static_partition",
            "p99": sp_data["p99"].mean(),
            "cvar99": sp_data["cvar99"].mean(),
            "mean": sp_data["mean"].mean(),
            "description": "Static resource partitioning",
        })

    ablation_df = pd.DataFrame(variants)
    ablation_df.to_csv(f"{config['output']['derived_dir']}/ablations.csv", index=False)

    all_results["ablation_df"] = ablation_df

    print(f"Ablation variants: {len(variants)}")
    for v in variants:
        print(f"  {v['variant']}: p99={v['p99']:.1f}, CVaR99={v['cvar99']:.1f}")

    return all_results


def phase8_worst_case(config: Dict, all_results: Dict) -> Dict:
    """Phase 8: Worst-case analysis."""
    print("\n" + "=" * 70)
    print("PHASE 8: Worst-Case Analysis")
    print("=" * 70)

    from sit.analysis.worst_case import (
        identify_hardest_conditions,
        evaluate_on_hardest,
        compute_blowup_avoidance,
    )

    sched_df = all_results["sched_results"]

    hard_conditions = identify_hardest_conditions(
        sched_df, scheduler_name="random", metric="p99",
        quantile=0.90, regime_filter="adversarial",
    )

    if len(hard_conditions) == 0:
        # Fallback: try without regime filter
        hard_conditions = identify_hardest_conditions(
            sched_df, scheduler_name="random", metric="p99",
            quantile=0.90, regime_filter=None,
        )

    if len(hard_conditions) > 0:
        worst_case_df = evaluate_on_hardest(sched_df, hard_conditions)
        blowup = compute_blowup_avoidance(
            sched_df, hard_conditions,
            sit_scheduler="sit_dpp", baseline_scheduler="random",
        )
        print(f"Hardest conditions: {len(hard_conditions)}")
        if "reduction_pct" in blowup:
            print(f"SIT p99 reduction on hardest: {blowup['reduction_pct']:.1f}%")
    else:
        worst_case_df = pd.DataFrame()
        blowup = {}
        print("Warning: Could not identify hardest conditions")

    all_results["worst_case_df"] = worst_case_df
    all_results["worst_case_summary"] = blowup
    return all_results


def phase9_qa_checks(config: Dict, all_results: Dict) -> Dict:
    """Phase 9: QA checks."""
    print("\n" + "=" * 70)
    print("PHASE 9: QA Checks")
    print("=" * 70)

    qa_results = {}

    sched_df = all_results.get("sched_results", pd.DataFrame())

    # 1. Fixed ratio check
    if len(sched_df) > 0 and all(c in sched_df.columns for c in ["mean", "p99", "cvar99"]):
        passed, msg = check_no_fixed_ratio(
            sched_df["mean"].values,
            sched_df["p99"].values,
            sched_df["cvar99"].values,
        )
        qa_results["fixed_ratio"] = {"passed": passed, "message": msg}
        print(f"  Fixed ratio check: {'PASS' if passed else 'FAIL'} - {msg}")
    else:
        qa_results["fixed_ratio"] = {"passed": True, "message": "Insufficient data"}

    # 2. Monotonicity check
    if len(sched_df) > 0 and "load" in sched_df.columns and "p99" in sched_df.columns:
        adv = sched_df[sched_df["regime"] == "adversarial"] if "regime" in sched_df.columns else sched_df
        if len(adv) > 0:
            load_means = adv.groupby("load")["p99"].mean()
            passed, vr, msg = check_monotonicity(
                load_means.index.values.astype(float),
                load_means.values,
            )
            qa_results["monotonicity_load_p99"] = {"passed": passed, "message": msg, "violation_rate": vr}
            print(f"  Monotonicity (load vs p99): {'PASS' if passed else 'FAIL'} - {msg}")
        else:
            qa_results["monotonicity_load_p99"] = {"passed": True, "message": "No adversarial data"}
    else:
        qa_results["monotonicity_load_p99"] = {"passed": True, "message": "Insufficient data"}

    # 3. Seed reproducibility (spot check)
    from sit.simulator.latency_generator import generate_trial_samples
    from sit.simulator.workloads import get_targets
    from sit.simulator.device_profiles import get_device_profiles
    targets = get_targets()
    devices = get_device_profiles()
    t = list(targets.values())[0]
    d = list(devices.values())[0]
    s1 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100,
                                np.random.default_rng(12345))
    s2 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100,
                                np.random.default_rng(12345))
    seed_pass = np.allclose(s1, s2)
    qa_results["seed_reproducibility"] = {
        "passed": seed_pass,
        "message": "Same seed produces identical samples" if seed_pass else "SEED MISMATCH"
    }
    print(f"  Seed reproducibility: {'PASS' if seed_pass else 'FAIL'}")

    # 4. CVaR >= p99 check
    if len(sched_df) > 0 and "p99" in sched_df.columns and "cvar99" in sched_df.columns:
        cvar_ge_p99 = (sched_df["cvar99"] >= sched_df["p99"] - 1e-6).all()
        qa_results["cvar_ge_p99"] = {
            "passed": bool(cvar_ge_p99),
            "message": "CVaR99 >= p99 for all rows" if cvar_ge_p99 else "CVaR99 < p99 found!"
        }
        print(f"  CVaR99 >= p99: {'PASS' if cvar_ge_p99 else 'FAIL'}")
    else:
        qa_results["cvar_ge_p99"] = {"passed": True, "message": "Insufficient data"}

    all_results["qa_results"] = qa_results
    return all_results


def phase10_figures_and_tables(config: Dict, all_results: Dict) -> Dict:
    """Phase 10: Generate all figures and tables."""
    print("\n" + "=" * 70)
    print("PHASE 10: Generating Figures and Tables")
    print("=" * 70)

    from sit.analysis.figures import generate_all_figures
    from sit.analysis.tables import generate_all_tables

    # Build phenomenon DataFrame for F2
    sched_df = all_results.get("sched_results", pd.DataFrame())
    if len(sched_df) > 0:
        all_results["phenomenon_df"] = sched_df
        all_results["phenomenon_target"] = config["targets"][0] if config["targets"] else "kv_lookup"
        all_results["phenomenon_spectator"] = config["spectators"][0] if config["spectators"] else "cache_thrash"

    # Summary for scheduler comparison
    if len(sched_df) > 0:
        sched_summary = sched_df.groupby(["scheduler", "regime"]).agg({
            "mean": "mean", "p95": "mean", "p99": "mean", "cvar99": "mean",
        }).reset_index()
        all_results["sched_summary"] = sched_summary

    figures_dir = config["output"]["figures_dir"]
    tables_dir = config["output"]["tables_dir"]

    generate_all_figures(all_results, figures_dir)
    generate_all_tables(all_results, tables_dir)

    print(f"Figures saved to {figures_dir}/")
    print(f"Tables saved to {tables_dir}/")

    return all_results


def phase11_workbook(config: Dict, all_results: Dict) -> Dict:
    """Phase 11: Generate Excel workbook."""
    print("\n" + "=" * 70)
    print("PHASE 11: Generating Results Workbook")
    print("=" * 70)

    try:
        from sit.artifacts.make_workbook import create_workbook
        wb_path = create_workbook(config, all_results)
        print(f"Workbook saved to {wb_path}")
    except Exception as e:
        print(f"Warning: Workbook generation failed: {e}")
        import traceback
        traceback.print_exc()

    return all_results


def phase12_report(config: Dict, all_results: Dict) -> Dict:
    """Phase 12: Generate report."""
    print("\n" + "=" * 70)
    print("PHASE 12: Generating Report")
    print("=" * 70)

    try:
        from sit.artifacts.make_report import create_report
        report_path = create_report(config, all_results)
        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    return all_results


def phase13_manifest(config: Dict, all_results: Dict) -> Dict:
    """Phase 13: Generate figure manifest."""
    print("\n" + "=" * 70)
    print("PHASE 13: Generating Figure Manifest")
    print("=" * 70)

    try:
        from sit.artifacts.figure_manifest import create_manifest
        manifest_path = create_manifest(config, all_results)
        print(f"Manifest saved to {manifest_path}")
    except Exception as e:
        print(f"Warning: Manifest generation failed: {e}")
        import traceback
        traceback.print_exc()

    return all_results


def print_final_summary(config: Dict, all_results: Dict):
    """Print final console summary."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Mode: {config['experiment']['mode']}")
    print(f"Total conditions: {all_results.get('n_conditions', 'N/A')}")
    print(f"Total trials: {all_results.get('trial_count', 'N/A')}")
    print(f"Total raw samples: {all_results.get('sample_count', 'N/A')}")

    sched_df = all_results.get("sched_results", pd.DataFrame())
    if len(sched_df) > 0:
        print("\nTop 3 Headline Results (Structured + Adversarial):")
        high_risk = sched_df[sched_df["regime"].isin(["structured", "adversarial"])]
        if len(high_risk) > 0:
            for metric in ["p99", "cvar99"]:
                if metric not in high_risk.columns:
                    continue
                sit_vals = high_risk[high_risk["scheduler"] == "sit_dpp"][metric]
                rand_vals = high_risk[high_risk["scheduler"] == "random"][metric]
                if len(sit_vals) > 0 and len(rand_vals) > 0:
                    rng = np.random.default_rng(42)
                    sit_mean, sit_lo, sit_hi = bootstrap_mean_ci(sit_vals.values, 2000, rng=rng)
                    rand_mean, rand_lo, rand_hi = bootstrap_mean_ci(rand_vals.values, 2000, rng=rng)
                    reduction = (rand_mean - sit_mean) / rand_mean * 100 if rand_mean > 0 else 0
                    print(f"  {metric}: SIT-DPP={sit_mean:.1f} [{sit_lo:.1f}, {sit_hi:.1f}] "
                          f"vs Random={rand_mean:.1f} [{rand_lo:.1f}, {rand_hi:.1f}] "
                          f"({reduction:.1f}% reduction)")

    qa = all_results.get("qa_results", {})
    all_pass = all(v.get("passed", True) for v in qa.values())
    print(f"\nQA Tests: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    for name, result in qa.items():
        status = "PASS" if result.get("passed", True) else "FAIL"
        print(f"  {name}: {status}")

    print("\nAll results in: results/")
    print("Raw data in: data/")


def main():
    parser = argparse.ArgumentParser(description="SIT: Run all experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    all_results = {}

    t_start = time.time()

    # Execute all phases
    all_results = phase1_irbs_and_tomography(config, all_results)
    all_results = phase2_tomography_matrices(config, all_results)
    all_results = phase3_sparse_recovery(config, all_results)
    all_results = phase4_scheduling(config, all_results)
    all_results = phase5_baseline_mismatch(config, all_results)
    all_results = phase6_drift_bias_demo(config, all_results)
    all_results = phase7_ablations(config, all_results)
    all_results = phase8_worst_case(config, all_results)
    all_results = phase9_qa_checks(config, all_results)
    all_results = phase10_figures_and_tables(config, all_results)
    all_results = phase11_workbook(config, all_results)
    all_results = phase12_report(config, all_results)
    all_results = phase13_manifest(config, all_results)

    t_end = time.time()
    print(f"\nTotal runtime: {t_end - t_start:.1f}s")

    print_final_summary(config, all_results)


if __name__ == "__main__":
    main()

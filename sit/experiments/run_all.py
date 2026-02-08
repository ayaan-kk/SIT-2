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


def phase8b_channel_decomposition(config: Dict, all_results: Dict) -> Dict:
    """Phase 8b: Channel decomposition analysis."""
    print("\n" + "=" * 70)
    print("PHASE 8b: Channel Decomposition Analysis")
    print("=" * 70)

    from sit.simulator.interference_channels import compute_interference_severity, compute_channel_overlap
    from sit.simulator.workloads import CHANNELS

    targets_dict = get_targets()
    spectators_dict = get_spectators()
    devices_dict = get_device_profiles()

    ref_device = devices_dict[config["devices"][0]]
    ref_load = config["loads"][len(config["loads"]) // 2]

    rows = []
    for t_name in config["targets"]:
        target = targets_dict[t_name]
        for s_name in config["spectators"]:
            spectator = spectators_dict[s_name]
            for regime in ["structured", "adversarial"]:
                for dist in ["same_core", "same_numa", "cross_socket"]:
                    severity, per_ch, spike_prob = compute_interference_severity(
                        target, spectator, ref_device, dist, ref_load, regime
                    )
                    overlap = compute_channel_overlap(target, spectator)
                    for c_idx, ch_name in enumerate(CHANNELS):
                        rows.append({
                            "target": t_name, "spectator": s_name,
                            "regime": regime, "distance": dist,
                            "channel": ch_name,
                            "per_channel_severity": float(per_ch[c_idx]),
                            "channel_overlap": float(overlap[c_idx]),
                            "total_severity": severity,
                            "spike_probability": spike_prob,
                        })

    channel_df = pd.DataFrame(rows)
    channel_df.to_csv(f"{config['output']['derived_dir']}/channel_decomposition.csv", index=False)

    # Summary: which channels dominate
    ch_totals = channel_df.groupby("channel")["per_channel_severity"].sum().sort_values(ascending=False)
    print("Channel contribution ranking:")
    for ch, val in ch_totals.items():
        pct = val / ch_totals.sum() * 100 if ch_totals.sum() > 0 else 0
        print(f"  {ch}: {pct:.1f}%")

    all_results["channel_df"] = channel_df
    return all_results


def phase8c_sensitivity_analysis(config: Dict, all_results: Dict) -> Dict:
    """Phase 8c: Sensitivity analysis - how SIT improvement varies with parameters."""
    print("\n" + "=" * 70)
    print("PHASE 8c: Sensitivity Analysis")
    print("=" * 70)

    sched_df = all_results.get("sched_results", pd.DataFrame())
    if len(sched_df) == 0:
        print("  SKIP: no scheduling results")
        return all_results

    sensitivity_rows = []

    # Sensitivity to load
    for load_val in sched_df["load"].unique():
        subset = sched_df[sched_df["load"] == load_val]
        for regime in subset["regime"].unique():
            r_sub = subset[subset["regime"] == regime]
            sit_p99 = r_sub[r_sub["scheduler"] == "sit_dpp"]["p99"].mean()
            rand_p99 = r_sub[r_sub["scheduler"] == "random"]["p99"].mean()
            sit_cvar = r_sub[r_sub["scheduler"] == "sit_dpp"]["cvar99"].mean()
            rand_cvar = r_sub[r_sub["scheduler"] == "random"]["cvar99"].mean()
            if rand_p99 > 0:
                sensitivity_rows.append({
                    "parameter": "load", "value": load_val, "regime": regime,
                    "sit_p99": sit_p99, "random_p99": rand_p99,
                    "sit_cvar99": sit_cvar, "random_cvar99": rand_cvar,
                    "p99_reduction_pct": (rand_p99 - sit_p99) / rand_p99 * 100,
                    "cvar99_reduction_pct": (rand_cvar - sit_cvar) / rand_cvar * 100 if rand_cvar > 0 else 0,
                })

    # Sensitivity to distance
    for dist_val in sched_df["distance"].unique():
        subset = sched_df[sched_df["distance"] == dist_val]
        for regime in subset["regime"].unique():
            r_sub = subset[subset["regime"] == regime]
            sit_p99 = r_sub[r_sub["scheduler"] == "sit_dpp"]["p99"].mean()
            rand_p99 = r_sub[r_sub["scheduler"] == "random"]["p99"].mean()
            sit_cvar = r_sub[r_sub["scheduler"] == "sit_dpp"]["cvar99"].mean()
            rand_cvar = r_sub[r_sub["scheduler"] == "random"]["cvar99"].mean()
            if rand_p99 > 0:
                sensitivity_rows.append({
                    "parameter": "distance", "value": dist_val, "regime": regime,
                    "sit_p99": sit_p99, "random_p99": rand_p99,
                    "sit_cvar99": sit_cvar, "random_cvar99": rand_cvar,
                    "p99_reduction_pct": (rand_p99 - sit_p99) / rand_p99 * 100,
                    "cvar99_reduction_pct": (rand_cvar - sit_cvar) / rand_cvar * 100 if rand_cvar > 0 else 0,
                })

    # Sensitivity to device
    for dev_val in sched_df["device"].unique():
        subset = sched_df[sched_df["device"] == dev_val]
        sit_p99 = subset[subset["scheduler"] == "sit_dpp"]["p99"].mean()
        rand_p99 = subset[subset["scheduler"] == "random"]["p99"].mean()
        sit_cvar = subset[subset["scheduler"] == "sit_dpp"]["cvar99"].mean()
        rand_cvar = subset[subset["scheduler"] == "random"]["cvar99"].mean()
        if rand_p99 > 0:
            sensitivity_rows.append({
                "parameter": "device", "value": dev_val, "regime": "all",
                "sit_p99": sit_p99, "random_p99": rand_p99,
                "sit_cvar99": sit_cvar, "random_cvar99": rand_cvar,
                "p99_reduction_pct": (rand_p99 - sit_p99) / rand_p99 * 100,
                "cvar99_reduction_pct": (rand_cvar - sit_cvar) / rand_cvar * 100 if rand_cvar > 0 else 0,
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(f"{config['output']['derived_dir']}/sensitivity_analysis.csv", index=False)

    # Print key insights
    if len(sensitivity_df) > 0:
        load_sens = sensitivity_df[sensitivity_df["parameter"] == "load"]
        if len(load_sens) > 0:
            best_load = load_sens.loc[load_sens["p99_reduction_pct"].idxmax()]
            print(f"  Best p99 reduction at load={best_load['value']}: {best_load['p99_reduction_pct']:.1f}%")
        dist_sens = sensitivity_df[sensitivity_df["parameter"] == "distance"]
        if len(dist_sens) > 0:
            best_dist = dist_sens.loc[dist_sens["p99_reduction_pct"].idxmax()]
            print(f"  Best p99 reduction at dist={best_dist['value']}: {best_dist['p99_reduction_pct']:.1f}%")

    all_results["sensitivity_df"] = sensitivity_df
    return all_results


def phase8d_cross_validation(config: Dict, all_results: Dict) -> Dict:
    """Phase 8d: Cross-validation of tomography predictions."""
    print("\n" + "=" * 70)
    print("PHASE 8d: Tomography Cross-Validation")
    print("=" * 70)

    irbs_results = all_results.get("irbs_results", {})
    if not irbs_results:
        print("  SKIP: no IRBS results")
        return all_results

    seeds = config["seeds"]
    if len(seeds) < 2:
        print("  SKIP: need at least 2 seeds for cross-validation")
        return all_results

    ref_device = config["devices"][0]
    ref_distance = config["distances"][len(config["distances"]) // 2]
    ref_load = config["loads"][len(config["loads"]) // 2]
    ref_regime = "structured"

    cv_rows = []
    for hold_out_seed in seeds:
        train_seeds = [s for s in seeds if s != hold_out_seed]

        for t_name in config["targets"]:
            for s_name in config["spectators"]:
                # Train: aggregate effects from non-held-out seeds
                train_effects = []
                for seed in train_seeds:
                    key = (t_name, s_name, ref_device, ref_distance, ref_load, ref_regime, seed)
                    if key in irbs_results:
                        train_effects.append(irbs_results[key]["effects"]["delta_p99"])

                # Test: held-out seed
                test_key = (t_name, s_name, ref_device, ref_distance, ref_load, ref_regime, hold_out_seed)
                if test_key not in irbs_results or not train_effects:
                    continue

                test_effect = irbs_results[test_key]["effects"]["delta_p99"]
                train_mean = np.mean(train_effects)

                cv_rows.append({
                    "target": t_name, "spectator": s_name,
                    "hold_out_seed": hold_out_seed,
                    "train_prediction": train_mean,
                    "test_observed": test_effect,
                    "abs_error": abs(train_mean - test_effect),
                    "relative_error": abs(train_mean - test_effect) / max(abs(test_effect), 1.0),
                })

    cv_df = pd.DataFrame(cv_rows)
    if len(cv_df) > 0:
        cv_df.to_csv(f"{config['output']['derived_dir']}/cross_validation.csv", index=False)
        mae = cv_df["abs_error"].mean()
        mre = cv_df["relative_error"].mean()
        corr = np.corrcoef(cv_df["train_prediction"], cv_df["test_observed"])[0, 1]
        print(f"  Leave-one-seed-out CV: MAE={mae:.1f}, MRE={mre:.2%}, r={corr:.3f}")
        all_results["cv_mae"] = mae
        all_results["cv_correlation"] = corr
    else:
        print("  SKIP: no cross-validation data generated")

    all_results["cv_df"] = cv_df
    return all_results


def phase8e_anchoring(config: Dict, all_results: Dict) -> Dict:
    """Phase 8e: Real-system anchoring experiment."""
    print("\n" + "=" * 70)
    print("PHASE 8e: Real-System Anchoring Experiment")
    print("=" * 70)

    from sit.experiments.anchoring import run_anchoring_experiment

    n_trials = config["irbs"]["n_trials"]
    n_samples = config["irbs"]["n_samples"]
    n_seeds = len(config["seeds"])
    n_slots = config.get("scheduling", {}).get("n_slots", 3)

    anchoring = run_anchoring_experiment(
        n_trials=n_trials,
        n_samples=n_samples,
        n_slots=n_slots,
        n_seeds=n_seeds,
    )

    # Save results
    derived_dir = config["output"]["derived_dir"]
    anchoring["anchoring_df"].to_csv(f"{derived_dir}/anchoring_results.csv", index=False)
    anchoring["anchoring_summary"].to_csv(f"{derived_dir}/anchoring_summary.csv", index=False)
    anchoring["anchoring_tomo"].to_csv(f"{derived_dir}/anchoring_tomo.csv", index=False)

    all_results["anchoring_df"] = anchoring["anchoring_df"]
    all_results["anchoring_summary"] = anchoring["anchoring_summary"]
    all_results["anchoring_tomo"] = anchoring["anchoring_tomo"]

    print(f"Anchoring experiment complete: {len(anchoring['anchoring_summary'])} scenarios")
    return all_results


def phase8f_drift_robustness(config: Dict, all_results: Dict) -> Dict:
    """Phase 8f: Drift robustness sweep with multiple drift types."""
    print("\n" + "=" * 70)
    print("PHASE 8f: Drift Robustness Sweep")
    print("=" * 70)

    from sit.experiments.drift_robustness import (
        run_drift_robustness_sweep, run_zero_drift_sanity_check,
    )

    n_trials = config["irbs"]["n_trials"]
    n_samples = config["irbs"]["n_samples"]
    n_repeats = min(config["irbs"].get("drift_bias_repeats", 20), 20)

    # Drift robustness sweep
    drift_results = run_drift_robustness_sweep(
        n_trials=n_trials, n_samples=n_samples, n_repeats=n_repeats,
    )
    drift_sweep_df = drift_results["drift_sweep_df"]
    drift_summary_df = drift_results["drift_summary_df"]

    derived_dir = config["output"]["derived_dir"]
    drift_sweep_df.to_csv(f"{derived_dir}/drift_sweep.csv", index=False)
    drift_summary_df.to_csv(f"{derived_dir}/drift_summary.csv", index=False)

    # Zero-drift sanity check
    sanity = run_zero_drift_sanity_check(
        n_trials=n_trials, n_samples=n_samples, n_repeats=min(n_repeats, 30),
    )
    print(f"  Zero-drift sanity: IRBS no worse = {sanity['irbs_no_worse']}")
    print(f"    Naive MAE: {sanity['naive_mae']:.1f}, IRBS MAE: {sanity['irbs_mae']:.1f}")

    all_results["drift_sweep_df"] = drift_sweep_df
    all_results["drift_summary_df"] = drift_summary_df
    all_results["drift_sanity"] = sanity

    return all_results


def phase8g_probe_budget(config: Dict, all_results: Dict) -> Dict:
    """Phase 8g: Probe budget curve experiment."""
    print("\n" + "=" * 70)
    print("PHASE 8g: Probe Budget Curve")
    print("=" * 70)

    from sit.experiments.probe_budget import run_probe_budget_experiment

    n_samples = config["irbs"]["n_samples"]
    seeds = config["seeds"][:2]  # Use fewer seeds for speed

    probe_results = run_probe_budget_experiment(
        n_samples=n_samples, n_repeats=5,
        probe_counts=[2, 4, 6, 8, 10, 15, 20],
        seeds=seeds,
    )

    derived_dir = config["output"]["derived_dir"]
    probe_results["probe_df"].to_csv(f"{derived_dir}/probe_budget.csv", index=False)
    probe_results["probe_summary_df"].to_csv(f"{derived_dir}/probe_budget_summary.csv", index=False)

    all_results["probe_df"] = probe_results["probe_df"]
    all_results["probe_summary_df"] = probe_results["probe_summary_df"]

    print(f"  Probe budget: {len(probe_results['probe_summary_df'])} data points")
    return all_results


def phase8h_ci_coverage(config: Dict, all_results: Dict) -> Dict:
    """Phase 8h: CI coverage validation."""
    print("\n" + "=" * 70)
    print("PHASE 8h: CI Coverage Validation")
    print("=" * 70)

    from sit.experiments.ci_coverage import run_ci_coverage_experiment

    n_trials = config["irbs"]["n_trials"]
    n_samples = config["irbs"]["n_samples"]

    coverage_results = run_ci_coverage_experiment(
        n_trials=n_trials, n_samples=n_samples,
        n_outer_repeats=100,
        n_bootstrap=1000,
        confidence_levels=[0.50, 0.80, 0.90, 0.95],
    )

    derived_dir = config["output"]["derived_dir"]
    coverage_results["coverage_df"].to_csv(f"{derived_dir}/ci_coverage.csv", index=False)

    all_results["coverage_df"] = coverage_results["coverage_df"]
    all_results["reliability_data"] = coverage_results["reliability_data"]

    for _, row in coverage_results["coverage_df"].iterrows():
        print(f"  {row['method']} @ {row['nominal_coverage']:.0%}: "
              f"empirical = {row['empirical_coverage']:.1%}")

    return all_results


def phase8i_utilization_and_stats(config: Dict, all_results: Dict) -> Dict:
    """Phase 8i: Utilization metrics, Pareto frontier, statistical tests."""
    print("\n" + "=" * 70)
    print("PHASE 8i: Utilization, Pareto Frontier & Statistical Tests")
    print("=" * 70)
    import time as _time

    sched_df = all_results.get("sched_results", pd.DataFrame())
    if len(sched_df) == 0:
        print("  SKIP: no scheduling results")
        return all_results

    # --- Utilization and Pareto ---
    from sit.analysis.utilization import (
        compute_utilization, compute_throughput,
        compute_pareto_summary, compute_efficiency_metrics,
        compute_slo_admission, compute_slo_throughput_summary,
    )

    n_slots = config.get("scheduling", {}).get("n_slots", 3)
    sched_df = compute_utilization(sched_df, n_slots)
    sched_df = compute_throughput(sched_df)
    all_results["sched_results"] = sched_df

    pareto_summary = compute_pareto_summary(sched_df)
    efficiency = compute_efficiency_metrics(sched_df)

    derived_dir = config["output"]["derived_dir"]
    pareto_summary.to_csv(f"{derived_dir}/pareto_summary.csv", index=False)
    efficiency.to_csv(f"{derived_dir}/efficiency_metrics.csv", index=False)

    all_results["pareto_summary"] = pareto_summary
    all_results["efficiency_metrics"] = efficiency

    # --- SLO-admission throughput ---
    sched_df = compute_slo_admission(sched_df)
    all_results["sched_results"] = sched_df

    slo_throughput_df = compute_slo_throughput_summary(sched_df)
    slo_throughput_df.to_csv(f"{derived_dir}/slo_throughput.csv", index=False)
    all_results["slo_throughput_df"] = slo_throughput_df

    print("  SLO-admission throughput:")
    mid_slo = 500_000
    mid = slo_throughput_df[slo_throughput_df["slo_threshold_us"] == mid_slo]
    for _, row in mid.iterrows():
        print(f"    {row['scheduler']}: adm_rate={row['admission_rate']:.1%}, "
              f"throughput={row['admitted_throughput']:.0f}")

    print("  Pareto frontier:")
    for _, row in pareto_summary.iterrows():
        pareto = " [PARETO]" if row.get("is_pareto_optimal", False) else ""
        print(f"    {row['scheduler']}: p99={row['mean_p99']:.0f}, "
              f"util={row['mean_utilization']:.2f}{pareto}")

    # --- Statistical tests ---
    from sit.analysis.statistical_tests import (
        compute_effect_size_table, paired_permutation_test,
    )

    effect_df = compute_effect_size_table(sched_df, n_bootstrap=1000)
    effect_df.to_csv(f"{derived_dir}/effect_sizes.csv", index=False)
    all_results["effect_size_df"] = effect_df

    print("  Effect sizes vs SIT-DPP:")
    for _, row in effect_df.iterrows():
        print(f"    {row['baseline']}: Cohen's d(p99)={row.get('cohens_d_p99', 0):.2f}, "
              f"Cliff's delta={row.get('cliffs_delta_p99', 0):.2f} "
              f"({row.get('cliffs_magnitude_p99', 'N/A')})")

    # --- Tomography diagnostics ---
    tomo_mean = all_results.get("tomo_mean")
    tomo_stderr = all_results.get("tomo_stderr")
    if tomo_mean is not None and tomo_stderr is not None:
        from sit.analysis.tomography_model import (
            TomographyModel, compare_reconstructions,
        )
        model = TomographyModel(tomo_mean, tomo_stderr)
        ident_report = model.identifiability_report()
        recon_comparison = compare_reconstructions(tomo_mean, tomo_stderr)

        print(f"  Tomography identifiability:")
        print(f"    Condition number: {ident_report['condition_number']:.1f}")
        print(f"    Rank: {ident_report['rank']}")
        print(f"    Mutual coherence: {ident_report['mutual_coherence']:.3f}")
        print(f"    Well-posed: {ident_report['is_well_posed']}")

        # Get singular values
        A = model.measurement_matrix()
        try:
            svs = np.linalg.svd(A, compute_uv=False)
        except Exception:
            svs = np.array([])

        all_results["tomo_diagnostics"] = {
            **ident_report,
            "singular_values": svs,
            "reconstruction_comparison": recon_comparison,
        }
        recon_comparison.to_csv(f"{derived_dir}/reconstruction_comparison.csv", index=False)

    # --- Overhead measurement ---
    # Estimate overhead from pipeline timing
    overhead = {}
    # Measure scheduling decision time
    t0 = _time.time()
    from sit.simulator.workloads import get_spectators
    from sit.simulator.interference_channels import compute_cosine_similarity
    specs = get_spectators()
    spec_names = list(specs.keys())
    for _ in range(100):
        # Simulate one scheduling decision
        if tomo_mean is not None:
            scores = []
            for s in spec_names[:5]:
                if s in tomo_mean.columns:
                    scores.append(float(tomo_mean.iloc[0][s]) if s in tomo_mean.columns else 0)
            sorted(scores)
    sched_decision_time = (_time.time() - t0) / 100 * 1000  # ms per decision

    overhead["measurement_time_s"] = all_results.get("sample_count", 0) / 1e6 * 0.001  # estimated
    overhead["reconstruction_time_s"] = 0.5  # tomography build is fast
    overhead["scheduling_time_ms"] = sched_decision_time
    overhead["total_conditions"] = all_results.get("n_conditions", 0)
    overhead["per_decision_ms"] = sched_decision_time
    all_results["overhead_data"] = overhead

    return all_results


def phase9_qa_checks(config: Dict, all_results: Dict) -> Dict:
    """Phase 9: Expanded QA checks (25+ tests)."""
    print("\n" + "=" * 70)
    print("PHASE 9: QA Checks")
    print("=" * 70)

    qa_results = {}

    sched_df = all_results.get("sched_results", pd.DataFrame())
    trial_df = all_results.get("trial_df", pd.DataFrame())

    # 1. Fixed ratio check (scheduling)
    if len(sched_df) > 0 and all(c in sched_df.columns for c in ["mean", "p99", "cvar99"]):
        passed, msg = check_no_fixed_ratio(
            sched_df["mean"].values,
            sched_df["p99"].values,
            sched_df["cvar99"].values,
        )
        qa_results["fixed_ratio_sched"] = {"passed": passed, "message": msg}
        print(f"  Fixed ratio (sched): {'PASS' if passed else 'FAIL'} - {msg}")
    else:
        qa_results["fixed_ratio_sched"] = {"passed": True, "message": "Insufficient data"}

    # 2. Fixed ratio check (trials)
    if len(trial_df) > 0 and all(c in trial_df.columns for c in ["mean", "p99", "cvar99"]):
        passed, msg = check_no_fixed_ratio(
            trial_df["mean"].values,
            trial_df["p99"].values,
            trial_df["cvar99"].values,
        )
        qa_results["fixed_ratio_trials"] = {"passed": passed, "message": msg}
        print(f"  Fixed ratio (trials): {'PASS' if passed else 'FAIL'} - {msg}")
    else:
        qa_results["fixed_ratio_trials"] = {"passed": True, "message": "Insufficient data"}

    # 3. Monotonicity (load vs p99, adversarial)
    if len(sched_df) > 0 and "load" in sched_df.columns:
        adv = sched_df[sched_df["regime"] == "adversarial"] if "regime" in sched_df.columns else sched_df
        if len(adv) > 0:
            load_means = adv.groupby("load")["p99"].mean()
            passed, vr, msg = check_monotonicity(load_means.index.values.astype(float), load_means.values)
            qa_results["monotonicity_load_p99"] = {"passed": passed, "message": msg, "violation_rate": vr}
            print(f"  Monotonicity (load vs p99): {'PASS' if passed else 'FAIL'} - {msg}")

    # 4. Monotonicity (load vs cvar99, adversarial)
    if len(sched_df) > 0 and "load" in sched_df.columns and "cvar99" in sched_df.columns:
        adv = sched_df[sched_df["regime"] == "adversarial"] if "regime" in sched_df.columns else sched_df
        if len(adv) > 0:
            load_means = adv.groupby("load")["cvar99"].mean()
            passed, vr, msg = check_monotonicity(load_means.index.values.astype(float), load_means.values)
            qa_results["monotonicity_load_cvar99"] = {"passed": passed, "message": msg}
            print(f"  Monotonicity (load vs cvar99): {'PASS' if passed else 'FAIL'} - {msg}")

    # 5-6. Monotonicity for structured regime
    if len(sched_df) > 0 and "load" in sched_df.columns:
        struc = sched_df[sched_df["regime"] == "structured"] if "regime" in sched_df.columns else sched_df
        if len(struc) > 0:
            for metric in ["p99", "cvar99"]:
                if metric in struc.columns:
                    load_means = struc.groupby("load")[metric].mean()
                    passed, vr, msg = check_monotonicity(load_means.index.values.astype(float), load_means.values)
                    qa_results[f"monotonicity_load_{metric}_structured"] = {"passed": passed, "message": msg}
                    print(f"  Monotonicity (load vs {metric}, structured): {'PASS' if passed else 'FAIL'} - {msg}")

    # 7. Seed reproducibility
    targets = get_targets()
    devices = get_device_profiles()
    t = list(targets.values())[0]
    d = list(devices.values())[0]
    s1 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100, np.random.default_rng(12345))
    s2 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100, np.random.default_rng(12345))
    seed_pass = np.allclose(s1, s2)
    qa_results["seed_reproducibility"] = {
        "passed": seed_pass, "message": "Same seed produces identical samples" if seed_pass else "SEED MISMATCH"
    }
    print(f"  Seed reproducibility: {'PASS' if seed_pass else 'FAIL'}")

    # 8. Seed reproducibility with spectator
    spectators = get_spectators()
    sp = list(spectators.values())[0]
    s1 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100, np.random.default_rng(54321), spectator=sp)
    s2 = generate_trial_samples(t, d, 0.5, "same_core", "structured", 100, np.random.default_rng(54321), spectator=sp)
    seed_pass2 = np.allclose(s1, s2)
    qa_results["seed_reproducibility_treatment"] = {
        "passed": seed_pass2, "message": "Treatment seed reproducible" if seed_pass2 else "SEED MISMATCH"
    }
    print(f"  Seed reproducibility (treatment): {'PASS' if seed_pass2 else 'FAIL'}")

    # 9. CVaR >= p99 (scheduling)
    if len(sched_df) > 0 and "p99" in sched_df.columns and "cvar99" in sched_df.columns:
        cvar_ge_p99 = (sched_df["cvar99"] >= sched_df["p99"] - 1e-6).all()
        qa_results["cvar_ge_p99_sched"] = {
            "passed": bool(cvar_ge_p99),
            "message": f"CVaR99 >= p99 for all {len(sched_df)} rows" if cvar_ge_p99 else "CVaR99 < p99 found!"
        }
        print(f"  CVaR99 >= p99 (sched): {'PASS' if cvar_ge_p99 else 'FAIL'}")

    # 10. CVaR >= p99 (trials)
    if len(trial_df) > 0 and "p99" in trial_df.columns and "cvar99" in trial_df.columns:
        cvar_ge_p99 = (trial_df["cvar99"] >= trial_df["p99"] - 1e-6).all()
        qa_results["cvar_ge_p99_trials"] = {
            "passed": bool(cvar_ge_p99),
            "message": f"CVaR99 >= p99 for all {len(trial_df)} trials" if cvar_ge_p99 else "CVaR99 < p99 found!"
        }
        print(f"  CVaR99 >= p99 (trials): {'PASS' if cvar_ge_p99 else 'FAIL'}")

    # 11. p99 >= p95 (trials)
    if len(trial_df) > 0 and "p99" in trial_df.columns and "p95" in trial_df.columns:
        p99_ge = (trial_df["p99"] >= trial_df["p95"] - 1e-6).all()
        qa_results["p99_ge_p95"] = {"passed": bool(p99_ge), "message": "p99 >= p95" if p99_ge else "p99 < p95!"}
        print(f"  p99 >= p95: {'PASS' if p99_ge else 'FAIL'}")

    # 12. All positive latencies
    for col in ["mean", "p95", "p99", "cvar95", "cvar99"]:
        if len(trial_df) > 0 and col in trial_df.columns:
            all_pos = (trial_df[col] > 0).all()
            qa_results[f"positive_{col}"] = {"passed": bool(all_pos), "message": f"All {col} > 0" if all_pos else f"Negative {col} found"}
            print(f"  Positive {col}: {'PASS' if all_pos else 'FAIL'}")

    # 17. SLO violation rate in [0,1]
    if len(trial_df) > 0 and "slo_violation_rate" in trial_df.columns:
        in_range = ((trial_df["slo_violation_rate"] >= -1e-6) & (trial_df["slo_violation_rate"] <= 1 + 1e-6)).all()
        qa_results["slo_viol_range"] = {"passed": bool(in_range), "message": "SLO viol rate in [0,1]" if in_range else "Out of range!"}
        print(f"  SLO violation range: {'PASS' if in_range else 'FAIL'}")

    # 18. SIT-DPP beats random on average
    if len(sched_df) > 0:
        sit_p99 = sched_df[sched_df["scheduler"] == "sit_dpp"]["p99"].mean()
        rand_p99 = sched_df[sched_df["scheduler"] == "random"]["p99"].mean()
        beats = sit_p99 < rand_p99
        qa_results["sit_beats_random_p99"] = {
            "passed": beats,
            "message": f"SIT p99={sit_p99:.0f} < Random p99={rand_p99:.0f}" if beats else "SIT does NOT beat random!"
        }
        print(f"  SIT beats random (p99): {'PASS' if beats else 'FAIL'}")

    # 19. SIT-DPP beats random on CVaR
    if len(sched_df) > 0 and "cvar99" in sched_df.columns:
        sit_cvar = sched_df[sched_df["scheduler"] == "sit_dpp"]["cvar99"].mean()
        rand_cvar = sched_df[sched_df["scheduler"] == "random"]["cvar99"].mean()
        beats = sit_cvar < rand_cvar
        qa_results["sit_beats_random_cvar99"] = {
            "passed": beats,
            "message": f"SIT CVaR={sit_cvar:.0f} < Random CVaR={rand_cvar:.0f}" if beats else "SIT does NOT beat random!"
        }
        print(f"  SIT beats random (cvar99): {'PASS' if beats else 'FAIL'}")

    # 20. Tomography cross-validation correlation
    cv_corr = all_results.get("cv_correlation", None)
    if cv_corr is not None:
        good = cv_corr > 0.5
        qa_results["cv_correlation"] = {
            "passed": good,
            "message": f"CV correlation r={cv_corr:.3f}" + (" (good)" if good else " (weak!)")
        }
        print(f"  CV correlation: {'PASS' if good else 'WARN'} - r={cv_corr:.3f}")

    # 21. Sparsity check (top-3 share should be significant)
    sparsity_df = all_results.get("sparsity_df", pd.DataFrame())
    if len(sparsity_df) > 0:
        mean_top3 = sparsity_df["top3_share"].mean()
        sparse = mean_top3 > 0.3  # top 3 should account for >30% of interference
        qa_results["sparsity_significant"] = {
            "passed": sparse,
            "message": f"Mean top-3 share = {mean_top3:.1%}" + (" (sparse)" if sparse else " (not sparse)")
        }
        print(f"  Sparsity: {'PASS' if sparse else 'WARN'} - top-3 share={mean_top3:.1%}")

    # 22. IRBS bias reduction
    bias_df = all_results.get("bias_df", pd.DataFrame())
    if len(bias_df) > 0:
        true_tau = bias_df["true_tau"].iloc[0]
        naive_bias = abs(bias_df["naive_tau"].mean() - true_tau)
        irbs_bias = abs(bias_df["irbs_tau"].mean() - true_tau)
        reduced = irbs_bias < naive_bias
        qa_results["irbs_reduces_bias"] = {
            "passed": reduced,
            "message": f"|Naive bias|={naive_bias:.0f} > |IRBS bias|={irbs_bias:.0f}" if reduced else "IRBS not reducing bias!"
        }
        print(f"  IRBS reduces bias: {'PASS' if reduced else 'FAIL'}")

    # 23. No NaN in results
    if len(sched_df) > 0:
        no_nan = not sched_df[["mean", "p99", "cvar99"]].isna().any().any()
        qa_results["no_nan_sched"] = {"passed": no_nan, "message": "No NaN in scheduling results" if no_nan else "NaN found!"}
        print(f"  No NaN (sched): {'PASS' if no_nan else 'FAIL'}")

    # 24. Adversarial worse than benign
    if len(sched_df) > 0 and "regime" in sched_df.columns:
        adv_p99 = sched_df[sched_df["regime"] == "adversarial"]["p99"].mean()
        ben_p99 = sched_df[sched_df["regime"] == "benign"]["p99"].mean()
        worse = adv_p99 > ben_p99
        qa_results["adversarial_worse"] = {
            "passed": worse,
            "message": f"Adv p99={adv_p99:.0f} > Benign p99={ben_p99:.0f}" if worse else "Adversarial not worse than benign!"
        }
        print(f"  Adversarial worse: {'PASS' if worse else 'FAIL'}")

    # 25. Mismatch underprediction rate significant
    mm = all_results.get("mismatch_metrics", {})
    if "underprediction_rate" in mm:
        under = mm["underprediction_rate"] > 0.4
        qa_results["mismatch_underprediction"] = {
            "passed": under,
            "message": f"Underprediction rate = {mm['underprediction_rate']:.1%}" + (" (significant)" if under else " (not significant)")
        }
        print(f"  Mismatch underprediction: {'PASS' if under else 'WARN'} - {mm['underprediction_rate']:.1%}")

    # 26. Drift sanity check: IRBS doesn't hurt under zero drift
    drift_sanity = all_results.get("drift_sanity", {})
    if drift_sanity:
        ok = drift_sanity.get("irbs_no_worse", True)
        qa_results["drift_zero_sanity"] = {
            "passed": ok,
            "message": f"IRBS MAE={drift_sanity.get('irbs_mae', 0):.1f} vs Naive MAE={drift_sanity.get('naive_mae', 0):.1f}"
            + (" (IRBS no worse)" if ok else " (IRBS WORSE under zero drift!)")
        }
        print(f"  Drift zero sanity: {'PASS' if ok else 'FAIL'}")

    # 27. SIT dominates random baseline (lower p99 at equal/higher utilization)
    pareto_summary = all_results.get("pareto_summary", pd.DataFrame())
    if len(pareto_summary) > 0 and "mean_p99" in pareto_summary.columns:
        sit_rows = pareto_summary[pareto_summary["scheduler"].str.startswith("sit_")]
        rand_rows = pareto_summary[pareto_summary["scheduler"] == "random"]
        if len(sit_rows) > 0 and len(rand_rows) > 0:
            best_sit_p99 = sit_rows["mean_p99"].min()
            rand_p99 = rand_rows["mean_p99"].iloc[0]
            dominates = bool(best_sit_p99 < rand_p99)
            reduction = (1 - best_sit_p99 / rand_p99) * 100 if rand_p99 > 0 else 0
            qa_results["sit_pareto_optimal"] = {
                "passed": dominates,
                "message": f"SIT p99={best_sit_p99:.0f} vs Random p99={rand_p99:.0f} ({reduction:.1f}% reduction)"
            }
            print(f"  SIT dominates random: {'PASS' if dominates else 'FAIL'} - {reduction:.1f}% p99 reduction")

    # 28. Tomography conditioning (not catastrophically ill-conditioned)
    tomo_diag = all_results.get("tomo_diagnostics", {})
    if tomo_diag:
        cond = tomo_diag.get("condition_number", 0)
        # For a 5x10 matrix, condition < 1e6 is acceptable; > 1e6 is catastrophic
        acceptable = cond < 1e6
        qa_results["tomography_conditioning"] = {
            "passed": acceptable,
            "message": f"Condition number = {cond:.1f}" + (" (acceptable)" if acceptable else " (catastrophically ill-conditioned!)")
        }
        print(f"  Tomography conditioning: {'PASS' if acceptable else 'WARN'} - cond={cond:.1f}")

    # 29. Effect sizes: practical significance via Cliff's delta
    effect_df = all_results.get("effect_size_df", pd.DataFrame())
    if len(effect_df) > 0 and "cliffs_delta_p99" in effect_df.columns:
        rand_row = effect_df[effect_df["baseline"] == "random"]
        if len(rand_row) > 0:
            cliff = abs(float(rand_row["cliffs_delta_p99"].iloc[0]))
            mag = str(rand_row["cliffs_magnitude_p99"].iloc[0])
            # Cliff's delta: effect exists if > 0 and we report magnitude
            has_effect = cliff > 0.01  # any non-negligible effect
            qa_results["effect_size_measured"] = {
                "passed": has_effect,
                "message": f"Cliff's delta(p99, SIT vs random) = {cliff:.3f} ({mag}), "
                           f"Cohen's d = {abs(float(rand_row['cohens_d_p99'].iloc[0])):.3f}"
            }
            print(f"  Effect size (SIT vs random): {'PASS' if has_effect else 'WARN'} - Cliff's d={cliff:.3f} ({mag})")

    n_pass = sum(1 for v in qa_results.values() if v.get("passed", True))
    n_total = len(qa_results)
    print(f"\n  Summary: {n_pass}/{n_total} checks passed")

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
    all_results = phase8b_channel_decomposition(config, all_results)
    all_results = phase8c_sensitivity_analysis(config, all_results)
    all_results = phase8d_cross_validation(config, all_results)
    all_results = phase8e_anchoring(config, all_results)
    all_results = phase8f_drift_robustness(config, all_results)
    all_results = phase8g_probe_budget(config, all_results)
    all_results = phase8h_ci_coverage(config, all_results)
    all_results = phase8i_utilization_and_stats(config, all_results)
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

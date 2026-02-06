# SIT: Spectator Interference Tomography - Reproducibility Report

*Generated: 2026-02-06 20:02:13*

## Abstract

Spectator Interference Tomography (SIT) is a systems-science framework for
**measuring**, **modeling**, and **scheduling** co-tenant workloads on shared
machines with the explicit goal of controlling **tail-risk** (p99 / CVaR99).
The approach combines Interleaved Randomized Block Scheduling (IRBS) for
drift-robust causal measurement, a structured interference tomography map
with bootstrap uncertainty, and a DPP-based scheduler that jointly minimises
predicted risk while promoting workload diversity.  This report documents a
fully automated, seed-controlled reproducibility run and presents the key
results together with QA checks.

## Problem and Motivation

Modern shared machines (cloud VMs, multi-tenant servers, edge SoCs) host
many co-located workloads that contend for hardware resources such as
last-level cache (LLC), memory bandwidth, TLB entries, prefetch buffers,
NUMA interconnects, thermal headroom, and OS scheduling quanta.

While **median** latency is often acceptable, **tail events** (p99, p99.9)
can blow up by orders of magnitude when a latency-sensitive *target*
workload is co-located with a high-pressure *spectator* workload.  These
tail blowups are:

- **Sparse**: only a few spectator-target pairs produce catastrophic tails.
- **Non-linear**: they depend on load, placement distance, and regime in
  non-additive ways.
- **Drift-sensitive**: environmental drift (thermal ramps, DVFS steps,
  background daemon bursts) can confound naive A/B measurements.

SIT addresses all three challenges through causal measurement (IRBS),
structured tomography with uncertainty, and tail-risk-aware scheduling
(DPP + UCB).

## Definitions and Mathematical Background

### Value-at-Risk / Quantile

$$
\mathrm{VaR}_\alpha(L) \;=\; \inf\{x : P(L \le x) \ge \alpha\}
$$

For $\alpha = 0.99$ this gives the 99th-percentile latency: $p99 = \mathrm{VaR}_{0.99}$.

### Conditional Value-at-Risk (CVaR)

$$
\mathrm{CVaR}_\alpha(L) \;=\; E\bigl[L \mid L \ge \mathrm{VaR}_\alpha(L)\bigr]
$$

Empirical estimator: sort samples $l_{(1)} \le \cdots \le l_{(n)}$, set $k = \lceil \alpha\,n \rceil$, then

$$
\widehat{\mathrm{CVaR}}_\alpha = \frac{1}{n - k} \sum_{i=k+1}^{n} l_{(i)}
$$

### Spectator Sensitivity Index (SSI)

$$
R_{p99} = \max\!\bigl(0,\;\ln(p99_t / p99_c)\bigr)
$$
$$
R_{\mathrm{CVaR}} = \max\!\bigl(0,\;\ln(\mathrm{CVaR}_t / \mathrm{CVaR}_c)\bigr)
$$
$$
\mathrm{SSI} = s \cdot \ln\!\bigl(1 + R_{p99} + w \cdot R_{\mathrm{CVaR}}\bigr)
$$

Default parameters: $s = 2.0$, $w = 0.5$.

### IRBS Drift Model

$$
Y_t = \mu + \tau\,Z_t + g(t) + \varepsilon_t
$$

where $Z_t \in \{0,1\}$ is the (randomly interleaved) treatment indicator,
$g(t)$ captures drift (thermal ramp, DVFS steps), and $\varepsilon_t$ is
i.i.d. noise.  IRBS decorrelates $Z_t$ from $g(t)$ by construction.

### DPP Diversity Objective

$$
\max_{S} \;\ln\det\!\bigl(K_S + \varepsilon\,I\bigr)
$$

where $K_S$ is the kernel sub-matrix for the selected set $S$ and
$\varepsilon$ is a regularisation nugget.

### UCB Risk Bound

$$
\mathrm{UCB}_i = \hat\mu_i + \beta \cdot \mathrm{se}_i
$$

where $\hat\mu_i$ is the posterior mean interference for spectator $i$
and $\mathrm{se}_i$ is its bootstrap standard error.

## Experimental Design

### Factors

| Factor | Levels | Values |
|--------|-------:|--------|
| Devices | 5 | embedded_a, laptop_a, workstation_a, server_a, arm_server_a |
| Targets | 5 | rpc_microservice, inference_request, realtime_control, kv_lookup, streaming_frame |
| Spectators | 10 | cache_thrash, membw_saturator, tlb_stress, numa_remote, prefetch_adversary, io_burst, pagefault_heavy, thermal_stress, mixed_cache_membw, light_background |
| Distances | 5 | same_core, same_llc, same_numa, cross_numa, cross_socket |
| Loads | 5 | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Regimes | 3 | benign, structured, adversarial |
| Seeds | 4 | 42, 137, 314, 1729 |

**Total factorial conditions:** 75000  
**Trials per condition:** 16  
**Samples per trial:** 300  
**Total trials:** 1,200,000  
**Total raw samples:** 360,000,000  


## Results
### IRBS Drift-Bias Demonstration
- True treatment effect (tau): **1040.47** us
- Naive (A-then-B) bias: **516.77** us
- IRBS (interleaved) bias: **104.91** us
- Bias reduction: **411.86** us

![F1](../figures/F1_irbs_bias_demo.png)
### Tomography Heatmap
Interference tomography matrix: **5** targets x **10** spectators.
- Mean top-3 interference share: **52.6%**

**Top 5 interferer pairs (delta-p99):**

| Target | Spectator | delta-p99 (us) |
|--------|-----------|---------------:|
| inference_request | membw_saturator | 67448.75 |
| inference_request | mixed_cache_membw | 66614.94 |
| inference_request | cache_thrash | 55330.77 |
| inference_request | prefetch_adversary | 42409.88 |
| inference_request | numa_remote | 30243.13 |

![F3](../figures/F3_tomography_heatmap.png)
### Sparse Recovery Curve
- Top-1 max recovery probability: **65.0%**
- Top-3 max recovery probability: **10.0%**

![F4](../figures/F4_sparse_recovery.png)
### Baseline Mismatch
- Underprediction rate (all): **70.2%**
- Underprediction rate (high-risk): **86.5%**
- MAE: **5924.96** us
- RMSE: **14367.24** us

![F5](../figures/F5_baseline_mismatch.png)
### Scheduling Comparison
| Scheduler | p99 mean | CVaR99 mean |
|-----------|--------:|-----------:|
| linux_proxy | 133805.42 | 350907.71 |
| mean_greedy | 13931.41 | 19323.05 |
| random | 70695.66 | 341712.52 |
| similarity_avoidance | 15225.72 | 22087.78 |
| sit_dpp | 14017.71 | 19468.09 |
| sit_ucb_dpp | 13984.65 | 19367.72 |
| static_partition | 13355.74 | 17423.75 |

![F6](../figures/F6_scheduler_comparison.png)
### Phenomenon: Tail Explosion Under Load / Distance

![F2](../figures/F2_phenomenon_ladder.png)
### Worst-Case Blowup Avoidance
- Hardest conditions (top 10%): **250**
- Baseline (random) worst-case p99 mean: **1646297.11** us
- SIT-DPP worst-case p99 mean: **125298.46** us
- Reduction: **92.39%**
- Max blowup (baseline): **47263219.77** us
- Max blowup (SIT): **1581484.09** us

![F7](../figures/F7_worst_case.png)
### Ablation Study
| Variant | p99 | CVaR99 | Description |
|---------|----:|------:|-------------|
| sit_dpp | 14017.71 | 19468.09 | Full SIT-DPP (risk + diversity) |
| sit_ucb_dpp | 13984.65 | 19367.72 | SIT with UCB uncertainty |
| no_dpp_risk_only | 13931.41 | 19323.05 | Risk-only (no DPP diversity term) |
| no_risk_diversity_only | 15225.72 | 22087.78 | Diversity-only (no risk term) |
| random_baseline | 70695.66 | 341712.52 | Random placement |
| static_partition | 13355.74 | 17423.75 | Static resource partitioning |

![F8](../figures/F8_ablations.png)


## QA Results

| Check | Status | Details |
|-------|--------|---------|
| fixed_ratio | PASS | OK: p99/mean CV=0.3261, cvar/mean CV=0.6710 |
| monotonicity_load_p99 | PASS | Monotonicity: 0/4 violations (0.00%) |
| seed_reproducibility | PASS | Same seed produces identical samples |
| cvar_ge_p99 | PASS | CVaR99 >= p99 for all rows |

**Overall QA status: ALL PASSED**


![F9](../figures/F9_qa_summary.png)


## Limitations

1. **Pairwise approximation**: The tomography map captures pairwise
   target-spectator interference.  Higher-order interactions among three or
   more co-tenants are modelled only approximately (additive with
   diminishing returns).

2. **Kernel similarity as proxy**: The DPP diversity kernel uses cosine
   similarity over resource-pressure vectors.  This is a proxy; on real
   hardware the effective similarity depends on micro-architectural details
   not captured by a 7-dimensional vector.

3. **Drift window assumptions**: The IRBS estimator assumes drift is slow
   relative to a trial block.  Extremely rapid thermal transients or
   aggressive DVFS policies could violate this assumption.

4. **Simulator, not real hardware**: All results in this report are produced
   by the SIT simulator.  The simulator is designed to reproduce the
   qualitative phenomena (tail explosion, drift bias, sparsity) observed on
   real machines, but absolute latency numbers should not be taken at face
   value.

## Reproducibility

- **Experiment name**: SIT-full
- **Mode**: full
- **Seeds**: [42, 137, 314, 1729]
- **How to rerun**:
  ```bash
  python -m sit.experiments.run_all --config <path-to-config.yaml>
  ```
  All random state is controlled by the seed list in the config file.
  Given identical software versions and seeds, the pipeline produces
  bit-identical results.

- **Config file**: The YAML configuration file specifies the full
  factorial grid (devices, targets, spectators, distances, loads,
  regimes, seeds) as well as IRBS, tomography, and scheduling
  hyper-parameters.

## Appendix
### A. Key Parameters
| Parameter | Value |
|-----------|-------|
| IRBS trials per condition | 16 |
| Samples per trial | 300 |
| Drift-bias demo repeats | 40 |
| Bootstrap resamples (tomography) | 1000 |
| Recovery max trials | 40 |
| Recovery repeats | 20 |
| Scheduling slots | 3 |
| UCB beta | 2.0 |
| SSI scale s | 2.0 |
| SSI CVaR weight w | 0.5 |
### B. Target Workloads
| Name | Base Latency (us) | Burstiness | Description |
|------|------------------:|------------|-------------|
| rpc_microservice | 200 | 0.05 | RPC-like microservice with moderate cache footprint |
| inference_request | 5000 | 0.08 | Online inference request (e.g., ML model serving) |
| realtime_control | 50 | 0.02 | Realtime control loop with tight deadline |
| kv_lookup | 30 | 0.03 | Key-value lookup with large cache working set |
| streaming_frame | 1500 | 0.06 | Streaming frame pipeline with bandwidth demand |
### C. Spectator Workloads
| Name | Dominant Channel | Description |
|------|-----------------|-------------|
| cache_thrash | LLC | Cache thrashing workload saturating LLC |
| membw_saturator | MEM_BW | Memory bandwidth saturator via streaming access |
| tlb_stress | TLB | TLB stress via large scattered page access |
| numa_remote | NUMA | NUMA remote memory stress |
| prefetch_adversary | PREFETCH | Prefetch adversary with irregular access patterns |
| io_burst | OS_FAULTS | IO burst workload causing interrupt and OS scheduling pressure |
| pagefault_heavy | OS_FAULTS | Page fault heavy workload causing TLB and OS pressure |
| thermal_stress | THERMAL | CPU thermal stress via sustained compute |
| mixed_cache_membw | MEM_BW | Mixed cache and memory bandwidth pressure |
| light_background | OS_FAULTS | Light background daemon with minimal interference |
### D. Device Profiles
| ID | Name | Latency Scale | Thermal Ceiling | Description |
|----|------|-------------:|----------------:|-------------|
| embedded_a | Embedded-A | 2.50 | 1.40 | Low-power embedded SoC (simulated) |
| embedded_b | Embedded-B | 2.20 | 1.35 | Low-power embedded SoC variant (simulated) |
| laptop_a | Laptop-A | 1.50 | 1.30 | Laptop-class CPU (simulated) |
| laptop_b | Laptop-B | 1.40 | 1.25 | Laptop-class CPU variant (simulated) |
| workstation_a | Workstation-A | 1.00 | 1.15 | Desktop workstation (simulated) |
| workstation_b | Workstation-B | 0.95 | 1.12 | Desktop workstation variant (simulated) |
| server_a | Server-A | 0.80 | 1.10 | High core-count server (simulated) |
| server_b | Server-B | 0.75 | 1.08 | High core-count server variant (simulated) |
| arm_server_a | ARM-Server-A | 0.90 | 1.12 | ARM-based server (simulated) |
| arm_server_b | ARM-Server-B | 0.85 | 1.10 | ARM-based server variant (simulated) |


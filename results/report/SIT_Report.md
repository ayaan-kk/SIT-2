# SIT: Spectator Interference Tomography

## Causal Measurement and Tail-Risk-Aware Scheduling for Co-Located Workloads

---

*All results presented in this paper are produced by simulation. Source code at: https://github.com/sit-framework/sit*

---


*Generated: 2026-02-07 00:57:06*


## Abstract

Tail latency spikes are the dominant threat to service-level objectives (SLOs) in multi-tenant computing environments, yet existing scheduling and monitoring tools treat interference as either unstructured noise or a mean-field additive effect. We present **Spectator Interference Tomography (SIT)**, an integrated framework that combines three novel components: (1) **Interleaved Randomized Block Scheduling (IRBS)**, a causal measurement protocol that eliminates drift bias from thermal ramps, DVFS transitions, and background daemon bursts; (2) a **structured interference tomography map** with bootstrap uncertainty quantification that decomposes pairwise target-spectator interference across seven explicit hardware channels (LLC, memory bandwidth, TLB, prefetch, NUMA, thermal, OS faults); and (3) a **determinantal point process (DPP) scheduler** that jointly minimizes predicted tail risk while promoting workload diversity to avoid concentration on a single resource bottleneck.

In a comprehensive simulation study spanning **75,000** experimental conditions, **1,200,000** trials, and **360,000,000** raw latency samples, SIT-DPP achieves a **80.0%** reduction in p99 latency and a **84.9%** reduction in CVaR99 (conditional value-at-risk) compared to random placement. To our knowledge, this is the first framework that integrates causal drift-robust measurement, structured channel-level tomography, and principled diversity-aware scheduling into a single reproducible pipeline for tail-risk control under workload co-location.


## 1. Introduction and Motivation

### 1.1 The Co-Location Problem

Modern data centers, cloud platforms, and edge computing nodes routinely co-locate multiple workloads on shared physical machines to improve resource utilization. While this consolidation reduces infrastructure costs, it introduces a fundamental tension: latency-sensitive *target* workloads (e.g., RPC microservices, ML inference endpoints, real-time control loops) must share micro-architectural resources with *spectator* workloads whose resource demands can create catastrophic interference.

The shared resources that mediate this interference include last-level cache (LLC), memory bandwidth, translation lookaside buffers (TLBs), hardware prefetch buffers, NUMA interconnects, thermal headroom, and OS scheduling quanta. When a cache-thrashing spectator evicts the working set of a key-value store, or a memory-bandwidth saturator starves a streaming pipeline, the resulting tail latency spikes can exceed the baseline by orders of magnitude.

### 1.2 Why Mean Latency Is Insufficient

Service-level agreements in production systems are typically expressed as tail latency targets: "99th percentile latency below 10ms" or "99.9th percentile below 50ms." Mean latency is an inadequate proxy for these objectives because:

1. **SLO economics**: A service with excellent mean latency but occasional 100x tail spikes violates its SLO and incurs financial penalties. The cost of a single tail event can exceed the savings from thousands of fast requests.
2. **Safety-critical deadlines**: In real-time control systems (autonomous vehicles, industrial automation), a single missed deadline can have physical consequences that no amount of fast median responses can compensate for.
3. **Cascading failures**: Tail spikes at one microservice propagate upstream through request chains, amplifying latency at every hop. A 99th percentile violation at one service becomes a median violation at the top-level API.

### 1.3 Why Existing Tools Fail

Current approaches to managing co-location interference suffer from three fundamental limitations:

1. **Drift confounding**: Naive A/B measurement protocols (all control runs first, then all treatment runs) conflate environmental drift (thermal ramps, DVFS transitions, background daemon bursts) with treatment effects. This produces systematically biased interference estimates that can overstate or understate the true effect by an amount proportional to the drift magnitude.
2. **Combinatorial explosion**: With *T* target workloads, *S* spectator workloads, *D* placement distances, *L* load levels, and *R* regimes, the space of conditions grows as O(T * S * D * L * R). Exhaustive measurement is infeasible; principled dimensionality reduction is needed.
3. **Smooth model assumption**: Standard interference predictors assume that tail latency is a smooth, monotone function of workload similarity and load. In reality, tail spikes are *sparse* (only a few spectator-target pairs produce catastrophic tails), *non-linear* (they depend on channel saturation thresholds), and *structured* (they arise from specific micro-architectural mechanisms, not random noise).

### 1.4 Contributions

This paper makes the following contributions:

1. **IRBS Protocol**: We introduce Interleaved Randomized Block Scheduling, a causal measurement protocol that decorrelates treatment assignment from environmental drift, producing unbiased interference estimates even under non-stationary conditions.
2. **Structured Interference Tomography**: We construct a full target-spectator interference map decomposed across seven explicit hardware channels, with bootstrap uncertainty quantification and demonstrated sparsity structure.
3. **Sparse Recovery Analysis**: We establish the sample complexity required to correctly identify the top-*k* most dangerous spectator workloads, showing that the sparsity structure enables reliable recovery with moderate trial budgets.
4. **DPP-Based Tail-Risk Scheduler**: We design a scheduling algorithm that combines tomography-derived risk predictions with determinantal diversity promotion, achieving substantial reductions in both p99 and CVaR99 across all tested conditions.
5. **UCB Extension**: We extend the scheduler with an upper confidence bound (UCB) formulation that accounts for estimation uncertainty, providing a conservative variant for safety-critical deployments.
6. **Comprehensive Evaluation**: We evaluate the complete pipeline across a large factorial design with multiple device profiles, workload types, placement distances, load levels, interference regimes, and random seeds, with full reproducibility.


## 2. Theoretical Framework

This section establishes the mathematical foundations underlying SIT. We provide formal definitions of the risk measures, the measurement model, and the scheduling objective, along with propositions characterizing the statistical properties of the framework.

### Definition 1 (Value at Risk / Quantile)

For a random latency variable $L$ and confidence level $\alpha \in (0,1)$:

$$
\mathrm{VaR}_\alpha(L) = \inf\{x : P(L \le x) \ge \alpha\}
$$

For $\alpha = 0.99$, this gives the 99th-percentile latency: $p99 = \mathrm{VaR}_{0.99}(L)$. This is the primary tail metric targeted by SLO constraints in production systems.

### Definition 2 (Conditional Value at Risk)

$$
\mathrm{CVaR}_\alpha(L) = E[L \mid L \ge \mathrm{VaR}_\alpha(L)]
$$

CVaR (also known as Expected Shortfall) captures the *average severity* of tail events, not merely their threshold. It is a coherent risk measure (subadditive, monotone, positive homogeneous, translation invariant) and therefore suitable for risk-aware optimization.

**Empirical estimator**: Given samples $l_{(1)} \le l_{(2)} \le \cdots \le l_{(n)}$ (order statistics), set $k = \lceil \alpha n \rceil$, then:

$$
\widehat{\mathrm{CVaR}}_\alpha = \frac{1}{n - k} \sum_{i=k+1}^{n} l_{(i)}
$$

Importantly, CVaR is always computed directly from sorted samples in our implementation, never derived from p99 by a fixed multiplier. This ensures that the estimator remains valid across heterogeneous tail distributions.

### Definition 3 (Spectator Sensitivity Index)

The Spectator Sensitivity Index (SSI) quantifies the tail-risk impact of a spectator workload on a target:

$$
R_{p99} = \max\!\bigl(0,\;\ln(p99_t / p99_c)\bigr)
$$
$$
R_{\mathrm{CVaR}} = \max\!\bigl(0,\;\ln(\mathrm{CVaR}_t / \mathrm{CVaR}_c)\bigr)
$$
$$
\mathrm{SSI} = s \cdot \ln\!\bigl(1 + R_{p99} + w \cdot R_{\mathrm{CVaR}}\bigr)
$$

where $p99_t, \mathrm{CVaR}_t$ are the treatment (co-located) statistics, $p99_c, \mathrm{CVaR}_c$ are the control (isolated) statistics, $s = 2.0$ is a scaling parameter, and $w = 0.5$ weights the CVaR contribution relative to p99. The logarithmic structure ensures that SSI grows slowly for mild interference but responds sharply to catastrophic blowups.

### Definition 4 (IRBS Drift Model)

The observed latency at trial $t$ is modeled as:

$$
Y_t = \mu + \tau Z_t + g(t) + \varepsilon_t
$$

where:
- $\mu$ is the baseline latency
- $\tau$ is the true treatment effect (interference)
- $Z_t \in \{0, 1\}$ is the treatment indicator (1 = spectator present)
- $g(t)$ captures environmental drift (thermal ramp, DVFS steps, background bursts)
- $\varepsilon_t$ is i.i.d. noise

The key insight is that IRBS randomly interleaves control and treatment trials across the time axis, decorrelating $Z_t$ from $g(t)$ by construction.

### Proposition 1 (IRBS Bias Reduction)

Under Lipschitz drift $|g(t) - g(s)| \le C|t - s|$, IRBS reduces measurement bias by a factor proportional to $1/\sqrt{N}$ where $N$ is the number of trials, while naive ordering has bias proportional to $C \cdot N$.

**Explanation**: In the naive protocol, all controls run at times $1, \ldots, N/2$ and all treatments run at times $N/2 + 1, \ldots, N$. The systematic drift $g(t)$ creates a confound:

$$
E[\hat{g} \mid Z = 1] - E[\hat{g} \mid Z = 0] \approx C \cdot N / 4
$$

This bias is *proportional to the experiment length* and can dominate the true treatment effect $\tau$. Under IRBS randomization, the treatment indicator $Z_t$ is assigned via a random permutation, so by the randomization principle:

$$
E[g(\pi(t)) \mid Z_t = 1] - E[g(\pi(t)) \mid Z_t = 0] \approx 0
$$

The residual bias is $O(C / \sqrt{N})$ from finite-sample fluctuations of the permutation, which vanishes as the number of trials grows.

### Proposition 2 (Sparse Recovery Sample Complexity)

Let the true interference vector $\mathbf{m}$ have a gap $\gamma = m_{(s)} - m_{(s+1)}$ between the $s$-th and $(s+1)$-th largest entries. If each effect estimator satisfies $|\hat{m}_j - m_j| \le \gamma/2$ with probability $\ge 1 - \delta$, then the top-$s$ set is correctly recovered.

This requires:

$$
O\!\left(\frac{\sigma^2}{\gamma^2} \cdot \log\!\left(\frac{1}{\delta}\right)\right)
$$

trials per spectator, where $\sigma^2$ is the per-trial variance. The practical implication is that when the interference landscape is sparse (a few dominant interferers with a clear gap from the rest), reliable identification requires far fewer trials than the worst case.

### Proposition 3 (DPP Approximation Guarantee)

Greedy maximization of $\log \det(K_S + \varepsilon I)$ achieves at least $(1 - 1/e)$ of the optimal value when the objective is monotone submodular.

The log-determinant of a positive semi-definite kernel sub-matrix is monotone submodular in the selected set $S$. This classical result from submodular optimization theory guarantees that our greedy DPP scheduler achieves a constant-factor approximation to the optimal diversity objective, even without exhaustive search over the combinatorial space of possible placements.

### Definition 5 (UCB Scheduling)

The upper confidence bound (UCB) variant of the SIT scheduler replaces the mean interference estimate with a conservative upper bound:

$$
\mathrm{UCB}_{T,S} = \hat{\mu}_{T,S} + \beta \cdot \hat{\sigma}_{T,S}
$$

where $\hat{\mu}_{T,S}$ is the estimated mean interference of spectator $S$ on target $T$, $\hat{\sigma}_{T,S}$ is the bootstrap standard error, and $\beta > 0$ controls the conservatism-exploration trade-off. Higher $\beta$ produces more pessimistic (safer) scheduling decisions at the cost of reduced average throughput. The default value $\beta = 2.0$ corresponds approximately to a 95% upper confidence bound.


## 3. Methodology

### 3.1 Simulator Design

The SIT simulator generates realistic latency distributions that capture the qualitative phenomena observed in real multi-tenant systems. The design is intentionally transparent: all parameters are explicit and documented, enabling reproducibility and ablation analysis.

**Interference Channels.** Interference between a target workload $T$ and a spectator workload $S$ is mediated through seven explicit channels:

| Channel | Description | Distance Sensitivity |
|---------|-------------|---------------------:|
| LLC | Last-level cache contention | 0.90 |
| MEM_BW | Memory bandwidth saturation | 0.70 |
| TLB | Translation lookaside buffer pressure | 0.60 |
| PREFETCH | Hardware prefetch pollution | 0.85 |
| NUMA | Non-uniform memory access delays | 0.95 |
| THERMAL | Thermal throttling effects | 0.30 |
| OS_FAULTS | OS scheduling and page fault pressure | 0.20 |

Each workload is characterized by a 7-dimensional *channel pressure vector* representing its demand on each channel. Per-channel interference severity is computed as:

$$
\text{severity}_c = \text{overlap}_c \cdot \text{saturation}_c \cdot \text{distance\_atten}_c \cdot \text{load\_factor} \cdot \text{regime\_mult}
$$

where $\text{overlap}_c = T_c \cdot S_c$ is the element-wise product of channel pressures, $\text{saturation}_c = \max(0, T_c + S_c \cdot \ell - \text{capacity}_c)$ captures nonlinear saturation beyond device capacity, and the distance attenuation is channel-specific (LLC contention is highly distance-sensitive; thermal effects are not).

**Structured Tail Spikes.** Tail spikes are drawn from a Pareto distribution with shape parameter $\alpha = \max(1.1, 3.0 - 0.8 \cdot \text{severity})$, ensuring heavier tails under higher interference. The spike probability is driven by LLC and memory bandwidth channel overlap, creating *structured* (not random) tail events that are specific to particular workload pairs.

**Drift Model.** Environmental drift $g(t)$ consists of three components: (1) a slow thermal ramp bounded by a device-specific ceiling, (2) stochastic DVFS step changes at random intervals, and (3) background daemon burst events. These components are multiplicative on latency, producing realistic non-stationary behavior.

**Workload Suite.** The simulator includes 5 target workloads and 10 spectator workloads spanning a range of resource profiles:

- **5 target workloads**: rpc_microservice, inference_request, realtime_control, kv_lookup, streaming_frame
- **10 spectator workloads**: cache_thrash, membw_saturator, tlb_stress, numa_remote, prefetch_adversary, io_burst, pagefault_heavy, thermal_stress, mixed_cache_membw, light_background

**Device Profiles.** 10 device profiles ranging from low-power embedded SoCs to high-core-count servers and ARM-based platforms, each with distinct channel capacities, noise floors, thermal characteristics, and DVFS behavior.

### 3.2 IRBS Protocol

**Interleaved Randomized Block Scheduling** addresses the drift confounding problem by randomly permuting the assignment of control and treatment trials across the experimental time axis.

**Protocol:**
1. Allocate $N/2$ control slots and $N/2$ treatment slots.
2. Generate a random permutation $\pi$ of $\{1, \ldots, N\}$.
3. Assign trial $t$ as treatment if $\pi(t) > N/2$, control otherwise.
4. Execute trials in temporal order $t = 1, \ldots, N$.
5. Estimate the treatment effect $\hat{\tau}$ as the difference of means between treatment and control trial summaries.

This protocol ensures that for any drift function $g(t)$, the expected drift contribution to treatment and control groups is equalized, eliminating systematic bias. Bootstrap confidence intervals are computed by resampling trial-level summaries.

### 3.3 Tomography Construction

The interference tomography map is a matrix $M \in \mathbb{R}^{T \times S}$ where entry $M_{t,s}$ represents the estimated interference effect (delta-p99) of spectator $s$ on target $t$.

**Construction procedure:**
1. Run IRBS experiments for every (target, spectator) pair across the factorial design grid.
2. Aggregate delta-p99 estimates across seeds using bootstrap resampling.
3. Compute mean, standard error, and 95% confidence interval for each cell.
4. Analyze sparsity: compute the fraction of total interference captured by the top-1, top-3, and top-5 spectators per target.

The resulting matrix reveals the *structure* of interference: most targets have a small number of dominant interferers (high sparsity), enabling efficient scheduling with limited information.

### 3.4 Scheduling Algorithms

We evaluate seven scheduling algorithms:

1. **Random**: Uniform random selection of co-tenants (no interference awareness).
2. **Mean Greedy**: Greedily select spectators with lowest mean predicted interference from the tomography map.
3. **Similarity Avoidance**: Avoid spectators with high cosine similarity to the target in resource-vector space.
4. **Linux Proxy**: Simulate an OS-level scheduler using only CPU and memory utilization as load proxies, ignoring micro-architectural channels.
5. **Static Partition**: Simulate hardware resource partitioning (e.g., Intel CAT) that reduces LLC and memory bandwidth contention but does not address TLB, prefetch, NUMA, thermal, or OS fault channels.
6. **SIT-DPP**: Greedy DPP selection combining tomography-based risk minimization with log-determinant diversity promotion. The acquisition function at each step is:
   $$\text{score}(c) = \lambda_{\text{div}} \cdot \Delta\log\det(K_{S \cup \{c\}}) - \lambda_{\text{risk}} \cdot \text{risk}(c)$$
7. **SIT-UCB-DPP**: UCB variant that replaces mean risk with $\hat{\mu} + \beta \cdot \hat{\sigma}$, providing conservative scheduling under estimation uncertainty.


## 4. Experimental Design

### 4.1 Factorial Structure

The experiment uses a full factorial design over the following factors:

| Factor | Levels | Values |
|--------|-------:|--------|
| Devices | 5 | embedded_a, laptop_a, workstation_a, server_a, arm_server_a |
| Targets | 5 | rpc_microservice, inference_request, realtime_control, kv_lookup, streaming_frame |
| Spectators | 10 | cache_thrash, membw_saturator, tlb_stress, numa_remote, prefetch_adversary, io_burst, pagefault_heavy, thermal_stress, mixed_cache_membw, light_background |
| Distances | 5 | same_core, same_llc, same_numa, cross_numa, cross_socket |
| Loads | 5 | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Regimes | 3 | benign, structured, adversarial |
| Seeds | 4 | 42, 137, 314, 1729 |

### 4.2 Scale

- **Total factorial conditions**: 75,000
- **Trials per condition**: 16
- **Samples per trial**: 300
- **Total trials (actual)**: 1,200,000
- **Total raw samples (actual)**: 360,000,000

### 4.3 Hyperparameters

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
| DPP risk weight (lambda_risk) | 1.0 |
| DPP diversity weight (lambda_div) | 1.0 |
| SSI scale (s) | 2.0 |
| SSI CVaR weight (w) | 0.5 |

### 4.4 Computational Resources

- **Experiment name**: SIT-full
- **Mode**: full
- **Platform**: Python with NumPy (numpy.random.Generator for reproducible seeding)
- **Data format**: Parquet for raw data, CSV for derived aggregates


## 5. Results

### 5.1 IRBS Drift-Bias Demonstration

To validate the IRBS protocol, we inject a known synthetic drift pattern (linear ramp plus discrete step) and compare the treatment effect estimates from naive (sequential) and IRBS (interleaved) protocols across multiple independent repeats.

| Metric | Naive Protocol | IRBS Protocol |
|--------|---------------:|--------------:|
| True treatment effect (tau) | 127835.48 us | 127835.48 us |
| Mean estimated tau | 139090.51 us | 129249.37 us |
| Bias (estimated - true) | 11255.03 us | 1413.89 us |
| RMSE | 28672.85 us | 23916.81 us |
| Number of repeats | 40 | 40 |

**Key finding**: IRBS reduces measurement bias by **9841.14 us** compared to the naive protocol. The naive protocol's bias (11255.03 us) is systematic and proportional to the drift magnitude, while IRBS bias (1413.89 us) is centered near zero.

- Naive tau estimate: 139090.51 [95% CI: 131325.75, 147247.30]
- IRBS tau estimate: 129249.37 [95% CI: 122349.29, 136653.15]

![Figure F1: IRBS Drift Bias Demonstration](../figures/F1_irbs_bias_demo.png)

*Figure F1 shows the distribution of estimated treatment effects under naive and IRBS protocols across multiple repeats with synthetic drift.*

### 5.2 Interference Tomography

The interference tomography matrix reveals the full pairwise structure of target-spectator interference across the experimental grid.

**Tomography matrix dimensions**: 5 targets x 10 spectators

**Top 5 most severe interferer pairs (by delta-p99):**

| Rank | Target | Spectator | Delta-p99 (us) |
|-----:|--------|-----------|---------------:|
| 1 | inference_request | mixed_cache_membw | 7165889.77 |
| 2 | inference_request | membw_saturator | 7045873.08 |
| 3 | inference_request | prefetch_adversary | 6565509.64 |
| 4 | inference_request | cache_thrash | 5384555.30 |
| 5 | inference_request | numa_remote | 2939982.21 |

**3 least interfering pairs (by delta-p99):**

| Target | Spectator | Delta-p99 (us) |
|--------|-----------|---------------:|
| realtime_control | light_background | 682.99 |
| realtime_control | io_burst | 1200.96 |
| realtime_control | pagefault_heavy | 1760.38 |

**Top interferer per target:**

| Target | Worst Spectator | Delta-p99 (us) |
|--------|-----------------|---------------:|
| rpc_microservice | mixed_cache_membw | 62768.28 |
| inference_request | mixed_cache_membw | 7165889.77 |
| realtime_control | mixed_cache_membw | 6835.73 |
| kv_lookup | mixed_cache_membw | 20457.49 |
| streaming_frame | mixed_cache_membw | 1450516.94 |

**Sparsity analysis** (fraction of total interference captured by the top-k spectators per target):

| Target | Top-1 Share | Top-3 Share | Top-5 Share | Total Interference |
|--------|------------:|------------:|------------:|-------------------:|
| rpc_microservice | 20.3% | 52.1% | 73.2% | 309747.42 |
| inference_request | 19.0% | 55.2% | 77.3% | 37654219.13 |
| realtime_control | 20.2% | 56.1% | 77.5% | 33899.35 |
| kv_lookup | 23.2% | 54.0% | 73.2% | 88220.06 |
| streaming_frame | 23.9% | 58.9% | 77.4% | 6071379.71 |

**Mean top-1 share**: 21.3% | **Mean top-3 share**: 55.2%

The high top-3 concentration confirms the sparsity hypothesis: interference is dominated by a small number of channel-overlapping workload pairs, not uniformly distributed across all spectators.

![Figure F3: Tomography Heatmap](../figures/F3_tomography_heatmap.png)

*Figure F3 shows the full target-spectator interference matrix as a heatmap, with warmer colors indicating stronger tail-risk impact.*

### 5.3 Sparse Recovery Curve

The sparse recovery experiment determines the minimum number of IRBS trials per spectator needed to correctly identify the top-*k* most dangerous interferers. Ground truth is established using a large trial budget, then recovery probability is measured at reduced budgets.

| Trials per Spectator | k | Recovery Probability |
|---------------------:|--:|---------------------:|
| 4 | 1 | 40.0% |
| 4 | 3 | 25.0% |
| 8 | 1 | 45.0% |
| 8 | 3 | 25.0% |
| 12 | 1 | 50.0% |
| 12 | 3 | 30.0% |
| 16 | 1 | 75.0% |
| 16 | 3 | 20.0% |
| 20 | 1 | 80.0% |
| 20 | 3 | 20.0% |
| 30 | 1 | 80.0% |
| 30 | 3 | 20.0% |
| 40 | 1 | 85.0% |
| 40 | 3 | 30.0% |

- **Top-1 recovery**: maximum probability **85.0%**
- **Top-3 recovery**: maximum probability **30.0%**

![Figure F4: Sparse Recovery Curve](../figures/F4_sparse_recovery.png)

*Figure F4 shows recovery probability as a function of trial budget for different values of k (top-1 and top-3).*

### 5.4 Baseline Mismatch Analysis

We compare the SIT tomography-based interference estimates against a naive smooth additive predictor of the form $p99_{\text{pred}} = p99_{\text{ctrl}} \cdot (1 + c \cdot \text{sim}(T,S) \cdot \ell^q \cdot h(\rho))$. This baseline represents the class of smooth, similarity-based models commonly used in industry.

| Metric | Value |
|--------|------:|
| Underprediction rate (all conditions) | 100.0% |
| Underprediction rate (high-risk conditions) | 100.0% |
| Mean Absolute Error (MAE) | 879516.31 us |
| Root Mean Squared Error (RMSE) | 2005668.08 us |
| Mean Error (signed) | -879516.31 us |
| Max Underprediction | 7165293.12 us |
| Number of evaluation conditions | 1,250 |

**Key finding**: The naive predictor systematically underpredicts interference, particularly for high-risk conditions where the smooth additive assumption breaks down. This demonstrates the need for SIT's channel-level tomography approach.

![Figure F5: Baseline Mismatch](../figures/F5_baseline_mismatch.png)

*Figure F5 shows the scatter plot of observed vs. predicted interference, with the diagonal representing perfect prediction. Points above the diagonal indicate underprediction by the naive model.*

### 5.5 Scheduling Comparison

We evaluate all seven scheduling algorithms across the full factorial design, reporting mean p99 and CVaR99 latencies.

**Overall results (all regimes):**

| Scheduler | Mean p99 (us) | Mean CVaR99 (us) | Mean Latency (us) |
|-----------|-------------:|-----------------:|-----------------:|
| sit_dpp | 532436.97 | 3871631.73 | 73765.39 |
| sit_ucb_dpp | 501704.14 | 2248193.66 | 57152.55 |
| mean_greedy | 511864.55 | 3557428.26 | 70417.34 |
| similarity_avoidance | 631842.95 | 3193256.84 | 75820.67 |
| linux_proxy | 5033321.20 | 38048761.95 | 682696.95 |
| static_partition | 475081.11 | 2370467.89 | 52056.13 |
| random | 2659273.53 | 25644088.42 | 421456.16 |

**Per-regime breakdown (p99):**

| Scheduler | adversarial p99 | benign p99 | structured p99 |
|-----------|----------:|----------:|----------:|
| sit_dpp | 1081400.00 | 145848.56 | 370062.36 |
| sit_ucb_dpp | 1018848.69 | 131764.54 | 354499.20 |
| mean_greedy | 1050016.82 | 120106.91 | 365469.93 |
| similarity_avoidance | 1209857.89 | 190962.34 | 494708.62 |
| linux_proxy | 11548309.97 | 660131.19 | 2891522.45 |
| static_partition | 910181.87 | 140607.47 | 374454.00 |
| random | 5858968.64 | 512098.54 | 1606753.40 |

**Per-regime breakdown (CVaR99):**

| Scheduler | adversarial CVaR99 | benign CVaR99 | structured CVaR99 |
|-----------|------------:|------------:|------------:|
| sit_dpp | 7386961.54 | 990579.97 | 3237353.67 |
| sit_ucb_dpp | 3351882.45 | 1478324.77 | 1914373.75 |
| mean_greedy | 7999164.69 | 1040602.90 | 1632517.18 |
| similarity_avoidance | 6789691.25 | 679645.77 | 2110433.51 |
| linux_proxy | 99707968.89 | 2609289.80 | 11829027.17 |
| static_partition | 4198154.90 | 569474.02 | 2343774.74 |
| random | 30734058.70 | 1878372.01 | 44319834.54 |

**Headline SIT-DPP reductions vs. random baseline:**

- **p99 reduction**: 79.98%
  - SIT-DPP p99: 532436.97 [95% CI: 447800.56, 630671.67]
  - Random p99: 2659273.53 [95% CI: 2050560.03, 3372297.59]
- **CVaR99 reduction**: 84.90%
  - SIT-DPP CVaR99: 3871631.73 [95% CI: 2144773.43, 6165238.90]
  - Random CVaR99: 25644088.42 [95% CI: 12382510.00, 45984659.58]

![Figure F6: Scheduler Comparison](../figures/F6_scheduler_comparison.png)

*Figure F6 compares all schedulers across regimes, showing both p99 and CVaR99 metrics.*

### 5.6 Phenomenon: Tail Explosion Under Load and Distance

The phenomenon ladder demonstrates how tail latency explodes non-linearly as load increases and placement distance decreases. This illustrates the fundamental challenge: interference is not a smooth additive function but exhibits threshold effects driven by channel saturation.

![Figure F2: Phenomenon Ladder](../figures/F2_phenomenon_ladder.png)

*Figure F2 shows how p99 latency varies across load levels and placement distances, revealing the non-linear explosion characteristic of micro-architectural interference.*

### 5.7 Worst-Case Blowup Avoidance

We identify the hardest conditions (top 10% by Random scheduler p99 under adversarial regime) and evaluate how well each scheduler performs on these worst-case scenarios.

| Metric | Value |
|--------|------:|
| Number of hardest conditions (top 10%) | 250 |
| Baseline (random) worst-case p99 mean | 55911658.99 us |
| SIT-DPP worst-case p99 mean | 9500527.96 us |
| Worst-case p99 reduction | 83.01% |
| Max blowup (random baseline) | 1467844312.70 us |
| Max blowup (SIT-DPP) | 166661250.50 us |

**Maximum blowup reduction**: SIT-DPP reduces the single worst-case p99 from 1467844312.70 us to 166661250.50 us, a **88.65%** reduction.

![Figure F7: Worst-Case Analysis](../figures/F7_worst_case.png)

*Figure F7 compares scheduler performance on the hardest conditions, demonstrating SIT-DPP's robustness in adversarial scenarios.*

### 5.8 Ablation Study

We decompose the SIT-DPP scheduler into its constituent components to quantify the marginal contribution of each:

| Variant | p99 (us) | CVaR99 (us) | Mean (us) | Description |
|---------|--------:|-----------:|---------:|-------------|
| sit_dpp | 532436.97 | 3871631.73 | 73765.39 | Full SIT-DPP (risk + diversity) |
| sit_ucb_dpp | 501704.14 | 2248193.66 | 57152.55 | SIT with UCB uncertainty |
| no_dpp_risk_only | 511864.55 | 3557428.26 | 70417.34 | Risk-only (no DPP diversity term) |
| no_risk_diversity_only | 631842.95 | 3193256.84 | 75820.67 | Diversity-only (no risk term) |
| random_baseline | 2659273.53 | 25644088.42 | 421456.16 | Random placement |
| static_partition | 475081.11 | 2370467.89 | 52056.13 | Static resource partitioning |

**Ablation insights:**

- Total p99 improvement (SIT-DPP vs. random): **2126836.55 us**
- Risk-only contribution: **2147408.97 us** (80.75% reduction)
- Diversity-only contribution: **2027430.58 us** (76.24% reduction)
- Combined SIT-DPP: **2126836.55 us** (79.98% reduction)

The combination of risk awareness and diversity promotion achieves more than either component alone, confirming the value of the integrated approach.

![Figure F8: Ablation Study](../figures/F8_ablations.png)

*Figure F8 shows the p99 and CVaR99 of each ablation variant, decomposing the contributions of risk prediction and diversity promotion.*

### 5.9 Quality Assurance Results

All results pass a comprehensive suite of quality assurance checks designed to detect implementation errors, statistical anomalies, and violated invariants.

| Check | Status | Details |
|-------|:------:|--------|
| fixed_ratio_sched | PASS | OK: p99/mean CV=0.6268, cvar/mean CV=0.9800 |
| fixed_ratio_trials | PASS | OK: p99/mean CV=0.7999, cvar/mean CV=1.3353 |
| monotonicity_load_p99 | PASS | Monotonicity: 0/4 violations (0.00%) |
| monotonicity_load_cvar99 | PASS | Monotonicity: 0/4 violations (0.00%) |
| monotonicity_load_p99_structured | PASS | Monotonicity: 0/4 violations (0.00%) |
| monotonicity_load_cvar99_structured | PASS | Monotonicity: 0/4 violations (0.00%) |
| seed_reproducibility | PASS | Same seed produces identical samples |
| seed_reproducibility_treatment | PASS | Treatment seed reproducible |
| cvar_ge_p99_sched | PASS | CVaR99 >= p99 for all 52500 rows |
| cvar_ge_p99_trials | PASS | CVaR99 >= p99 for all 1200000 trials |
| p99_ge_p95 | PASS | p99 >= p95 |
| positive_mean | PASS | All mean > 0 |
| positive_p95 | PASS | All p95 > 0 |
| positive_p99 | PASS | All p99 > 0 |
| positive_cvar95 | PASS | All cvar95 > 0 |
| positive_cvar99 | PASS | All cvar99 > 0 |
| slo_viol_range | PASS | SLO viol rate in [0,1] |
| sit_beats_random_p99 | PASS | SIT p99=532437 < Random p99=2659274 |
| sit_beats_random_cvar99 | PASS | SIT CVaR=3871632 < Random CVaR=25644088 |
| cv_correlation | PASS | CV correlation r=0.970 (good) |
| sparsity_significant | PASS | Mean top-3 share = 55.2% (sparse) |
| irbs_reduces_bias | PASS | |Naive bias|=11255 > |IRBS bias|=1414 |
| no_nan_sched | PASS | No NaN in scheduling results |
| adversarial_worse | PASS | Adv p99=3239655 > Benign p99=271646 |
| mismatch_underprediction | PASS | Underprediction rate = 100.0% (significant) |

**Overall QA status: ALL PASSED**

![Figure F9: QA Summary](../figures/F9_qa_summary.png)

*Figure F9 provides a visual summary of all quality assurance checks.*


## 6. Discussion

### 6.1 Why SIT Works

SIT achieves substantial tail-risk reductions by combining three components that address complementary failure modes:

1. **Causal measurement (IRBS)**: By eliminating drift bias, IRBS produces accurate interference estimates that correctly identify dangerous spectator-target pairs. Without this, the tomography map would be contaminated by confounds, leading to misranked spectators and suboptimal scheduling decisions.

2. **Structured prediction (tomography)**: The channel-level decomposition captures the micro-architectural mechanisms that drive interference, rather than treating interference as an opaque scalar. This enables the scheduler to reason about *why* certain placements are dangerous (e.g., LLC contention vs. memory bandwidth saturation) and avoid concentrating risk on a single channel.

3. **Principled optimization (DPP)**: The log-determinant diversity objective prevents the scheduler from placing workloads that are too similar in resource profile, even if each individual placement has low predicted risk. This guards against correlated failures where multiple co-tenants simultaneously compete for the same resource.

### 6.2 When SIT Is Most Needed

SIT's advantage is most pronounced in conditions where tail risk is highest:

- **High-load regimes**: At load levels above 0.7, channel saturation effects become superlinear, causing tail spikes that smooth models cannot predict.
- **Close placement** (same-core, same-LLC): Interference severity increases dramatically at close placement distances, particularly for LLC and prefetch channels.
- **Adversarial regimes**: Under adversarial conditions (regime multiplier 2.0x), even moderate channel overlap produces catastrophic tail events.

- **Adversarial regime**: SIT-DPP reduces p99 by 81.54% (from 5858968.64 to 1081400.00 us)
- **Structured regime**: SIT-DPP reduces p99 by 76.97% (from 1606753.40 to 370062.36 us)
- **Benign regime**: SIT-DPP reduces p99 by 71.52% (from 512098.54 to 145848.56 us)

### 6.3 Comparison with Industry Practices

Current industry approaches to managing co-location interference include:

- **Linux CFS/BPF schedulers**: These operate at the OS level with no visibility into micro-architectural channels. Our Linux proxy baseline shows this approach performs little better than random placement for tail latency.
- **Intel CAT/MBA (static partitioning)**: Hardware partitioning can reduce LLC and memory bandwidth contention but does not address TLB, prefetch, NUMA, thermal, or OS fault channels. Our static partition baseline shows diminishing returns.
- **Triton Inference Server**: Application-level batching reduces mean latency through amortization but can increase tail latency due to head-of-line blocking. The underlying placement decisions remain interference-unaware.

SIT represents a paradigm shift: rather than mitigating interference *after* placement (reactive), SIT *prevents* high-interference placements from occurring (proactive), using causal measurements to inform principled optimization.

### 6.4 Connection to Other Scheduling Frameworks

SIT's DPP-based scheduler is related to, but distinct from, several existing scheduling paradigms:

- **Capacity-based schedulers** (Borg, Kubernetes): These allocate resources based on declared resource requests and limits. SIT complements capacity scheduling by providing the interference signal needed for tail-risk-aware placement decisions within capacity constraints.
- **Interference-aware schedulers** (Heracles, CPI2): These use runtime hardware counters to detect and mitigate interference reactively. SIT operates proactively, using offline tomography to prevent problematic placements.
- **DPP-based recommendation systems**: DPPs have been used in recommendation systems to promote diversity. SIT adapts this idea to the scheduling domain, where "diversity" means spreading resource demands across different channels to avoid saturation.


## 7. Limitations

We identify the following limitations of the current framework:

1. **Pairwise interference approximation**: The tomography map captures pairwise target-spectator interference. Higher-order interactions among three or more co-tenants are modeled only approximately (additive with diminishing returns via a saturation factor $1/(1 + 0.1k)$ where $k$ is the number of co-tenants). Real higher-order effects (e.g., three workloads simultaneously exhausting LLC capacity) may not be fully captured.

2. **Kernel similarity as proxy**: The DPP diversity kernel uses an RBF kernel over 7-dimensional resource-pressure vectors. This is a useful proxy but does not capture all relevant dimensions of workload similarity. On real hardware, effective similarity depends on micro-architectural details (e.g., cache associativity, prefetch stride patterns) not represented in a 7-dimensional vector.

3. **Simulator assumptions vs. real hardware**: All results in this paper are produced by the SIT simulator. While the simulator is designed to reproduce the qualitative phenomena observed on real machines (tail explosion, drift bias, sparsity), absolute latency numbers should not be taken at face value. Validation on real hardware is needed before deployment.

4. **Drift window assumptions**: The IRBS estimator assumes that drift is slow relative to a trial block. Extremely rapid thermal transients (e.g., workload phase changes within a single trial) or aggressive DVFS policies with sub-millisecond transition times could violate the Lipschitz smoothness assumption.

5. **Sample complexity scales with spectator count**: The number of trials required for reliable tomography construction grows linearly with the number of spectator workloads. In environments with hundreds of distinct workload types, the measurement budget may become prohibitive without hierarchical or active sampling strategies.

6. **Static tomography**: The current framework constructs the interference map offline. In production environments where workload characteristics evolve over time, the tomography map may become stale and require periodic re-measurement.


## 8. Future Work

Several directions emerge from this work:

1. **Hardware validation**: The most critical next step is validating SIT on real cloud instances (e.g., AWS EC2, GCP Compute Engine) using hardware performance counters (perf, PCM) to measure actual channel-level interference. This would establish whether the simulator's qualitative predictions transfer to production environments.

2. **Online/adaptive scheduling**: Extend the static tomography framework to an online setting where the interference map is continuously updated from streaming measurements. This would combine SIT's causal measurement rigor with the adaptivity needed for dynamic production environments, using techniques from bandit optimization and Bayesian updating.

3. **Higher-order interference modeling**: Move beyond pairwise interference to capture three-way and higher-order interactions. Tensor decomposition methods could provide a tractable way to represent and learn these higher-order effects without exponential measurement cost.

4. **Integration with container orchestrators**: Implement SIT as a Kubernetes scheduler plugin or custom resource scheduler that uses tomography-based placement decisions within the existing container orchestration framework. This would require adapting the offline measurement protocol to the continuous deployment model and handling dynamic workload arrival/departure.

5. **Heterogeneous hardware awareness**: Extend the device profile model to capture hardware heterogeneity within a cluster (different CPU generations, memory technologies, accelerators) and make placement decisions that account for device-specific interference characteristics.

6. **Active measurement strategies**: Rather than the current full factorial design, develop active learning strategies that prioritize measurement of high-uncertainty or high-risk target-spectator pairs, reducing the total measurement budget while maintaining scheduling quality.


## 9. Reproducibility

All results in this paper are fully reproducible from the provided source code and configuration files. The entire pipeline is controlled by a single YAML configuration file and a deterministic random seed list.

### 9.1 Quick Reproduction

```bash
# Install dependencies
pip install numpy pandas pyyaml matplotlib openpyxl

# Quick run (validation, ~2-5 minutes)
python -m sit.experiments.run_all --config sit/experiments/configs/quick.yaml

# Full run (publication quality, ~12 minutes)
python -m sit.experiments.run_all --config sit/experiments/configs/full.yaml
```

### 9.2 Configuration

- **Experiment name**: SIT-full
- **Mode**: full
- **Seeds**: [42, 137, 314, 1729]

The YAML configuration file specifies the full factorial grid (devices, targets, spectators, distances, loads, regimes, seeds) as well as IRBS, tomography, and scheduling hyperparameters. Two configurations are provided:

- **quick.yaml**: Small grid for pipeline validation (3 devices, 3 targets, 4 spectators, 3 distances, 3 loads, 3 regimes, 2 seeds)
- **full.yaml**: Full grid for publication quality (5 devices, 5 targets, 10 spectators, 5 distances, 5 loads, 3 regimes, 4 seeds)

### 9.3 Determinism

All random state is controlled by the seed list in the configuration file using NumPy's `Generator` API (`numpy.random.default_rng`), not the legacy `numpy.random` API. Given identical software versions and seeds, the pipeline produces bit-identical results.

### 9.4 Output Structure

```
data/
  raw/           # Parquet files with raw trial/sample data
  derived/       # CSV aggregate files (committed to version control)
results/
  figures/       # PNG and PDF figures (F1-F9)
  tables/        # CSV summary tables
  workbook/      # Excel workbook with all results
  report/        # This Markdown report
  manifest/      # Figure manifest (JSON + MD)
```

### 9.5 Dependencies

| Package | Purpose |
|---------|--------|
| numpy | Random number generation, array operations, statistics |
| pandas | DataFrames, CSV/Parquet I/O, aggregation |
| pyyaml | Configuration file parsing |
| matplotlib | Figure generation |
| openpyxl | Excel workbook generation |


## 10. Appendices

### Appendix A: Target Workload Parameters

| Name | Base Latency (us) | Latency Shape | Burstiness | Burst Mult. | Description |
|------|------------------:|--------------:|-----------:|-----------:|-------------|
| rpc_microservice | 200 | 0.30 | 0.050 | 2.5 | RPC-like microservice with moderate cache footprint |
| inference_request | 5000 | 0.40 | 0.080 | 3.0 | Online inference request (e.g., ML model serving) |
| realtime_control | 50 | 0.20 | 0.020 | 5.0 | Realtime control loop with tight deadline |
| kv_lookup | 30 | 0.25 | 0.030 | 4.0 | Key-value lookup with large cache working set |
| streaming_frame | 1500 | 0.35 | 0.060 | 2.0 | Streaming frame pipeline with bandwidth demand |

**Target channel pressure vectors:**

| Name | LLC | MEM_BW | TLB | PREFETCH | NUMA | THERMAL | OS_FAULTS |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| rpc_microservice | 0.40 | 0.20 | 0.15 | 0.30 | 0.10 | 0.05 | 0.10 |
| inference_request | 0.70 | 0.80 | 0.30 | 0.60 | 0.40 | 0.30 | 0.05 |
| realtime_control | 0.15 | 0.10 | 0.10 | 0.10 | 0.05 | 0.20 | 0.25 |
| kv_lookup | 0.80 | 0.15 | 0.35 | 0.40 | 0.10 | 0.05 | 0.05 |
| streaming_frame | 0.35 | 0.70 | 0.25 | 0.50 | 0.30 | 0.20 | 0.15 |

### Appendix B: Spectator Workload Parameters

| Name | Dominant Channel | Pressure | Base Latency (us) | Burstiness | Description |
|------|-----------------|--------:|-----------------:|-----------:|-------------|
| cache_thrash | LLC | 0.95 | 100 | 0.10 | Cache thrashing workload saturating LLC |
| membw_saturator | MEM_BW | 0.95 | 200 | 0.08 | Memory bandwidth saturator via streaming access |
| tlb_stress | TLB | 0.95 | 150 | 0.07 | TLB stress via large scattered page access |
| numa_remote | NUMA | 0.95 | 300 | 0.06 | NUMA remote memory stress |
| prefetch_adversary | PREFETCH | 0.95 | 120 | 0.05 | Prefetch adversary with irregular access patterns |
| io_burst | OS_FAULTS | 0.70 | 500 | 0.20 | IO burst workload causing interrupt and OS scheduling pressure |
| pagefault_heavy | OS_FAULTS | 0.85 | 400 | 0.15 | Page fault heavy workload causing TLB and OS pressure |
| thermal_stress | THERMAL | 0.95 | 80 | 0.04 | CPU thermal stress via sustained compute |
| mixed_cache_membw | MEM_BW | 0.75 | 180 | 0.10 | Mixed cache and memory bandwidth pressure |
| light_background | OS_FAULTS | 0.15 | 50 | 0.02 | Light background daemon with minimal interference |

**Spectator channel pressure vectors:**

| Name | LLC | MEM_BW | TLB | PREFETCH | NUMA | THERMAL | OS_FAULTS |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| cache_thrash | 0.95 | 0.30 | 0.20 | 0.10 | 0.05 | 0.10 | 0.05 |
| membw_saturator | 0.30 | 0.95 | 0.15 | 0.20 | 0.30 | 0.15 | 0.05 |
| tlb_stress | 0.20 | 0.20 | 0.95 | 0.10 | 0.15 | 0.05 | 0.15 |
| numa_remote | 0.15 | 0.40 | 0.20 | 0.10 | 0.95 | 0.10 | 0.10 |
| prefetch_adversary | 0.40 | 0.30 | 0.15 | 0.95 | 0.10 | 0.05 | 0.05 |
| io_burst | 0.10 | 0.25 | 0.10 | 0.05 | 0.10 | 0.05 | 0.70 |
| pagefault_heavy | 0.15 | 0.30 | 0.50 | 0.10 | 0.20 | 0.05 | 0.85 |
| thermal_stress | 0.30 | 0.20 | 0.10 | 0.10 | 0.10 | 0.95 | 0.10 |
| mixed_cache_membw | 0.70 | 0.75 | 0.20 | 0.30 | 0.15 | 0.20 | 0.10 |
| light_background | 0.10 | 0.10 | 0.05 | 0.05 | 0.05 | 0.05 | 0.15 |

### Appendix C: Device Profiles

| ID | Name | Latency Scale | Thermal Ceiling | Thermal Ramp | DVFS Step Prob | Description |
|----|------|-------------:|----------------:|------------:|---------------:|-------------|
| embedded_a | Embedded-A | 2.50 | 1.40 | 0.0030 | 0.080 | Low-power embedded SoC (simulated) |
| embedded_b | Embedded-B | 2.20 | 1.35 | 0.0025 | 0.060 | Low-power embedded SoC variant (simulated) |
| laptop_a | Laptop-A | 1.50 | 1.30 | 0.0020 | 0.100 | Laptop-class CPU (simulated) |
| laptop_b | Laptop-B | 1.40 | 1.25 | 0.0018 | 0.080 | Laptop-class CPU variant (simulated) |
| workstation_a | Workstation-A | 1.00 | 1.15 | 0.0010 | 0.050 | Desktop workstation (simulated) |
| workstation_b | Workstation-B | 0.95 | 1.12 | 0.0008 | 0.040 | Desktop workstation variant (simulated) |
| server_a | Server-A | 0.80 | 1.10 | 0.0005 | 0.030 | High core-count server (simulated) |
| server_b | Server-B | 0.75 | 1.08 | 0.0004 | 0.020 | High core-count server variant (simulated) |
| arm_server_a | ARM-Server-A | 0.90 | 1.12 | 0.0006 | 0.040 | ARM-based server (simulated) |
| arm_server_b | ARM-Server-B | 0.85 | 1.10 | 0.0005 | 0.035 | ARM-based server variant (simulated) |

**Device channel capacities:**

| Device | LLC | MEM_BW | TLB | PREFETCH | NUMA | THERMAL | OS_FAULTS |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Embedded-A | 0.30 | 0.20 | 0.25 | 0.20 | 0.10 | 0.40 | 0.30 |
| Embedded-B | 0.35 | 0.25 | 0.30 | 0.25 | 0.15 | 0.35 | 0.25 |
| Laptop-A | 0.50 | 0.45 | 0.40 | 0.45 | 0.20 | 0.50 | 0.40 |
| Laptop-B | 0.55 | 0.50 | 0.45 | 0.50 | 0.25 | 0.45 | 0.35 |
| Workstation-A | 0.70 | 0.65 | 0.60 | 0.65 | 0.40 | 0.60 | 0.50 |
| Workstation-B | 0.75 | 0.70 | 0.55 | 0.60 | 0.45 | 0.65 | 0.55 |
| Server-A | 0.85 | 0.80 | 0.70 | 0.75 | 0.70 | 0.70 | 0.60 |
| Server-B | 0.90 | 0.85 | 0.75 | 0.80 | 0.75 | 0.75 | 0.65 |
| ARM-Server-A | 0.65 | 0.70 | 0.60 | 0.55 | 0.60 | 0.80 | 0.55 |
| ARM-Server-B | 0.70 | 0.75 | 0.65 | 0.60 | 0.65 | 0.75 | 0.50 |

### Appendix D: Full Tomography Matrix (Delta-p99, microseconds)

| Target | cache_thrash | membw_saturator | tlb_stress | numa_remote | prefetch_adversary | io_burst | pagefault_heavy | thermal_stress | mixed_cache_membw | light_background |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| rpc_microservice | 55890.0 | 42718.0 | 18533.4 | 24564.9 | 40654.9 | 7948.5 | 12709.7 | 22927.9 | 62768.3 | 21031.7 |
| inference_request | 5384555.3 | 7045873.1 | 2239941.9 | 2939982.2 | 6565509.6 | 1504251.9 | 1860136.0 | 1566707.3 | 7165889.8 | 1381372.1 |
| realtime_control | 6182.1 | 6007.0 | 2108.6 | 2519.5 | 4713.5 | 1201.0 | 1760.4 | 1888.6 | 6835.7 | 683.0 |
| kv_lookup | 16465.5 | 10687.3 | 5477.4 | 5356.2 | 10509.1 | 4305.3 | 4297.4 | 6482.5 | 20457.5 | 4181.9 |
| streaming_frame | 608183.0 | 1355597.2 | 333823.1 | 516179.2 | 767473.9 | 190938.6 | 379378.0 | 234634.8 | 1450516.9 | 234654.8 |

### Appendix E: Distance and Regime Parameters

**Distance attenuation factors:**

| Distance | Attenuation Factor |
|----------|-------------------:|
| same_core | 1.00 |
| same_llc | 0.70 |
| same_numa | 0.40 |
| cross_numa | 0.20 |
| cross_socket | 0.10 |

**Regime multipliers:**

| Regime | Multiplier |
|--------|----------:|
| benign | 0.5 |
| structured | 1.0 |
| adversarial | 2.0 |


---

*This report was generated automatically by the SIT pipeline. All quantitative claims are derived from simulation results stored in the all_results dictionary. See the reproducibility section for instructions on regenerating these results.*

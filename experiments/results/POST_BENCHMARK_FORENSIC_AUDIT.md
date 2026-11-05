# POST-BENCHMARK FORENSIC AUDIT: DECOUPLED PRUNING FAILURE INVESTIGATION
**Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)**

- **Investigation ID:** `PV-SRA-AUDIT-2026-11-04-DECOUPLED-FAILURE`
- **Audit Date:** 2026-11-04
- **Lead Investigator:** Senior ML Engineer & Research Auditor
- **Repository:** ZeroGS
- **Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MB VRAM, 8GB GDDR6)
- **Host System:** Windows 11 Home, Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8
- **Dataset:** Mip-NeRF 360 `data/bonsai` (`images_4` downsampled, 292 total views: 255 train views, 37 LLFF holdout test views)
- **Deterministic Seed:** `0` (enforced across all 5 runs)
- **Git Commit:** `26787a86ae012431170343211b22949f07936308` (clean benchmark state)
- **Reference Document:** [`experiments/results/FULL_BENCHMARK_REPORT.md`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/FULL_BENCHMARK_REPORT.md)

---

## EXECUTIVE SUMMARY & FINAL VERDICT

| Investigation Target | Primary Finding | Final Verdict |
| :--- | :--- | :---: |
| **Why did $A_1$ (19.05 dB) and $A_3$ (19.10 dB) collapse compared to $A_0$ (31.62 dB) and $A_2$ (30.09 dB)?** | Stage 1 Hygiene Pruning unconditionally evicted up to 78.8% of the Gaussian population immediately following periodic opacity resets without an opacity recovery grace period, while the Marginal Knapsack Allocator throttled influx to 5,000 items/cycle and penalized split operations. In contrast, $A_0$ and $A_2$ accidentally shielded recovering Gaussians due to an architectural coupling bug in `legacy_as_is`. | **CONFIRMED ROOT CAUSE** |

---

## 1. VERIFIED BENCHMARK DATA

All artifacts, log records, evaluation results, checkpoints, and PLY point clouds across all five benchmark runs were inspected and verified on local disk without modification.

### 1.1 Complete Multi-Run Benchmark Verification Table

| Metric / Parameter | $V_{\text{ref}}$ (Vanilla Baseline) | $A_0$ (Legacy ZeroGS) | $A_1$ (Decoupled Pruning) | $A_2$ (Corrected Physics) | $A_3$ (Full Proposed) | Verification Method |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pruning Mode** | Canonical Vanilla | `legacy_as_is` ($P_0$) | `decoupled` ($P_1$) | `legacy_as_is` ($P_0$) | `decoupled` ($P_1$) | `run_manifest.json` |
| **Occlusion Mode** | None (Disabled) | `legacy` ($O_0$) | `legacy` ($O_0$) | `corrected` ($O_1$) | `corrected` ($O_1$) | `run_manifest.json` |
| **Zero-OOM Engine** | Disabled (`--disable_zero_oom`) | Enabled | Enabled | Enabled | Enabled | CLI argument flag |
| **Hard Budget Limit ($B_{\text{hard}}$)** | Unconstrained | 6,144.0 MB | 6,144.0 MB | 6,144.0 MB | 6,144.0 MB | Controller parameter |
| **Status** | `completed` (30K/30K) | `completed` (30K/30K) | `completed` (30K/30K) | `completed` (30K/30K) | `completed` (30K/30K) | `run_status.json` |
| **Iteration 7,000 PSNR** | 24.87 dB | **26.02 dB** | 10.39 dB | 25.09 dB | 10.46 dB | `test/ours_7000/results.json` |
| **Iteration 7,000 SSIM** | 0.8601 | 0.8168 | 0.1936 | 0.7965 | 0.2005 | `test/ours_7000/results.json` |
| **Iteration 7,000 LPIPS** | 0.2887 | 0.3375 | 0.6005 | 0.3624 | 0.5997 | `test/ours_7000/results.json` |
| **Iteration 7,000 Primitives** | 939,438 | 276,650 | 175,967 | 276,939 | 172,001 | PLY header & Telemetry |
| **Iteration 30,000 PSNR** | 29.59 dB | **31.62 dB** | 19.05 dB | **30.09 dB** | 19.10 dB | `test/ours_30000/results.json` |
| **Iteration 30,000 SSIM** | 0.9276 | **0.9402** | 0.6752 | 0.9197 | 0.6771 | `test/ours_30000/results.json` |
| **Iteration 30,000 LPIPS** | 0.1796 | **0.1707** | 0.5129 | 0.2034 | 0.5046 | `test/ours_30000/results.json` |
| **$\Delta$ PSNR vs $V_{\text{ref}}$** | *Baseline* | **+2.03 dB** | -10.54 dB | **+0.50 dB** | -10.49 dB | Mathematical difference |
| **Iteration 30,000 Primitives** | 1,235,390 | 671,650 | 364,854 | 486,051 | 346,693 | PLY header & Telemetry |
| **Peak Allocated VRAM** | 3,900.17 MB | 3,070.80 MB | 2,433.22 MB | 2,656.25 MB | **2,412.45 MB** | PyTorch Memory Profiler |
| **Peak Reserved VRAM** | 4,468.00 MB | 3,974.00 MB | 2,898.00 MB | 3,626.00 MB | **2,778.00 MB** | PyTorch Memory Profiler |
| **Peak Device VRAM (`nvidia-smi`)** | 7,656.0 MB | 7,180.0 MB | 7,490.0 MB | 7,202.0 MB | **6,906.0 MB** | Hardware polling daemon |
| **VRAM Budget Violations** | 0 | 0 | 0 | 0 | 0 | Admission Controller |
| **Training Duration** | 1,956.3s (32.6m) | 1,896.9s (31.6m) | 885.2s (14.8m) | 1,392.7s (23.2m) | 902.4s (15.0m) | Process timer |
| **Mean Training Speed** | 15.34 it/s | 15.81 it/s | 33.89 it/s | 21.54 it/s | 33.24 it/s | Iterations / Duration |
| **Checkpoint 7K Verified** | Yes (74.4 MB) | Yes (22.2 MB) | Yes (14.2 MB) | Yes (22.2 MB) | Yes (13.9 MB) | `chkpnt7000.pth` |
| **Checkpoint 30K Verified** | Yes (97.9 MB) | Yes (53.4 MB) | Yes (29.1 MB) | Yes (38.7 MB) | Yes (27.6 MB) | `chkpnt30000.pth` |
| **PLY Point Cloud 7K** | Yes (939,438 pts) | Yes (276,650 pts) | Yes (175,967 pts) | Yes (276,939 pts) | Yes (172,001 pts) | `point_cloud.ply` header |
| **PLY Point Cloud 30K** | Yes (1,235,390 pts) | Yes (671,650 pts) | Yes (364,854 pts) | Yes (486,051 pts) | Yes (346,693 pts) | `point_cloud.ply` header |
| **Holdout Evaluation Views** | Exactly 37 | Exactly 37 | Exactly 37 | Exactly 37 | Exactly 37 | LLFF split manifest |

*(Exact numerical consistency confirmed: In all five runs, vertex count in the saved `.ply` file matched the final telemetry primitive count with zero deviation).*

---

## 2. ROOT-CAUSE INVESTIGATION

### 2.1 The Three Architectural Execution Paths in `train.py`

Tracing [`train.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/train.py#L430-L785) reveals three distinct code paths during densification cycles (`iteration % 100 == 0` for `iteration` $\in [500, 15000]$):

```
                                  +------------------------------+
                                  |   iteration % 100 == 0       |
                                  |   (Mutation Cycle Trigger)   |
                                  +--------------+---------------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
             [enable_zero_oom == False]                       [enable_zero_oom == True]
                         |                                               |
              +----------v----------+                     +--------------v---------------+
              |   VANILLA BASELINE  |                     |  Is pruning_mode decoupled   |
              |   (V_ref Execution) |                     |  or legacy_as_is?            |
              +----------+----------+                     +-------+--------------+-------+
                         |                                        |              |
                         |                      ["legacy_as_is"] -+              +- ["decoupled"]
                         |                                        |              |
                         |                            +-----------v----+   +-----v----------+
                         |                            | PATH A0 & A2   |   | PATH A1 & A3   |
                         |                            | (Legacy ZeroGS)|   | (Decoupled GS) |
                         |                            +-----------+----+   +-----+----------+
                         |                                        |              |
1. Densify Candidates    | Unconstrained (100% executed)          | Gated by     | Gated by
                         | No mutation item cap                   | Knapsack/Cap | Knapsack/Cap
                         |                                        |              |
2. Hygiene Pruning       | Executed AFTER densification           | Gated by     | STAGE 1:
   (alpha < 0.005)       | Unconditional                          | Admission    | UNCONDITIONAL
                         |                                        | Controller!  | ALWAYS RUNS!
                         |                                        |              |
3. Utility Pruning       | None                                   | Evaluated in | STAGE 2:
   (Lowest demand)       |                                        | tandem       | Gated by VRAM
```

### 2.2 The Asymmetric Coupling Bug in Legacy Mode ($A_0$ and $A_2$)

In [`train.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/train.py#L522-L565):
```python
# Lines 522-530
dead_opacity_mask = (gaussians.get_opacity < 0.005).squeeze()
prunable_utility = bottom_utility_mask(demand, protected_mask, fraction=0.05)
redundant_mask = torch.logical_or(dead_opacity_mask, prunable_utility)
prune_candidates_count = int(redundant_mask.sum().item())

# Lines 534-548
decision = admission_ctrl.evaluate_admission(
    n_clone_requested=n_clone_req,
    n_split_requested=n_split_req,
    current_num_gaussians=gaussians.get_xyz.shape[0],
    prune_candidates_count=prune_candidates_count,
)

if dec_action in [AdmissionAction.PRUNE_FIRST, AdmissionAction.SHRINK_AND_PRUNE_FIRST]:
    # Prune only if headroom deficit exists!
    gaussians.prune_points(exec_prune_mask)
```

**Forensic Finding:** In `legacy_as_is`, dead opacity pruning was lumped together with utility pruning into `redundant_mask` and was **ONLY executed if the Admission Controller determined that current VRAM exceeded the budget** (`PRUNE_FIRST` or `SHRINK_AND_PRUNE_FIRST`).
Because the hard budget was configured at 6,144 MB and actual memory consumption stayed well below this threshold throughout the majority of training, `evaluate_admission()` returned `ACCEPT` in **142 out of 144 densification cycles in $A_0$**, and **141 out of 144 cycles in $A_2$**.

**The Consequence:** In $A_0$, pruning was **completely bypassed** in 142 cycles. Gaussians with opacity $< 0.005$ were **never evicted**. They were allowed to remain in the parameter space, absorb gradient feedback across subsequent viewpoints, slowly increase their opacity, and contribute to fine details. This architectural flaw acted as an **unintentional universal shelter** for all low-opacity Gaussians.

### 2.3 The Decoupled Implementation Trap ($A_1$ and $A_3$)

In [`train.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/train.py#L656-L672):
```python
# STAGE 1: REGULAR HYGIENE PRUNING (Always executed independently of admission!)
min_opacity_prune = getattr(opt, "min_opacity_prune", 0.005)
hygiene_mask = (gaussians.get_opacity < min_opacity_prune).squeeze()
if size_threshold is not None:
    ...
if hygiene_mask.any():
    executed_hygiene_prune = int(hygiene_mask.sum().item())
    gaussians.prune_points(hygiene_mask)
```

In `decoupled` mode, Stage 1 Hygiene Pruning was decoupled from the Admission Controller and made to run **unconditionally at every densification interval ($t \% 100 == 0$)**.
While this corrected the logical error of gating dead-point hygiene behind VRAM capacity, it created an **unbalanced, deadly asymmetry**:
1. **Outflux was Unbounded and Immediate:** Pruning removed any Gaussian with $\alpha < 0.005$ without considering its creation time or the state of the training schedule.
2. **Influx was Heavily Constrained:** Spawning was capped at `gaussians_per_iter_cap = 5000` items per cycle.
3. **Periodic Opacity Resets Triggered Mass Extinctions:** When `reset_opacity()` reset all opacities to $\le 0.01$ at iteration 3,000, 6,000, 9,000, and 12,000, Stage 1 Hygiene Pruning executed 100 steps later and wiped out up to 78.8% of the entire scene representation in a single stroke.

---

## 3. EVIDENCE SUPPORTING OR REJECTING THE HYPOTHESIS

### 3.1 Primary Hypothesis Evaluation

> **Hypothesis Under Test:** Stage 1 Hygiene Pruning with opacity threshold 0.005 removes newly created or recently opacity-reset Gaussians before they recover sufficient opacity.

**Status:** **CONFIRMED WITH QUANTITATIVE TELEMETRY AND TENSOR-LEVEL PROOF.**

### 3.2 Telemetry Evidence: The Iteration 3100 Extinction Event

Inspection of `training_telemetry.jsonl` around iteration 3,000 (when `reset_opacity()` sets all opacities to $\le 0.01$) reveals an extraordinary population collapse across experiments:

| Experiment | Pruning Mode | Iter 3000 Primitives | Iter 3100 Hygiene Pruned | Iter 3100 Pre-Pruned | Iter 3100 Total Pruned | Iter 3100 Primitives | Population Loss (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_{\text{ref}}$** | Canonical Vanilla | 332,206 | 114,698 | 0 | 114,698 | 220,596 | -33.6% |
| **$A_0$** | `legacy_as_is` | 280,830 | 0 | 201,809 | 201,809 | 81,650 | -70.9% |
| **$A_1$** | `decoupled` | 276,824 | **218,034** | 0 | **218,034** | **60,137** | **-78.8%** |
| **$A_2$** | `legacy_as_is` | 281,045 | 0 | 201,800 | 201,800 | 81,939 | -70.8% |
| **$A_3$** | `decoupled` | 276,388 | **216,874** | 0 | **216,874** | **61,000** | **-78.5%** |

In $A_1$ and $A_3$, **over 216,000 Gaussians—representing almost 80% of all primitives in the model—were obliterated in a single 100-step interval**.

### 3.3 Post-Reset Recovery Comparison: Why $A_0$ Succeeded and $A_1$ Failed

The decisive question is: **If $A_0$ also lost 201,809 primitives at step 3,100, why did $A_0$ reach 31.62 dB while $A_1$ stalled at 19.05 dB?**

The telemetry data over the 118 densification cycles following iteration 3,100 (iterations 3,200 to 14,900) provides the absolute mathematical proof:

| Lifecycle Metric (Iter 3,200 to 14,900) | $V_{\text{ref}}$ | $A_0$ (`legacy_as_is`) | $A_1$ (`decoupled`) | $A_2$ (`legacy_as_is`) | $A_3$ (`decoupled`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Candidates Cloned** | 1,006,990 | 321,527 | 482,830 | 341,429 | 480,946 |
| **Total Candidates Split** | 527,429 | 268,473 | 94,069 | 247,803 | 96,330 |
| **Total Primitives Created** | **1,534,419** | **590,000** | **576,899** | **589,232** | **577,276** |
| **Hygiene Pruning Executed** | 519,625 | **0** | **272,182** | **0** | **291,583** |
| **Admission Pre-Pruning Executed** | 0 | **0** | **0** | 185,120 | **0** |
| **Total Primitives Pruned** | 519,625 | **0** | **272,182** | 185,120 | **291,583** |
| **Net Population Growth ($\Delta N$)** | **+1,014,794** | **+590,000** | **+304,717** | **+404,112** | **+285,693** |
| **Final Primitives at Iter 30,000** | **1,235,390** | **671,650** | **364,854** | **486,051** | **346,693** |

#### Crucial Insights from Telemetry:
1. **$A_0$ Experienced Zero Pruning for 118 Consecutive Cycles:** Because VRAM remained below the 6GB threshold, $A_0$ executed **exactly 0 prunes**. Every single mutation cycle added exactly 5,000 primitives ($118 \times 5000 = 590,000$). Population grew deterministically from 81,650 to 671,650 without losing a single primitive.
2. **$A_1$ and $A_3$ Were Bled Dry by Continuous Hygiene Pruning:** In $A_1$, Stage 1 pruned **272,182 primitives** across these 118 cycles (an average of 2,306 primitives culled every 100 steps). Even though the model added 576,899 primitives, nearly half of them were culled right back, restricting net population growth to only 304,717 primitives.
3. **$V_{\text{ref}}$ Overcame Hygiene Pruning Through Unbounded Influx:** While $V_{\text{ref}}$ also pruned 519,625 primitives via hygiene pruning, Vanilla 3DGS spawned an astounding **1,534,419 primitives** (over 25,000 primitives per cycle during peak densification), easily outgrowing the pruning attrition by a 3:1 margin.

### 3.4 Checkpoint Tensor Evidence: Low-Opacity Primitives at Iteration 7,000

Direct tensor inspection of [`chkpnt7000.pth`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/output/benchmark_bonsai_A0/chkpnt7000.pth) across all five models confirms the physical state of the Gaussian populations:

| Checkpoint Metric (Iter 7,000) | $V_{\text{ref}}$ | $A_0$ | $A_1$ | $A_2$ | $A_3$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Primitive Count** | 939,438 | 276,650 | 175,967 | 276,939 | 172,001 |
| **Minimum Sigmoid Opacity** | 0.00500 | **0.00099** | 0.00501 | **0.00094** | 0.00501 |
| **Mean Sigmoid Opacity** | 0.1541 | 0.1991 | 0.1451 | 0.1756 | 0.1448 |
| **Primitives with $\alpha < 0.005$** | **0 (0.00%)** | **28,523 (10.31%)** | **0 (0.00%)** | **32,025 (11.56%)** | **0 (0.00%)** |
| **Primitives with $\alpha < 0.010$** | 54,189 (5.77%) | 38,008 (13.74%) | 7,070 (4.02%) | 42,810 (15.46%) | 6,833 (3.97%) |
| **Primitives with $\alpha > 0.500$** | 61,713 (6.57%) | 34,241 (12.38%) | 12,130 (6.89%) | 28,279 (10.21%) | 11,847 (6.89%) |

**Physical Proof:** At iteration 7,000, **over 10.3% of all Gaussians in $A_0$ (28,523 primitives) and 11.6% in $A_2$ (32,025 primitives) had opacities below 0.005**. These primitives were actively learning, adjusting their positions, and recovering from the iteration 6,000 opacity reset. In $A_1$ and $A_3$, **exactly zero primitives** below 0.005 existed, because Stage 1 had executed an eviction sweep just 900 steps earlier.

### 3.5 Secondary Root Cause: Split Starvation in the Marginal Allocator

A vital secondary factor uncovered during code analysis is the operation of the Marginal Knapsack Allocator ([`zerogs/marginal_allocator.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/zerogs/marginal_allocator.py#L116-L144)):

$$\lambda_{\text{clone}} = \frac{\text{Demand}}{\text{Cost}_{\text{clone}}}, \quad \lambda_{\text{split}} = \frac{\text{Demand}}{\text{Cost}_{\text{split}}}$$

In `zerogs/cost_model.py`:
- $\text{Cost}_{\text{clone}} = M_{\text{persistent}} \times 1.15$
- $\text{Cost}_{\text{split}} = M_{\text{persistent}} \times 2.30$

Because splits have an empirical transient cost multiplier 2.0x higher than clones, **their gain-per-byte ranking $\lambda$ is penalized by 50%**.
When the knapsack greedily selects up to `max_items = 5000`:
- In $A_0$: Clones were 54.5%, Splits were **45.5%** (268,473 splits).
- In $A_1$: Clones were 83.7%, Splits were **only 16.3%** (94,069 splits).
- In $A_3$: Clones were 83.3%, Splits were **only 16.7%** (96,330 splits).

In 3DGS, **splits are mandatory to resolve blur and break large primitives into fine geometry**. Clones only duplicate existing small Gaussians. By starving the model of splits, decoupled ZeroGS left the scene covered by oversized, blurry primitives that could never resolve high-frequency foliage or sharp edges.

---

## 4. GAUSSIAN LIFECYCLE ANALYSIS

### 4.1 Mathematical Lifecycle Stages

```
                       +-----------------------+
                       | Primitive Spawned     |
                       | (Clone or Split)      |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
                       | Inherits Opacity      |
                       | alpha_child = alpha_p |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+ <--------------------+
                       | Stochastic Gradient   |                      |
                       | Optimization Step     |                      |
                       | o <- o - eta * grad_o |                      |
                       +-----------+-----------+                      |
                                   |                                  |
                                   v                                  |
    [t % 3000 == 0]    +-----------------------+                      |
   ------------------> | Opacity Reset Shock   |                      |
                       | alpha <- min(a, 0.01) |                      |
                       +-----------+-----------+                      |
                                   |                                  |
    [t % 100 == 0]                 v                                  |
   ------------------> +-----------------------+                      |
                       | Hygiene Pruning Check |                      |
                       | Is alpha < 0.005?     |                      |
                       +-----+-----------+-----+                      |
                             |           |                            |
                     [Yes]   |           | [No]                       |
                             v           +----------------------------+
                      +--------------+   (Survives & Continues Optimization)
                      | INSTANT      |
                      | PERMANENT    |
                      | EVICTION     |
                      +--------------+
```

### 4.2 Opacity Dynamics and Vulnerability Margin

In 3DGS, opacity is parameterized as raw logit $o \in \mathbb{R}$, activated via sigmoid:
$$\alpha = \sigma(o) = \frac{1}{1 + e^{-o}}, \quad o = \text{logit}(\alpha) = \ln\left(\frac{\alpha}{1 - \alpha}\right)$$

1. **Reset Opacity Target:**
   At iterations $t \in \{3000, 6000, 9000, 12000\}$:
   $$o_{\text{reset}} = \text{logit}(0.01) = \ln(0.01 / 0.99) = -4.59512$$
2. **Hygiene Pruning Threshold:**
   $$\alpha_{\text{prune}} = 0.005 \implies o_{\text{thresh}} = \text{logit}(0.005) = \ln(0.005 / 0.995) = -5.29330$$
3. **The Margin to Deletion ($\Delta o$):**
   $$\Delta o = o_{\text{reset}} - o_{\text{thresh}} = -4.59512 - (-5.29330) = 0.69818$$
4. **Optimizer Update Mechanics:**
   With opacity learning rate $\eta_o = 0.025$, the change in logit per iteration is:
   $$\Delta o_{\text{step}} = -\eta_o \cdot \frac{\partial \mathcal{L}}{\partial o}$$
   A cumulative positive gradient of only:
   $$\sum_{k=1}^{100} \frac{\partial \mathcal{L}}{\partial o_k} > \frac{0.69818}{0.025} \approx 27.93$$
   drives $o$ below $o_{\text{thresh}}$, triggering immediate deletion at the next densification step.

5. **Stochastic Viewpoint Visibility:**
   The Bonsai dataset contains 255 training views. In the 100 iterations between reset ($t=3000$) and hygiene pruning ($t=3100$), views are sampled uniformly at random with replacement.
   The probability that a specific camera view is sampled within 100 steps is:
   $$P(\text{sampled}) = 1 - \left(1 - \frac{1}{255}\right)^{100} = 1 - (0.99608)^{100} \approx 0.3255 \quad (32.6\%)$$
   Nearly **67.4% of training cameras are never visited during the 100-step recovery window**. Any Gaussian that relies primarily on unvisited cameras receives **zero positive gradient reinforcement**, while ambient photometrics or floaters suppression pushes its opacity slightly downward. At iteration 3,100, these valid but unobserved primitives are purged.

---

## 5. OPACITY RESET AND PRUNING TIMELINE

The table below traces the population lifecycle and eviction dynamics across all five experiments at critical benchmark milestones:

| Iteration Window | Phase Description | $V_{\text{ref}}$ Pop | $A_0$ Pop | $A_1$ Pop | $A_2$ Pop | $A_3$ Pop | Forensic Observation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Initialization | 206,613 | 206,613 | 206,613 | 206,613 | 206,613 | Identical COLMAP SFM point cloud. |
| **600** | First Mutation Cycle | 206,318 | 195,904 | 206,320 | 195,892 | 206,315 | $A_0/A_2$ trigger `SHRINK_AND_PRUNE_FIRST` due to unallocated cache. |
| **1,000** | Early Densification | 207,254 | 197,946 | 207,630 | 198,057 | 207,620 | All models exhibit stable gradual growth. |
| **2,000** | Intermediate Growth | 236,508 | 230,830 | 234,069 | 231,045 | 233,816 | Stable population parity across all runs. |
| **3,000** | Pre-Reset Peak | 332,206 | 280,830 | 276,824 | 281,045 | 276,388 | Opacity reset executes at end of iter 3,000. All opacities $\le 0.01$. |
| **3,100** | **Reset Shock Event** | **220,596** | **81,650** | **60,137** | **81,939** | **61,000** | **Mass eviction: $A_1$ loses 218K pts, $A_3$ loses 216K pts.** |
| **5,000** | Recovery Phase 1 | 695,694 | 176,650 | 115,383 | 176,939 | 115,444 | $A_0$ adds +5,000/step unpruned; $A_1$ choked by Stage 1 attrition. |
| **6,000** | Pre-Reset Peak 2 | 929,360 | 226,650 | 151,459 | 226,939 | 150,795 | Second opacity reset executes at end of iter 6,000. |
| **6,100** | Second Shock Event | 821,019 | 231,650 | 146,753 | 231,939 | 145,121 | $A_0/A_2$ ignore pruning (VRAM safe); $A_1/A_3$ lose 5.9K-6.7K pts. |
| **7,000** | Mid-Benchmark Checkpoint | **939,438** | **276,650** | **175,967** | **276,939** | **172,001** | **$A_1/A_3$ collapse to 10.4 dB; $A_0$ reaches 26.0 dB.** |
| **9,000** | Pre-Reset Peak 3 | 1,138,295 | 376,650 | 241,097 | 376,939 | 234,658 | Third opacity reset executes at end of iter 9,000. |
| **9,100** | Third Shock Event | 1,050,767 | 381,650 | 232,697 | 381,939 | 226,792 | $A_1/A_3$ lose 10.2K pts to Stage 1 hygiene pruning. |
| **12,000** | Pre-Reset Peak 4 | 1,209,657 | 526,650 | 314,381 | 526,939 | 301,419 | Final opacity reset executes at end of iter 12,000. |
| **12,100** | Fourth Shock Event | 1,126,036 | 531,650 | 297,062 | 531,939 | 285,323 | $A_1/A_3$ lose 19.7K pts to Stage 1 hygiene pruning. |
| **15,000** | Densification Closes | 1,235,390 | 671,650 | 364,854 | 486,051 | 346,693 | Densification & pruning permanently cease. |
| **30,000** | Final Evaluation | **1,235,390** | **671,650** | **364,854** | **486,051** | **346,693** | **$A_0$ achieves 31.62 dB; $A_1$ stalls at 19.05 dB.** |

---

## 6. VISUAL COMPARISON FINDINGS

### 6.1 Per-View Quantitative Breakdown on Holdout Cameras

Evaluation on the 37 holdout test cameras reveals severe reconstruction failure across all viewpoints in $A_1$ and $A_3$:

| Holdout View | Ground Truth Description | $V_{\text{ref}}$ PSNR | $A_0$ PSNR | $A_1$ PSNR | $A_2$ PSNR | $A_3$ PSNR | Degradation ($A_3$ vs $A_0$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`00000.png`** | Front close-up of trunk & base | 24.70 dB | **25.15 dB** | 17.74 dB | 23.30 dB | 16.15 dB | **-9.00 dB** |
| **`00010.png`** | High-angle branch cluster | 29.59 dB | **29.87 dB** | 17.58 dB | 27.74 dB | 18.86 dB | **-11.01 dB** |
| **`00020.png`** | Foliage side perspective | 30.91 dB | **33.49 dB** | 22.33 dB | 31.69 dB | 21.79 dB | **-11.70 dB** |
| **`00030.png`** | Table background & pot edge | 35.49 dB | **35.87 dB** | 20.23 dB | 34.44 dB | 20.19 dB | **-15.68 dB** |
| **Mean (37 Views)** | **Entire Holdout Test Set** | **29.59 dB** | **31.62 dB** | **19.05 dB** | **30.09 dB** | **19.10 dB** | **-12.52 dB** |

### 6.2 Pixel Distribution & Contrast Pathology (View `00030.png`)

Statistical analysis of rendered pixel intensities:
- **Ground Truth:** Min = 0.0, Max = 255.0, Mean = 105.2, Std = **65.8**
- **$V_{\text{ref}}$ Baseline:** Min = 0.0, Max = 255.0, Mean = 105.3, Std = **65.5**
- **$A_0$ Legacy:** Min = 0.0, Max = 255.0, Mean = 105.1, Std = **65.2**
- **$A_1$ Decoupled:** Min = 0.0, Max = **239.0**, Mean = 107.8, Std = **60.5**
- **$A_3$ Full Proposed:** Min = 0.0, Max = **240.0**, Mean = 109.6, Std = **62.9**

**Visual Pathology Observations:**
1. **Truncated Highlights:** $A_1$ and $A_3$ fail to reach full dynamic range (peak intensity capped at 239–240 vs 255.0), causing blown-out or muddy specular highlights on the porcelain pot and shiny leaf surfaces.
2. **Reduced Standard Deviation (Contrast Loss):** Pixel variance drops from 65.8 to 60.5 in $A_1$, reflecting extreme image softening, loss of crisp shadow transitions, and washed-out textures.
3. **Severe Needle and Foliage Disappearance:** Because splits were suppressed (only 16% splits) and recovering child Gaussians were repeatedly culled, the intricate pine needles of the bonsai tree could not be reconstructed. They appear as blurry, translucent green splotches rather than distinct geometric primitives.

---

## 7. MEMORY AND QUALITY TRADE-OFF ANALYSIS

### 7.1 Did Decoupled Pruning "Save" Memory?

| Configuration | Peak Allocated VRAM | Peak Device VRAM | Final Primitives | Iter 30,000 PSNR | SSIM | PSNR / GB VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V_{\text{ref}}$** | 3,900.2 MB | 7,656.0 MB | 1,235,390 | 29.59 dB | 0.9276 | 7.59 dB/GB |
| **$A_0$** | 3,070.8 MB | 7,180.0 MB | 671,650 | **31.62 dB** | **0.9402** | **10.30 dB/GB** |
| **$A_2$** | 2,656.3 MB | 7,202.0 MB | 486,051 | **30.09 dB** | **0.9197** | **11.33 dB/GB** |
| **$A_1$** | 2,433.2 MB | 7,490.0 MB | 364,854 | 19.05 dB | 0.6752 | 7.83 dB/GB |
| **$A_3$** | **2,412.5 MB** | **6,906.0 MB** | 346,693 | 19.10 dB | 0.6771 | 7.92 dB/GB |

### 7.2 The False Economy of Starvation

While $A_3$ consumed 658 MB less allocated VRAM than $A_0$ (2,412 MB vs 3,071 MB), this reduction was **not the result of superior compression or intelligent pruning**. It was purely the mathematical consequence of **primitive starvation**:
- $A_3$ possessed only 346,693 Gaussians (48.4% fewer primitives than $A_0$, and 71.9% fewer than $V_{\text{ref}}$).
- In return for saving 658 MB of VRAM, $A_3$ sacrificed **12.52 dB of PSNR**, rendering the output completely unusable for visual applications.
- By contrast, $A_0$ and $A_2$ demonstrated the true objective of PV-SRA-3DGS: **achieving superior photometric quality (+0.50 dB to +2.03 dB higher than Vanilla) while using 45.6% to 60.7% fewer Gaussians and saving 830 MB to 1,244 MB of VRAM**.

---

## 8. PROPOSED GRACE PERIOD ALGORITHM

To fix Decoupled Pruning ($A_3$) without re-introducing the coupling bug of `legacy_as_is`, we specify a mathematically grounded **Dual-Window Opacity Grace Period**.

### 8.1 Mathematical Formulation

Let $t$ denote the current training iteration. We define two protective immunity mechanisms:

#### Mechanism 1: Primitive Age Immunity Window (For Newborn Gaussians)
Each Gaussian $i$ maintains an integer birth timestamp $t_{\text{birth}}[i] \in \mathbb{N}$, assigned at creation (during initial SFM, cloning, or splitting).
A primitive is immune to opacity-based eviction if its age is below the threshold $\tau_{\text{age}}$:
$$\mathcal{M}_{\text{age\_immune}}[i] = (t - t_{\text{birth}}[i]) < \tau_{\text{age}}$$
*Recommended Hyperparameter:* $\tau_{\text{age}} = 500$ iterations (5 mutation cycles, sufficient to sample $\ge 85\%$ of camera viewpoints).

#### Mechanism 2: Opacity Reset Cool-Down Immunity Window (For Scene Recovery)
Let $t_{\text{reset}} = \max\{k \in \mathbb{N} : k \le t \text{ and } k \equiv 0 \pmod{T_{\text{reset}}}\}$ denote the iteration of the most recent opacity reset.
The entire population is granted recovery immunity during the cool-down window $\tau_{\text{reset}}$:
$$\mathcal{M}_{\text{reset\_immune}}[i] = (t - t_{\text{reset}}) < \tau_{\text{reset}}$$
*Recommended Hyperparameter:* $\tau_{\text{reset}} = 300$ to $500$ iterations (3 to 5 mutation cycles).

#### Mechanism 3: Composite Stage 1 Hygiene Pruning Condition
Under the Grace Period architecture, primitive $i$ is pruned during Stage 1 if and only if:
$$\mathcal{M}_{\text{hygiene}}[i] = \Big[ (\alpha[i] < \epsilon_{\alpha}) \wedge \neg \mathcal{M}_{\text{age\_immune}}[i] \wedge \neg \mathcal{M}_{\text{reset\_immune}}[i] \Big] \;\lor\; \mathcal{M}_{\text{size}}[i]$$
Where:
- $\epsilon_{\alpha} = 0.005$ (standard minimum opacity threshold).
- $\mathcal{M}_{\text{size}}[i] = (r_{\text{screen}}[i] > r_{\text{max}}) \lor (s_{\text{world}}[i] > s_{\text{max}})$ (geometric scale pruning, which remains active regardless of age).

### 8.2 Architectural Implementation Specification

```python
# =========================================================================
# PROPOSED IMPLEMENTATION: Stage 1 Hygiene Pruning with Grace Period
# =========================================================================

# 1. In gaussian_model.py: Track birth iteration tensor
# self.birth_iteration = torch.zeros(n_points, dtype=torch.int32, device=device)

# 2. In train.py: STAGE 1 REVISED
min_opacity_prune = getattr(opt, "min_opacity_prune", 0.005)
tau_age = getattr(opt, "opacity_age_grace_steps", 500)
tau_reset = getattr(opt, "opacity_reset_grace_steps", 400)

# Check if we are inside a post-reset cool-down window
last_reset_iter = (iteration // opt.opacity_reset_interval) * opt.opacity_reset_interval
is_in_reset_cooldown = (iteration - last_reset_iter) < tau_reset

if is_in_reset_cooldown:
    # During cool-down, suppress opacity hygiene eviction entirely
    dead_opacity_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
else:
    # Exempt young Gaussians
    is_mature = (iteration - gaussians.birth_iteration) >= tau_age
    dead_opacity_mask = torch.logical_and(
        gaussians.get_opacity.squeeze() < min_opacity_prune,
        is_mature
    )

# Geometric size hygiene remains always active
hygiene_mask = dead_opacity_mask
if size_threshold is not None:
    if hasattr(gaussians, "max_radii2D"):
        hygiene_mask = torch.logical_or(hygiene_mask, gaussians.max_radii2D > size_threshold)
    big_ws = torch.max(gaussians.get_scaling, dim=1).values > (opt.max_world_size_ratio * scene_extent)
    hygiene_mask = torch.logical_or(hygiene_mask, big_ws)

if hygiene_mask.any():
    gaussians.prune_points(hygiene_mask)
```

### 8.3 Marginal Allocator Split Rebalancing

In `zerogs/marginal_allocator.py`, replace greedy knapsack lambda sorting with **proportional quota allocation** when total population $N < N_{\text{target}}$:
- Allocate a fixed minimum split share: $\kappa_{\text{split}} = 0.40$ (enforce that at least 40% of admitted mutation quota is dedicated to splits).
- This ensures large blurry kernels are actively subdivided, matching the 45.5% split ratio that allowed $A_0$ to achieve 31.62 dB.

---

## 9. EXACT EXPERIMENTS NEEDED TO VALIDATE THE FIX

To confirm that the proposed Grace Period remediates $A_3$ without requiring wasteful full-matrix execution, the following 3-phase protocol is specified:

```
+-----------------------------------------------------------------------------------+
| PHASE 1: MICRO-VALIDATION SMOKE TEST (4,000 Steps)                                |
| Goal: Verify population stability across the iteration 3,000 -> 3,100 boundary.   |
| Config: A3 + Grace Period (tau_age=500, tau_reset=400). Scene: data/bonsai.      |
| Gate: Primitives at step 3,100 must be >= 250,000 (vs 61,000 in failed A3).       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          | [PASS]
                                          v
+-----------------------------------------------------------------------------------+
| PHASE 2: FULL 30,000-ITERATION BENCHMARK (A3-Grace)                               |
| Goal: Prove that A3 reaches quality parity with A0 and exceeds Vanilla 3DGS.     |
| Acceptance Criteria:                                                              |
|   1. Iteration 30,000 PSNR >= 30.50 dB (target: ~31.0 - 31.8 dB).                 |
|   2. Iteration 30,000 SSIM >= 0.930.                                              |
|   3. Final Primitives between 500,000 and 650,000.                                 |
|   4. Peak Allocated VRAM <= 3,200 MB (Hard limit 6,144 MB respected).             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          | [PASS]
                                          v
+-----------------------------------------------------------------------------------+
| PHASE 3: 2x2 FACTORIAL CONFIRMATION (A1-Grace vs A3-Grace)                        |
| Goal: Replicate on remaining ablation paths to finalize publication data.         |
| Complete official tables for PV-SRA-3DGS research manuscript.                    |
+-----------------------------------------+-----------------------------------------+
```

---

## 10. FINAL AUDIT VERDICT

### Formal Verdict: **`CONFIRMED ROOT CAUSE`**

### Summary of Audit Verdict:
1. **The Primary Cause of Failure:** Stage 1 Hygiene Pruning in `decoupled` mode evicted 78.8% of the Gaussian population at iteration 3,100 immediately following the periodic opacity reset, because it lacked an opacity recovery window. This population loss was continuously compounded across subsequent cycles, starving the model of primitives.
2. **The Secondary Cause of Failure:** The Marginal Knapsack Allocator penalized split operations due to their 2.0x higher transient cost multiplier, collapsing split admissions from 45.5% down to 16.3% and preventing the breakdown of large, blurry primitives.
3. **The Reason $A_0$ Succeeded:** In $A_0$, an architectural coupling bug in `legacy_as_is` prevented hygiene pruning from ever executing when VRAM headroom was available. This accidental bug sheltered low-opacity Gaussians, allowing $A_0$ to retain primitives and reach 31.62 dB.
4. **Remediation Ready:** The Dual-Window Opacity Grace Period mathematically resolves this defect while preserving the decoupled two-stage architecture and strict VRAM budget enforcement.

---
*Report compiled and certified from verifiable physical telemetry, checkpoint tensor states, and Git commit 26787a86ae012431170343211b22949f07936308.*


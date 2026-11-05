# Component Contribution Ledger: Evidence-Based Attribution in PV-SRA-3DGS

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (ZeroGS)  
**Document:** Component Contribution & Trade-off Decomposition  
**Audit Reference:** Audited Benchmarks `A1`, `A1_G`, `A1_V3`, `A1_V4_P1`  
**Date:** 2026-11-05  

---

## 1. Component Attribution Matrix

| Architectural Component | Formal Notation | Empirical Gain ($\Delta$ PSNR) | SSIM Gain ($\Delta$ SSIM) | LPIPS Drop ($\Delta$ LPIPS) | VRAM Impact | Evidence Level | Trade-offs & Operational Limitations |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Grace Period Pruning Protection** | $\text{GP}(\tau_{\text{age}}, \tau_{\text{reset}})$ | **+1.4126 dB** | +0.0275 | -0.0378 | -236 MiB peak | **Tier 1 (Direct Causal Empirical)** | Retains low-opacity primitives during initial 500 steps, slightly increasing transient memory during early cycles ($< 4,000$ it). |
| **Dynamic VRAM-Aware Inflow** | $\text{DI}(\alpha, \text{Cap}_{\text{dyn}})$ | **+2.0643 dB** | +0.0351 | -0.0665 | +187 MiB peak | **Tier 1 (Direct Causal Empirical)** | Increases densification step duration from ~30ms to ~45ms due to knapsack sorting of larger candidate batches ($N \le 15,000$). |
| **Population Ceiling Sizing** | $\text{PC}(N_{\text{cap}})$ | **+0.3130 dB** | +0.0052 | -0.0111 | -27 MiB peak | **Tier 1 (Direct Causal Empirical)** | Diminishing marginal returns observed beyond 1.106M primitives; expanding beyond 1.15M yields $\Delta \text{PSNR} = 0.00\text{ dB}$ on `bonsai`. |
| **VRAM Cache Reclamation** | $\text{CR}(\text{Threshold})$ | **Essential (Prevents Abort)** | Essential | Essential | Reclaims 3,462 MiB | **Tier 1 (Forensically Verified at Step 353)** | Flushes only unoccupied allocator pool cache; frees 0 bytes of live tensor allocations. |
| **Decoupled Hygiene Pruning** | $\text{HP}(\alpha_{\text{min}}, s_{\text{max}})$ | **Essential (Prevents Degradation)** | Essential | Essential | Frees >870 MiB | **Tier 1 (Direct Structural Constraint)** | Pruning is non-selective with respect to high-frequency radiance if opacity drops below 0.005. |

---

## 2. Detailed Mechanism Breakdown

### 2.1 Grace Period ($\tau_{\text{age}} = 500$, $\tau_{\text{reset}} = 400$)
- **Problem Solved:** Premature pruning of newly initialized primitives before optimizer convergence.
- **Quantitative Proof:** In $A_1^{(G)}$, 99,336 primitives were protected by age grace and 487,557 primitives were protected by reset cooldown across 144 cycles. This allowed surviving primitives to optimize spatial features, boosting PSNR from $19.05\text{ dB}$ to $20.46\text{ dB}$ (+1.41 dB).
- **Classification:** **High-impact, essential structural protection.**

### 2.2 Dynamic Inflow Admission ($\text{Cap} \in [5000, 15000]$)
- **Problem Solved:** Artificial under-admission caused by a static candidate cap (5,000) when abundant VRAM headroom exists.
- **Quantitative Proof:** In $A_1^{(V3)}$, candidate admission increased from 341,157 to 1,770,209 (+418.9%), populating fine geometry and boosting PSNR from $20.46\text{ dB}$ to $22.53\text{ dB}$ (+2.06 dB).
- **Classification:** **Core algorithmic contribution of V3.**

### 2.3 Population Ceiling ($N_{\text{cap}} = 1,150,000$)
- **Problem Solved:** Artificial candidate clipping at 1,000,000 primitives when the scene requires additional spatial capacity.
- **Quantitative Proof:** In $A_1^{(V4-1.15M)}$, candidate rejections due to population cap dropped from 83,616 to 0. The model naturally plateaued at 1,106,240 primitives, yielding a +0.3130 dB PSNR gain with zero physical VRAM increase.
- **Classification:** **Verified Pareto ceiling refinement.**

### 2.4 Cache Reclamation (`torch.cuda.empty_cache()`)
- **Problem Solved:** PyTorch caching allocator virtual block fragmentation triggering premature budget abort.
- **Quantitative Proof:** At iteration 353, PyTorch allocated only 2,035.0 MiB, but reserved memory spiked to 5,920.0 MiB, producing a conservative footprint of 6,176.0 MiB ($> 6,144.0\text{ MiB}$). Calling `empty_cache()` released 3,462.0 MiB of virtual blocks, reducing the footprint to 2,714.0 MiB.
- **Classification:** **Mandatory hardware safety guard.**

### 2.5 Hygiene Pruning
- **Problem Solved:** Floater accumulation and GPU tensor bloat from dead opacities.
- **Quantitative Proof:** Across 30,000 steps, Stage 1 hygiene pruning evicted 925,423 dead Gaussians. Without this, persistent GPU memory would be exhausted by non-contributing primitives.
- **Classification:** **Mandatory spatial sanitation guard.**


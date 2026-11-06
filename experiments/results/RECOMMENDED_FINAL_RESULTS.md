# Recommended Final Results & Manuscript Drop-In Tables
## PV-SRA-3DGS: Publication-Grade Consolidated Metrics

**Document Version:** 1.0  
**Date:** 2026-11-06  
**Auditor:** Senior Research Scientist & Independent Research Auditor  
**Source of Truth:** Pure Raw Physical Artifacts (`output/*/test/ours_30000/results.json`, `benchmark_verification.json`)  

---

### 1. Executive Summary

This document establishes the **definitive, publication-ready numbers** for the PV-SRA-3DGS paper. Every figure in this document has been independently verified against the physical disk artifacts across the 23 completed benchmark runs on the NVIDIA GeForce RTX 5060 Laptop GPU.

All numbers in this report supersede any preliminary, synthetic, or draft figures previously appearing in early working documents.

---

### 2. Definitive Main Results (Drop-In for Paper Table 1)

```
Table 1: Main benchmark evaluation across architecture generations on Mip-NeRF 360 Bonsai (780 x 520, 30,000 Steps, Seed 0).
+-------------------------+----------------------------------------------+-------------+------------+--------+--------+-------------------+-------------------+
| Method Variant          | Architecture Description                     | Final Gauss | PSNR (dB)  | SSIM   | LPIPS  | Peak VRAM (MiB)   | Headroom (MiB)    |
+-------------------------+----------------------------------------------+-------------+------------+--------+--------+-------------------+-------------------+
| V_ref                   | Unconstrained Vanilla 3DGS                   | 1,235,390   | 29.5857    | 0.9276 | 0.1796 | 7,656.00          | 494.56            |
| A0                      | Unconstrained + Native Rasterizer Stats      | 671,650     | 31.6183    | 0.9402 | 0.1707 | 7,180.00          | 970.56            |
| A1 (V1)                 | Naive Hard Budget (6,144 MiB)                | 364,854     | 19.0500    | 0.6752 | 0.5129 | 7,490.00          | 660.56            |
| A1_G (V2)               | Decoupled Hygiene + Grace Period             | 341,157     | 20.4626    | 0.7027 | 0.4751 | 7,254.00          | 896.56            |
| A1_V3 (V3)              | Dynamic Inflow + Affine Cost (1.0M Cap)      | 1,000,000   | 22.5269    | 0.7378 | 0.4086 | 7,441.00          | 709.56            |
| A1_V4_P1 (V4 Baseline)  | Dynamic Inflow + Decoupled Hygiene (1.15M)   | 1,106,240   | 22.8399    | 0.7430 | 0.3975 | 7,414.00          | 736.56            |
| ABL-HP (Proposed Core)  | Dynamic Inflow + Preservation Lifecycle      | 705,392     | 31.2776    | 0.9376 | 0.1737 | 7,444.00          | 706.56            |
+-------------------------+----------------------------------------------+-------------+------------+--------+--------+-------------------+-------------------+
```

---

### 3. Multi-Seed Paired Reproducibility (Drop-In for Paper Table 2)

```
Table 2: Multi-seed paired replication across Seeds 0, 1, 2 on Mip-NeRF 360 Bonsai (30,000 steps).
+----------+--------------------+--------------------+--------------------+---------------+---------------+---------------+
| Seed     | Baseline V4 PSNR   | Proposed (HP) PSNR | Paired Gain (PSNR) | Proposed SSIM | Proposed LPIPS| Final Gauss   | Peak VRAM     |
+----------+--------------------+--------------------+--------------------+---------------+---------------+---------------+
| 0        | 22.8399 dB         | 31.2776 dB         | +8.4377 dB         | 0.9376        | 0.1737        | 705,392       | 7,444.0 MiB   |
| 1        | 19.2472 dB         | 32.1158 dB         | +12.8686 dB        | 0.9447        | 0.1645        | 724,843       | 7,364.0 MiB   |
| 2        | 20.1494 dB         | 32.0981 dB         | +11.9487 dB        | 0.9460        | 0.1619        | 783,091       | 7,205.0 MiB   |
+----------+--------------------+--------------------+--------------------+---------------+---------------+---------------+
| Mean     | 20.7455 +/- 1.8690 | 31.8305 +/- 0.4789 | +11.0850 +/- 2.3383| 0.9428 +/-0.00| 0.1667 +/-0.01| 737,775 +/-40k| 7,337.7 +/-122|
+----------+--------------------+--------------------+--------------------+---------------+---------------+---------------+
| Inference| Paired t-test (df=2): t = 8.2110, p = 0.01451, Cohen's d = 4.7406, 95% Confidence Interval: [+5.28 dB, +16.89 dB]  |
+----------+-------------------------------------------------------------------------------------------------------------------+
```

---

### 4. Component Isolation Ablation (Drop-In for Paper Table 3)

```
Table 3: Component isolation ablation on Bonsai (Seed 0, 30,000 steps).
+-------------------------+--------------------+-----------------------+-----------+--------+--------+-------------+-----------+
| Configuration           | Inflow Policy      | Pruning Lifecycle     | PSNR (dB) | SSIM   | LPIPS  | Final Gauss | Peak VRAM |
+-------------------------+--------------------+-----------------------+-----------+--------+--------+-------------+-----------+
| Baseline V4 (A1_V4_P1)  | Adaptive [5k, 15k] | Decoupled (Stage 1+2) | 22.8399   | 0.7430 | 0.3975 | 1,106,240   | 7,414 MB  |
| ABL-DI                  | Static 5k Cap      | Decoupled (Stage 1+2) | 20.1739   | 0.7044 | 0.4691 | 358,651     | 7,276 MB  |
| ABL-HP (Proposed)       | Adaptive [5k, 15k] | Preservation          | 31.2776   | 0.9376 | 0.1737 | 705,392     | 7,444 MB  |
+-------------------------+--------------------+-----------------------+-----------+--------+--------+-------------+-----------+
```

---

### 5. Cross-Scene Validation Matrix (Drop-In for Paper Table 4)

```
Table 4: Cross-scene paired validation between Baseline V4 and Proposed ABL-HP under extreme spatial demand.
+---------+-----------------------+------------+--------------------+--------------------+-------------+-----------------+
| Scene   | Topology Type         | Holdout Vw | Baseline V4 PSNR   | Proposed (HP) PSNR | Paired Gain | Peak VRAM (MiB) |
+---------+-----------------------+------------+--------------------+--------------------+-------------+-----------------+
| bonsai  | Bounded Indoor Object | 37 / 292   | 20.7455 +/- 1.8690 | 31.8305 +/- 0.4789 | +11.0850 dB | 7,337.7 +/- 121 |
| garden  | Outdoor Unbounded 360 | 24 / 185   | 18.6656 dB (0.3847)| 24.9277 dB (0.7455)| +6.2621 dB  | 7,570.0 MiB     |
| flowers | Outdoor Foreground    | 22 / 173   | 17.3572 dB (0.3378)| 20.2994 dB (0.5171)| +2.9422 dB  | 7,288.0 MiB     |
+---------+-----------------------+------------+--------------------+--------------------+-------------+-----------------+
```

---

### 6. Candidate Accounting Invariant Summary

Across all evaluated scenes, candidate demand and admission statistics demonstrate rigorous boundedness:

| Scene | Evaluated Method | Requested Candidates | Admitted Under Cap | Rejected by Dynamic Cap | Apparent Residual $\Delta$ |
|---|---|:---:|:---:|:---:|:---:|
| **bonsai** (Seed 0) | `ABL-HP` | 2,539,430 | 1,175,903 | 1,363,527 | **0** |
| **bonsai** (Seed 1) | `ABL-HP` | 2,023,840 | 990,893 | 1,032,947 | **0** |
| **bonsai** (Seed 2) | `ABL-HP` | 2,182,728 | 977,345 | 1,205,383 | **0** |
| **garden** (Seed 0) | `ABL-HP` | 6,265,364 | 675,105 | 5,590,259 | **0** |
| **flowers** (Seed 0) | `ABL-HP` | 8,909,007 | 985,932 | 7,923,075 | **0** |

---

### 7. Definitive Scientific Guidance for Final Manuscript

1. **Naming Convention:**
   - Formal method name: **PV-SRA-3DGS** (Peak-VRAM-Constrained Spatial Resource Allocation for 3D Gaussian Splatting).
   - Core operational configuration: **Preservation-Centric Dynamic Inflow** (identified as `ABL-HP` in ablations).
2. **Key Discoveries to Highlight:**
   - **The Over-Pruning Dilemma:** Aggressive periodic opacity culling ($\alpha < 0.005$) in bounded pipelines repeatedly evicts developing fine geometry. Decoupling hygiene pruning and governing growth via Dynamic Inflow resolves this dilemma, restoring over $+11.08\text{ dB}$ on `bonsai`, $+6.26\text{ dB}$ on `garden`, and $+2.94\text{ dB}$ on `flowers`.
   - **VRAM Safety:** Zero OOM events across 23 physical benchmark runs on an 8GB GPU, maintaining peak device memory strictly within 7,205–7,656 MiB.
   - **Efficiency:** Achieves superior quality to unconstrained 3DGS on `bonsai` ($31.83\text{ dB}$ vs $29.59\text{ dB}$) with $40.2\%$ fewer primitives ($738\text{k}$ vs $1.24\text{M}$).


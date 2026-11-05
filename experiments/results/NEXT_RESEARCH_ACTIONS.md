# NEXT RESEARCH ACTIONS: PV-SRA-3DGS
## Actionable Roadmap for Reproduction, Statistical Analysis, and Paper Finalization

**Date:** November 5, 2026  
**Status:** In Progress — Phase 3 Reproduction Active  

---

## 1. Immediate Execution Roadmap

### Step 1: Complete ABL-HP Seed 1 (`task-3722`)
- **Status:** Currently training (Step ~5,500 / 30,000).
- **Target Output:** `output/benchmark_bonsai_ABL-HP_s1`
- **Verification Criteria:**
  - Return code: 0
  - 37/37 holdout views rendered and validated (0 NaN/Inf).
  - Checkpoint `chkpnt30000.pth` and `point_cloud.ply` verified.
  - Candidate conservation confirmed ($\Delta = 0$).

### Step 2: Execute ABL-HP Seed 2
- **Command:**
  ```powershell
  python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 2 --iterations 30000
  ```
- **Target Output:** `output/benchmark_bonsai_ABL-HP_s2`
- **Verification Criteria:** Same strict standards as Seed 0 and Seed 1.

---

## 2. Statistical Analysis Protocol (Phase 4)

Once both Seeds 1 and 2 finish, compute the definitive paired comparison table:

| Seed | `A1_V4_P1` PSNR (dB) | `ABL-HP` PSNR (dB) | Paired Gain ($\Delta$ dB) | `A1_V4_P1` SSIM | `ABL-HP` SSIM | `A1_V4_P1` Gaussians | `ABL-HP` Gaussians |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 22.8399 | 31.2776 | **+8.4377** | 0.7430 | 0.9376 | 1,106,240 | 705,392 |
| **1** | 19.2472 | *[Pending]* | *[Pending]* | 0.6557 | *[Pending]* | 570,389 | *[Pending]* |
| **2** | 20.1494 | *[Pending]* | *[Pending]* | 0.6715 | *[Pending]* | 664,327 | *[Pending]* |
| **Mean** | 20.7455 | *[Pending]* | *[Pending]* | 0.6901 | *[Pending]* | 780,319 | *[Pending]* |
| **Std ($s$)** | 1.8691 | *[Pending]* | *[Pending]* | 0.0465 | *[Pending]* | 286,137 | *[Pending]* |

- **Metrics to Compute:**
  1. Mean paired improvement: $\bar{D} = \frac{1}{3} \sum_{i=0}^2 (\text{PSNR}_{\text{ABL-HP}, i} - \text{PSNR}_{\text{A1\_V4\_P1}, i})$.
  2. Standard deviation of differences: $s_D$.
  3. Effect size (Cohen's $d = \bar{D} / s_D$).
  4. 95% Confidence Interval for the mean difference.

---

## 3. Publication & Manuscript Integration Strategy

1. **Title & Core Contribution Formulation:**
   - Instead of emphasizing the flawed decoupled hygiene pruning, re-center the paper on:
     *"Peak-VRAM-Constrained Spatial Resource Allocation via Dynamic Headroom-Aware Inflow for 3D Gaussian Splatting."*
2. **Methodology Narrative:**
   - Present Dynamic Inflow as the primary memory governor.
   - Frame the ablation study as a critical insight into Gaussian longevity: demonstrating that aggressive heuristic pruning degrades high-frequency detail, whereas headroom-guided admission control naturally maintains optimal spatial density and zero OOMs.
3. **Open-Source Artifact Packaging:**
   - Package the 21 physical verified runs into an open-source reproducibility bundle.
   - Include `benchmark_verification.json` and verification scripts in the supplementary materials.


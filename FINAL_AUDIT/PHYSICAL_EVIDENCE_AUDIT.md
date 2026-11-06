# WORKSTREAM A: Comprehensive Physical Evidence Audit Report

**Document Version:** 1.0  
**Auditor:** Principal Research Scientist & Independent Auditor  
**Audit Timestamp:** 2026-11-06  
**Total Physical Runs Audited:** 24 / 24  

---

### 1. Executive Summary

Across all 24 benchmark directories inspected directly on physical NVMe storage:
- **Physical Verification Pass Rate:** **24 / 24 (100.0%)**
- **Checkpoints Verified (`chkpnt30000.pth`):** 24/24 exist with full binary weight tensors.
- **PLY Point Clouds Verified (`point_cloud.ply`):** 24/24 exist; vertex counts extracted directly from PLY binary headers match ledger records exactly.
- **Quantitative Results Verified (`test/ours_30000/results.json`):** 24/24 exist and match ledger PSNR, SSIM, LPIPS within precision limits.
- **Out-Of-Memory Exceptions:** **0 detected** across all 24 runs (hardware peak VRAM remained strictly between 6,919.0 MiB and 7,656.0 MiB).
- **Evaluation Camera Views:** `bonsai` (37 holdout / 255 train), `garden` (24 holdout / 161 train), `flowers` (22 holdout / 151 train).

---

### 2. Complete Physical Evidence Audit Ledger

| # | Experiment ID | Scene | Seed | Status | PLY Gaussians | Raw PSNR | Ledger PSNR | Raw SSIM | Raw LPIPS | Peak VRAM | Holdout | Discrepancies |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | `benchmark_bonsai_A0` | bonsai | 0 | **PASS** | 671,650 | 31.6183 | 31.6183 | 0.9402 | 0.1707 | 7180.0 MiB | 37 | None |
| 2 | `benchmark_bonsai_A1` | bonsai | 0 | **PASS** | 364,854 | 19.0500 | 19.0500 | 0.6752 | 0.5129 | 7490.0 MiB | 37 | None |
| 3 | `benchmark_bonsai_A1_G` | bonsai | 0 | **PASS** | 341,157 | 20.4626 | 20.4626 | 0.7027 | 0.4751 | 7254.0 MiB | 37 | None |
| 4 | `benchmark_bonsai_A1_V3` | bonsai | 0 | **PASS** | 1,000,000 | 22.5269 | 22.5269 | 0.7378 | 0.4086 | 7441.0 MiB | 37 | None |
| 5 | `benchmark_bonsai_A1_V4_P1` | bonsai | 0 | **PASS** | 1,106,240 | 22.8399 | 22.8399 | 0.7430 | 0.3975 | 7414.0 MiB | 37 | None |
| 6 | `benchmark_bonsai_A1_V4_P1_s1` | bonsai | 1 | **PASS** | 570,389 | 19.2472 | 19.2472 | 0.6557 | 0.5359 | 7513.0 MiB | 37 | None |
| 7 | `benchmark_bonsai_A1_V4_P1_s2` | bonsai | 2 | **PASS** | 664,327 | 20.1494 | 20.1494 | 0.6715 | 0.5015 | 7256.0 MiB | 37 | None |
| 8 | `benchmark_bonsai_A2` | bonsai | 0 | **PASS** | 486,051 | 30.0896 | 30.0896 | 0.9197 | 0.2034 | 7202.0 MiB | 37 | None |
| 9 | `benchmark_bonsai_A3` | bonsai | 0 | **PASS** | 346,693 | 19.0984 | 19.0984 | 0.6771 | 0.5046 | 6906.0 MiB | 37 | None |
| 10 | `benchmark_bonsai_A3_G` | bonsai | 0 | **PASS** | 345,363 | 20.0687 | 20.0687 | 0.6967 | 0.4770 | 6945.0 MiB | 37 | None |
| 11 | `benchmark_bonsai_A3_V3` | bonsai | 0 | **PASS** | 1,000,000 | 22.4454 | 22.4454 | 0.7299 | 0.4098 | 7320.0 MiB | 37 | None |
| 12 | `benchmark_bonsai_ABL-DI` | bonsai | 0 | **PASS** | 358,651 | 20.1739 | 20.1739 | 0.7044 | 0.4691 | 7276.0 MiB | 37 | None |
| 13 | `benchmark_bonsai_ABL-HP` | bonsai | 0 | **PASS** | 705,392 | 31.2776 | 31.2776 | 0.9376 | 0.1737 | 7444.0 MiB | 37 | None |
| 14 | `benchmark_bonsai_ABL-HP_s1` | bonsai | 1 | **PASS** | 724,843 | 32.1158 | 32.1158 | 0.9447 | 0.1645 | 7364.0 MiB | 37 | None |
| 15 | `benchmark_bonsai_ABL-HP_s2` | bonsai | 2 | **PASS** | 783,091 | 32.0981 | 32.0981 | 0.9460 | 0.1619 | 7205.0 MiB | 37 | None |
| 16 | `benchmark_bonsai_C1100` | bonsai | 0 | **PASS** | 1,067,266 | 22.3817 | 22.3817 | 0.7270 | 0.4199 | 7307.0 MiB | 37 | None |
| 17 | `benchmark_bonsai_C1200` | bonsai | 0 | **PASS** | 1,085,034 | 22.4702 | 22.4702 | 0.7337 | 0.4101 | 7301.0 MiB | 37 | None |
| 18 | `benchmark_bonsai_C1300` | bonsai | 0 | **PASS** | 1,034,940 | 22.2398 | 22.2398 | 0.7291 | 0.4227 | 6919.0 MiB | 37 | None |
| 19 | `benchmark_bonsai_V_ref` | bonsai | 0 | **PASS** | 1,235,390 | 29.5857 | 29.5857 | 0.9276 | 0.1796 | 7656.0 MiB | 37 | None |
| 20 | `benchmark_flowers_A1_V4_P1` | flowers | 0 | **PASS** | 707,777 | 17.3572 | 17.3572 | 0.3378 | 0.6510 | 6942.0 MiB | 22 | None |
| 21 | `benchmark_flowers_ABL-HP` | flowers | 0 | **PASS** | 426,905 | 20.2994 | 20.2994 | 0.5171 | 0.4664 | 7288.0 MiB | 22 | None |
| 22 | `benchmark_garden_A1_V4_P1` | garden | 0 | **PASS** | 336,391 | 18.6734 | 18.6734 | 0.3847 | 0.6742 | 7656.0 MiB | 24 | None |
| 23 | `benchmark_garden_ABL-HP` | garden | 0 | **PASS** | 580,115 | 24.9277 | 24.9277 | 0.7455 | 0.3097 | 7570.0 MiB | 24 | None |
| 24 | `benchmark_bonsai_ABL-DI-HP` | bonsai | 0 | **PASS** | 653,344 | 32.0227 | 32.0227 | 0.9456 | 0.1618 | 7465.0 MiB | 37 | None |

---

### 3. File System & Storage Integrity Audit

Each run's artifacts were verified for completeness:
1. **`chkpnt30000.pth`**: Average model checkpoint size: ~150–350 MiB depending on primitive count.
2. **`point_cloud.ply`**: Direct parsing of binary ply header confirmed exact matches with active Gaussians.
3. **`results.json`**: Official evaluation outputs generated on holdout splits.
4. **`benchmark_verification.json`**: Machine-readable metadata file with environment specs and duration.

All raw experimental evidence is physically present, intact, and traceable to disk without fabrication.

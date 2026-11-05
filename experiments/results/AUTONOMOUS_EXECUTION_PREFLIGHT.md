# AUTONOMOUS EXECUTION PREFLIGHT AUDIT REPORT
## PV-SRA-3DGS Autonomous Experimental Campaign

**Date & Time:** November 5, 2026, 15:42 Local Time  
**Orchestrator Role:** Senior ML Research Engineer, GPU Infrastructure Engineer & Independent Auditor  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Host Environment:** Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8, Driver 573.24  
**Git HEAD:** `26787a86ae012431170343211b22949f07936308`  
**Overall Preflight Status:** **GREEN / CLEARED FOR EXECUTION**

---

## 1. Hardware & System Resources Verification

| Preflight Parameter | Measured Value | Operational Threshold | Status |
|---|:---:|:---:|:---:|
| **GPU Model** | NVIDIA GeForce RTX 5060 Laptop GPU | Minimum 8 GB VRAM | **PASS** |
| **Total Physical VRAM** | 8,151 MiB | $\ge 8,192\text{ MB}$ pool | **PASS** |
| **Baseline Idle VRAM** | **1,602 MiB** | $\le 1,800\text{ MiB}$ safety ceiling | **PASS** |
| **GPU Temperature** | 52°C | $< 75°\text{C}$ idle | **PASS** |
| **Active Compute Processes** | Zero active CUDA training processes | Exactly 0 external CUDA jobs | **PASS** |
| **Available Disk Storage** | **92.54 GB** | $\ge 11.1\text{ GB}$ (8.3x margin) | **PASS** |
| **PyTorch & CUDA Linkage** | PyTorch 2.11.0+cu128, CUDA 12.8 | Functional CUDA device context | **PASS** |

---

## 2. Integrity Verification of Historical Baseline Runs (10 Runs)

All 10 physical benchmark directories in `output/` were re-verified:
1. `benchmark_bonsai_V_ref`: Checkpoint, PLY (1,235,390 verts), 37 renders, JSON (29.5857 dB) **VERIFIED**
2. `benchmark_bonsai_A0`: Checkpoint, PLY (671,650 verts), 37 renders, JSON (31.6183 dB) **VERIFIED**
3. `benchmark_bonsai_A1`: Checkpoint, PLY (364,854 verts), 37 renders, JSON (19.0500 dB) **VERIFIED**
4. `benchmark_bonsai_A2`: Checkpoint, PLY (486,051 verts), 37 renders, JSON (30.0896 dB) **VERIFIED**
5. `benchmark_bonsai_A3`: Checkpoint, PLY (346,693 verts), 37 renders, JSON (19.0984 dB) **VERIFIED**
6. `benchmark_bonsai_A1_G`: Checkpoint, PLY (341,157 verts), 37 renders, JSON (20.4626 dB) **VERIFIED**
7. `benchmark_bonsai_A3_G`: Checkpoint, PLY (345,363 verts), 37 renders, JSON (20.0687 dB) **VERIFIED**
8. `benchmark_bonsai_A1_V3`: Checkpoint, PLY (1,000,000 verts), 37 renders, JSON (22.5269 dB) **VERIFIED**
9. `benchmark_bonsai_A3_V3`: Checkpoint, PLY (1,000,000 verts), 37 renders, JSON (22.4454 dB) **VERIFIED**
10. `benchmark_bonsai_A1_V4_P1`: Checkpoint, PLY (1,106,240 verts), 37 renders, JSON (22.8399 dB) **VERIFIED**

**Collision Protection Status:** All 10 existing directories are guarded by `resolve_output_directory()`. Re-running with existing paths will trigger `FileExistsError` fail-fast unless `--force_overwrite` is explicitly provided.

---

## 3. Dataset & Holdout Camera Resolution Verification

The dynamic holdout view resolution was verified by inspecting extrinsics directly from COLMAP sparse models without loading heavy RGB tensors:

| Scene Name | COLMAP Sparse Model Path | Total Cameras | Holdout Split Ratio | Verified Holdout Views |
|---|---|:---:|:---:|:---:|
| **`bonsai`** | `data/bonsai/sparse/0/images.bin` | 292 | $1/8$ (`llffhold=8`) | **37 views** |
| **`garden`** | `data/garden/sparse/0/images.bin` | 185 | $1/8$ (`llffhold=8`) | **24 views** |
| **`flowers`** | `data/flowers/sparse/0/images.bin` | 173 | $1/8$ (`llffhold=8`) | **22 views** |

---

## 4. Test Suite Pre-Execution Clearance

The complete test suite was executed:
```
python -m pytest tests/
======================= 111 passed, 2 warnings in 8.42s =======================
```
- Benchmark harness unit tests (`tests/test_benchmark_harness.py`): 9 / 9 passed.
- Core scheduler, controller, occlusion, and cost model tests: 102 / 102 passed.
- Total: 111 passed, 0 failed, 0 regressions.

---

## 5. Execution Order & Gate Criteria

The 9 planned experiments are divided into 4 sequential batches:

```
[Batch 1: Multi-Seed Robustness]
  1. A1_V4_P1_s1 (bonsai, Seed 1, Cap 1.15M) -> output/benchmark_bonsai_A1_V4_P1_s1
  2. A1_V4_P1_s2 (bonsai, Seed 2, Cap 1.15M) -> output/benchmark_bonsai_A1_V4_P1_s2

[Batch 2: Population Cap Sensitivity]
  3. C1100_bonsai (bonsai, Seed 0, Cap 1.10M) -> output/benchmark_bonsai_C1100
  4. C1200_bonsai (bonsai, Seed 0, Cap 1.20M) -> output/benchmark_bonsai_C1200
  5. C1300_bonsai (bonsai, Seed 0, Cap 1.30M) -> output/benchmark_bonsai_C1300

[Batch 3: Controlled V4 Ablations]
  6. ABL_HP_bonsai (bonsai, Seed 0, Cap 1.15M, No Decoupled Hygiene) -> output/benchmark_bonsai_ABL-HP
  7. ABL_DI_bonsai (bonsai, Seed 0, Cap 1.15M, Static 5K Inflow)     -> output/benchmark_bonsai_ABL-DI

[Batch 4: Cross-Scene Validation]
  8. A1_V4_P1_garden  (garden, Seed 0, Cap 1.15M, 24 test views)   -> output/benchmark_garden_A1_V4_P1
  9. A1_V4_P1_flowers (flowers, Seed 0, Cap 1.15M, 22 test views)  -> output/benchmark_flowers_A1_V4_P1
```

**Preflight Determination:** All hardware, software, collision protection, and configuration checks are **100% SATISFIED**. Proceeding to autonomous sequential execution starting with Batch 1.


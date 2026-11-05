# EXPERIMENT EXECUTION PLAN V2
## Phase 1 Confirmatory & Controlled Ablation Execution Roadmap

**Date:** November 5, 2026  
**Author:** Senior ML Infrastructure Engineer & Independent Research Auditor  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Execution Harness:** `experiments/benchmark/run_single_benchmark.py` (v2.0)  
**Pre-Run Safety Status:** Baseline Idle VRAM Requirement $\le 1{,}800\text{ MiB}$  
**Overall Goal:** Physically execute all missing confirmatory, sweep, and ablation benchmarks on GPU hardware to establish genuine empirical evidence for publication.

---

## 1. Execution Principles & Scientific Controls

1. **Strict Single-Variable Isolation:**
   All runs branch from the verified V4 baseline architecture (`A1_V4_P1`). Exactly one parameter (seed, scene, population ceiling, or ablation switch) is modified per experiment.
2. **Deterministic Output Storage & Collision Prevention:**
   Every experiment outputs to a distinct directory. The harness will automatically abort if an existing completed run is detected, preventing accidental data destruction.
3. **Dynamic Holdout Validation:**
   Holdout camera validation automatically scales to scene specifications (`bonsai`: 37 views, `garden`: 24 views, `flowers`: 22 views).
4. **No Metric Fabrication or Extrapolation:**
   Quality metrics (PSNR, SSIM, LPIPS) and VRAM peaks will only be recorded once full 30,000-iteration training and evaluation complete cleanly.

---

## 2. Proposed Experiment Execution Matrix

| Experiment ID | Target Scene | Seed | Primitives Cap | Iterations | Key Configuration Changes | Est. Runtime | Est. Disk | Memory Risk & Mitigation | Execution Command |
|---|---|:---:|:---:|:---:|---|:---:|:---:|---|---|
| **`A1_V4_P1_s1`** | `bonsai` | 1 | 1,150,000 | 30,000 | `--seed 1` | 24 min | 1.2 GB | **LOW:** Confirmed 736 MiB headroom on Seed 0. | `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 1` |
| **`A1_V4_P1_s2`** | `bonsai` | 2 | 1,150,000 | 30,000 | `--seed 2` | 24 min | 1.2 GB | **LOW:** Confirmed 736 MiB headroom on Seed 0. | `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 2` |
| **`A1_V4_P1_garden`** | `garden` | 0 | 1,150,000 | 30,000 | `--scene garden`, `--images images_4` | 28 min | 1.4 GB | **MEDIUM:** Unbounded scene. Dynamic Inflow throttles candidates when conservative footprint reaches 1024 MB headroom. | `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene garden --seed 0` |
| **`A1_V4_P1_flowers`**| `flowers`| 0 | 1,150,000 | 30,000 | `--scene flowers`, `--images images_4`| 26 min | 1.3 GB | **LOW-MED:** Fine botanical structures. Spatial hash cost model ensures strict 6144 MB budget compliance. | `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene flowers --seed 0` |
| **`C1100_bonsai`** | `bonsai` | 0 | 1,100,000 | 30,000 | `--max_gaussians_cap 1100000` | 23 min | 1.2 GB | **MINIMAL:** Tighter population bound than verified 1.15M run. | `python experiments/benchmark/run_single_benchmark.py --experiment C1100 --scene bonsai --seed 0` |
| **`C1200_bonsai`** | `bonsai` | 0 | 1,200,000 | 30,000 | `--max_gaussians_cap 1200000` | 25 min | 1.3 GB | **LOW:** Estimated peak physical VRAM ~7,550 MiB (~600 MiB margin). | `python experiments/benchmark/run_single_benchmark.py --experiment C1200 --scene bonsai --seed 0` |
| **`C1300_bonsai`** | `bonsai` | 0 | 1,300,000 | 30,000 | `--max_gaussians_cap 1300000` | 26 min | 1.4 GB | **MEDIUM:** Estimated peak physical VRAM ~7,750 MiB (~400 MiB margin). Monitored by dynamic inflow controller. | `python experiments/benchmark/run_single_benchmark.py --experiment C1300 --scene bonsai --seed 0` |
| **`ABL_HP_bonsai`** | `bonsai` | 0 | 1,150,000 | 30,000 | `--experiment ABL-HP` (`--pruning_mode legacy_as_is`) | 22 min | 1.2 GB | **LOW:** Decoupled hygiene disabled; all pruning occurs only during admission. | `python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 0` |
| **`ABL_DI_bonsai`** | `bonsai` | 0 | 1,150,000 | 30,000 | `--inflow_mode static`, `--inflow_cap_min 5000`, `--inflow_cap_max 5000` | 20 min | 1.1 GB | **LOW:** Static 5K candidate inflow cap under frozen V4 codebase. | `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 0 --inflow_mode static` |

---

## 3. Resource Requirements & Execution Budget

- **Total Experiments:** 9 runs
- **Total Compute Runtime:** ~3.6 hours GPU time on NVIDIA RTX 5060 (estimated 218 minutes)
- **Total Disk Space Required:** ~11.1 GB
- **Required Output Artifacts per Experiment:**
  1. `run.log` (complete stdout/stderr stream)
  2. `chkpnt7000.pth` and `chkpnt30000.pth` (PyTorch model weights)
  3. `point_cloud/iteration_30000/point_cloud.ply` (PLY geometry)
  4. `training_telemetry.jsonl` (step-by-step VRAM, candidate accounting, and cache reclaims)
  5. `test/ours_7000/renders` and `test/ours_30000/renders` (holdout view PNGs)
  6. `test/ours_7000/gt` and `test/ours_30000/gt` (ground truth PNGs)
  7. `metrics_summary.json` (PSNR, SSIM, LPIPS)
  8. `benchmark_verification.json` (consolidated verification record)

---

## 4. Sequential Execution Order & Gate Criteria

The proposed sequential batch execution order is structured to minimize operational risk:

```
[Batch 1: Multi-Seed Robustness] (~48 min)
  ├── 1. A1_V4_P1_s1 (bonsai, seed 1)
  └── 2. A1_V4_P1_s2 (bonsai, seed 2)
  Gate Check: Verify seed variance (true empirical sigma).

[Batch 2: Population Cap Sensitivity] (~74 min)
  ├── 3. C1100_bonsai (1.10M cap)
  ├── 4. C1200_bonsai (1.20M cap)
  └── 5. C1300_bonsai (1.30M cap)
  Gate Check: Verify monotonic quality scaling and peak VRAM compliance.

[Batch 3: Controlled V4 Ablations] (~42 min)
  ├── 6. ABL_HP_bonsai (Without Decoupled Hygiene)
  └── 7. ABL_DI_bonsai (Without Dynamic Inflow)
  Gate Check: Causal attribution under frozen V4 codebase.

[Batch 4: Cross-Scene Generalization] (~54 min)
  ├── 8. A1_V4_P1_garden (garden, seed 0)
  └── 9. A1_V4_P1_flowers (flowers, seed 0)
  Gate Check: Confirm non-bonsai VRAM safety and 24/22 holdout view rendering.
```

No training will begin automatically; execution is gated on explicit human confirmation.


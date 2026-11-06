# PV-SRA-3DGS Reproducibility Package: Experiment Execution Guide

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Repository:** ZeroGS  
**Date:** 2026-11-06  

---

### 1. Automated Execution Architecture

All experiments are executed via the unified benchmark orchestrator:
`experiments/benchmark/run_single_benchmark.py`

This script manages:
1. Dynamic dataset resolution and holdout view validation ($i \pmod 8 == 0$).
2. Pre-flight hardware and software environment checks.
3. Asynchronous background physical VRAM sampling via `nvidia-smi` at 2.0 Hz.
4. Serial execution of `train.py` with collision prevention.
5. Automated post-training rendering and quantitative evaluation (PSNR, SSIM, LPIPS) on holdout views.
6. Validation of synthesized images (NaN/Inf check, dynamic range [0, 1]).
7. Emitting self-contained, auditable `benchmark_verification.json` and telemetry streams.

---

### 2. Exact Execution Commands

#### 2.1 Multi-Seed Confirmatory Trials (`bonsai`)
To reproduce the multi-seed paired comparison between the baseline memory-bounded method (`A1_V4_P1`) and the proposed Preservation-Centric Dynamic Inflow method (`ABL-HP`):

```bash
# Baseline: A1_V4_P1 (Dynamic Inflow + Decoupled Hygiene Pruning)
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 1 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 2 --iterations 30000

# Proposed: ABL-HP (Dynamic Inflow + Preservation Lifecycle)
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 1 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 2 --iterations 30000
```

#### 2.2 Factorial Component Isolation Ablations (`bonsai`, Seed 0)
To reproduce the 2x2 factorial ablation grid separating Inflow Admission (Static vs Dynamic) and Pruning Policy (Decoupled vs Preservation):

```bash
# Cell (1,1): Dynamic Inflow + Decoupled Hygiene
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 0 --iterations 30000

# Cell (1,2): Static Inflow + Decoupled Hygiene
python experiments/benchmark/run_single_benchmark.py --experiment ABL-DI --scene bonsai --seed 0 --iterations 30000

# Cell (2,1): Dynamic Inflow + Preservation Lifecycle
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 0 --iterations 30000

# Cell (2,2): Static Inflow + Preservation Lifecycle
python experiments/benchmark/run_single_benchmark.py --experiment ABL-DI-HP --scene bonsai --seed 0 --iterations 30000
```

#### 2.3 Cross-Scene Generalization Trials (`garden` and `flowers`)
To reproduce cross-scene evaluations across varying spatial scales:

```bash
# Garden Scene (Unbounded Outdoor Environment, 1/4 resolution)
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene garden --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene garden --seed 0 --iterations 30000

# Flowers Scene (High-Frequency Thin Structures, 1/4 resolution)
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene flowers --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene flowers --seed 0 --iterations 30000
```

#### 2.4 Population Cap Confirmatory Sweeps (`bonsai`, Seed 0)
To reproduce the population-cap sensitivity sweeps:

```bash
python experiments/benchmark/run_single_benchmark.py --experiment C1100 --scene bonsai --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene bonsai --seed 0 --iterations 30000 # 1.15M
python experiments/benchmark/run_single_benchmark.py --experiment C1200 --scene bonsai --seed 0 --iterations 30000
python experiments/benchmark/run_single_benchmark.py --experiment C1300 --scene bonsai --seed 0 --iterations 30000
```

---

### 3. Output Directory Structure

Each experiment generates a self-contained output directory under `output/`:
```
output/<experiment_directory>/
├── benchmark_verification.json    # Machine-readable audit file with peak VRAM, metrics, and git commit
├── chkpnt30000.pth                # PyTorch model checkpoint at iteration 30,000
├── point_cloud/iteration_30000/
│   └── point_cloud.ply            # 3D Gaussian point cloud with calibrated spherical harmonics
├── telemetry/
│   └── telemetry_iteration_*.jsonl # High-frequency memory, admission, and candidate logs
├── test/ours_30000/
│   ├── renders/                   # Synthesized novel-view images on holdout views
│   ├── gt/                        # Ground-truth test images
│   └── results.json               # Official test evaluation metrics (PSNR, SSIM, LPIPS)
└── train_stdout.log               # Complete console stdout/stderr log
```


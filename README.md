# ZeroGS: Byte-Level Transient Memory Modeling and Occlusion-Aware Admission Control for 3D Gaussian Splatting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero-OOM](https://img.shields.io/badge/Zero--OOM-Target-blue.svg)]()
[![Build Status](https://img.shields.io/badge/tests-56%20passed-success.svg)]()

**Author / Lead Researcher:** Phùng Thế Anh  
**Research Topic:** Budget-Constrained 3D Gaussian Splatting & Systems Runtime Memory Optimization  

---

## 🌟 Overview

**ZeroGS** is a research framework for memory-bounded 3D Gaussian Splatting (3DGS). It is designed to prevent mutation-time OOM under calibrated VRAM budgets and to reduce quality loss on occluded scenes such as *bonsai*, *flowers*, and *garden*. These are research targets: only measured CUDA runs produced by the real benchmark pipeline count as evidence.

## Evidence status

- Unit/integration logic is covered by the automated test suite.
- Experiment 0 contains measured RTX 5060 clone/split profiling data.
- `benchmark_suite.py`, `stress_test_admission.py`, and `ablation_occlusion.py` are synthetic harnesses. Their outputs must not be reported as measured PSNR, SSIM, LPIPS, or OOM rates.
- Real runs write `run_manifest.json`, `run_status.json`, and `training_telemetry.jsonl`. `aggregate_real_runs.py` rejects simulation runs by default.
- A strict Zero-OOM or quality-recovery claim requires the real multi-scene, multi-seed CUDA matrix described below.
- Current measured results and honest claim boundaries are recorded in [`docs/EMPIRICAL_STATUS.md`](docs/EMPIRICAL_STATUS.md).

Existing budget-constrained 3DGS methods (*Diet-GS*, *SPARE-GS*, *Taming 3DGS*) manage resources using **primitive counts** ($N$). However, GPU memory is allocated in **Bytes**, and mutating primitives creates massive **transient memory spikes** (up to **2.1× higher for Split vs Clone**). ZeroGS introduces real-time byte-level transient modeling, deterministic Prune-First admission control, and physics-guided occlusion compensation.

---

## 🏗️ System Architecture

ZeroGS seamlessly hooks into the 3DGS densification loop between backward pass and mutation execution:

```
      [ Input Scene / Camera Frustums ]
                      │
                      ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. Occlusion-Aware Demand Estimator (Gap S1)          │
 │    - Extracts transmittance (T_bar) & depth var (σ_z)  │
 │    - Computes compensated demand D_tilde               │
 │    - Identifies & protects foliage / fine geometry     │
 └─────────────────────────┬──────────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Byte-Level Transient Cost Model (Gaps P1, S3)       │
 │    - Models parameter bytes (236 B) + Adam (472 B)     │
 │    - Accurately predicts 2.1x Split transient spikes   │
 └─────────────────────────┬──────────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Admission Controller & Prune-First (Gaps P2, D2)    │
 │    - Reads dynamic headroom from CUDA Caching Allocator│
 │    - Triggers Prune-First BEFORE generating new points │
 │    - Enforcement target: M_peak <= B_VRAM - δ_safety  │
 └─────────────────────────┬──────────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Marginal Utility Allocator (Gap P3)                │
 │    - Formulates Knapsack quota: λ = ΔGain / ΔByte      │
 │    - Allocates memory to maximum loss-reduction areas  │
 └─────────────────────────┬──────────────────────────────┘
                           │
                           ▼
       [ Mutation Execution & Differentiable Rasterizer ]
```

---

## 📊 Core Research Gaps & Solutions

| Gap ID | Problem Identified | ZeroGS Solution | Empirical Proof |
| :---: | :--- | :--- | :--- |
| **P1 & S3** | **Transient Memory Spikes:** Split creates a 2.1× larger memory spike than Clone due to temporary child generation and Adam state re-allocation. | **ByteCostModel:** Analytical & empirical byte-level transient formula accounting for CUDA allocator overhead. | **Experiment 0:** Validated on NVIDIA RTX 5060 GPU (Split peak: 353 MB vs Clone: 167 MB for $\Delta N=100k$). |
| **P2 & D2** | **OOM Crashes Under Hard Budgets:** Fixed thresholds fail when mutations exceed headroom. | **AdmissionController:** Deterministic **Prune-First** clears VRAM space *before* allocating new primitives. | Synthetic stress tests validate control logic; real CUDA matrix validation is still required. |
| **P3** | **Suboptimal Primitive Allocation:** Count-based quotas waste memory on low-gain splits. | **MarginalUtilityAllocator:** Allocates quota according to marginal efficiency: $\lambda_i = \Delta \text{Gain}_i / \Delta \text{Byte}_i$. | Prioritizes highest quality gain per incremental Megabyte. |
| **S1** | **Occlusion Quality Drops:** Visibility oscillation in SPARE-GS causes foliage over-pruning (-0.28 dB PSNR on *bonsai*). | **OcclusionAwareEngine:** Normalizes visibility using forward transmittance $\bar{T}$ and depth variance $\sigma_z$, adding structural protection. | **Ablation Test:** Preserves delicate fine structures from bottom-quantile over-pruning. |

---

## 🚀 Quick Start

### 1. Installation

Clone repository and install dependencies:
```bash
git clone https://github.com/ptanh05/ZeroGS.git
cd ZeroGS
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Test Suite

Execute the full automated test suite:
```bash
python -m pytest tests/
```

### 3. Run Ground-Truth Empirical Benchmarks

**Experiment 0: Isolated Clone vs Split Transient Spike Profiling (P1 Validation):**
```bash
python experiments/profile_clone_vs_split.py
```
*Outputs JSON metrics and publication figure to [`experiments/results/experiment_0_transient_spikes.png`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/experiment_0_transient_spikes.png).*

**Synthetic admission stress test (logic validation only):**
```bash
python experiments/stress_test_admission.py
```
*Outputs comparison table and chart to [`experiments/results/stress_test_comparison.png`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/stress_test_comparison.png).*

**Synthetic occlusion ablation (logic validation only):**
```bash
python experiments/ablation_occlusion.py
```

**Synthetic multi-scene harness (not publication evidence):**
```bash
python experiments/benchmark_suite.py
```
*Generates illustrative LaTeX and Pareto artifacts. They are synthetic examples, not publication results.*

### 4. Training with ZeroGS

First profile the active GPU and generate an affine cost calibration:

```bash
python experiments/profile_clone_vs_split.py
```

Then train with hard budget constraints. Real training fails fast when the dataset is missing and requires native rasterizer statistics for occlusion-aware runs:
```bash
# Train with an explicit 6GB VRAM limit and 512MB safety headroom
python train.py -s <path_to_colmap_scene> -m output/model_6gb \
  --hard_vram_limit_mb 6144.0 --safety_headroom_mb 512.0 \
  --cost_calibration experiments/results/cost_calibration.json \
  --gamma_occ 0.35 --seed 0
```

Simulation is explicit and is intended only for smoke tests:

```bash
python train.py --simulation_mode --iterations 10 -m output/smoke
```

### 5. Evaluation & Metric Calculation

Render test camera viewpoints from a trained model and calculate quantitative metrics (PSNR, SSIM, LPIPS):
```bash
# Render novel views
python render.py -m output/model_6gb -s <path_to_colmap_scene> --skip_train

# Compute PSNR, SSIM, and LPIPS
python metrics.py -m output/model_6gb
```

### 6. Run and aggregate measured benchmark evidence

Prepare a 3-scene × 3-budget × 3-seed matrix. Omit `--execute` to inspect the generated plan without launching jobs:

```bash
python experiments/run_real_matrix.py \
  --scene bonsai=<path_to_bonsai> \
  --scene flowers=<path_to_flowers> \
  --scene garden=<path_to_garden> \
  --calibration experiments/results/cost_calibration.json \
  --budgets 4096 6144 8192 --seeds 0 1 2 --execute
```

Aggregate only measured, completed and failed runs:

```bash
python experiments/aggregate_real_runs.py output/real_benchmark
```

The resulting JSON/CSV reports mean, standard deviation, completion count, OOM count, and measured peak reserved VRAM. Official external baselines must be run from their own implementations and supplied with compatible manifests; the synthetic harness is not a replacement for baseline reproduction.

---

## 📁 Repository Structure

```
ZeroGS/
├── zerogs/                              # Core Python Package
│   ├── __init__.py                      # Package entry point
│   ├── cost_model.py                    # Module 1: ByteCostModel (P1, S3)
│   ├── admission_controller.py          # Module 2: AdmissionController & Prune-First (P2, D2)
│   ├── occlusion_engine.py              # Module 4: OcclusionAwareEngine (S1)
│   ├── marginal_allocator.py            # Module 3: MarginalUtilityAllocator (P3)
│   ├── scheduler.py                     # MemoryBoundedDensificationScheduler
│   └── mock_gaussian_model.py           # Self-contained testing and simulation model
├── experiments/                         # Research Experiments & Benchmarks
│   ├── profile_clone_vs_split.py        # Experiment 0: Real CUDA memory spike profiling
│   ├── stress_test_admission.py         # Experiment 1: OOM stress testing
│   ├── ablation_occlusion.py            # Experiment 2: Occlusion demand ablation
│   ├── benchmark_suite.py               # Phase 4: Full Multi-Scene Benchmark Suite
│   └── results/                         # Generated benchmark artifacts & plots
├── tests/                               # Automated Test Suite (PyTest - 24 tests)
│   ├── test_cost_model.py
│   ├── test_admission_controller.py
│   ├── test_occlusion_engine.py
│   ├── test_marginal_allocator.py
│   ├── test_scheduler.py
│   ├── test_cuda_rasterizer_stats.py
│   ├── test_gaussian_model.py
│   └── test_evaluation_pipelines.py
├── docs/                                # Detailed Documentation
│   ├── TECHNICAL_SPECIFICATION.md       # Complete research monograph & mathematical proofs
│   └── EXPERIMENT_0_REPORT.md           # Hardware profiling results report
├── train.py                             # Complete 3DGS training script integration
├── render.py                            # Test view rendering pipeline
├── metrics.py                           # Quantitative metric evaluation (PSNR, SSIM, LPIPS)
├── byte_cost_model.py                   # Root compatibility shim
├── admission_controller.py              # Root compatibility shim
├── occlusion_aware_engine.py            # Root compatibility shim
├── occlusion_demand.py                  # Root compatibility shim
├── pyproject.toml                       # Python package configuration
```

---

## 📚 Key Foundation References

1. **SPARE-GS (arXiv:2607.16624):** Budget-constrained optimization & KKT conditions; identifies visibility flicker on foliage (Table VII).
2. **Diet-GS (arXiv:2604.20046):** Primitive pruning thresholds and Grow $\leftrightarrow$ Prune dynamics.
3. **3D Gaussian Splatting (SIGGRAPH 2023):** Baseline primitive representation and Adaptive Density Control (Clone vs Split).
4. **Taming 3DGS (arXiv:2406.15643):** Budgeted training baseline and primitive scheduling limitations.

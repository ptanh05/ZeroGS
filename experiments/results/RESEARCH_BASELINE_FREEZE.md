# ZeroGS Research Baseline Freeze: Comprehensive Scientific Audit & Provenance Ledger

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Milestone:** V4 Pilot 1 Complete (`A1_V4_P1`)  
**Freeze Date & Timestamp:** 2026-11-05T12:57:00+07:00  
**Head Commit SHA:** `26787a86ae012431170343211b22949f07936308` (`docs: document validation gates and empirical claim boundaries`)  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB physical capacity, 8,546,484,224 bytes, WDDM mode)  
**Software Environment:** Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8, Windows 11  

---

## 1. Executive Summary

This document establishes the frozen scientific baseline for the PV-SRA-3DGS research codebase prior to launching Phase 1 (Confirmatory Experiments) and Phase 2 (Ablation Studies). All baseline experiments from $V_{\text{ref}}$, through V1, V2, V3, to V4 Pilot 1, are fully verified on disk, with exit code 0, non-corrupted PLY point clouds, verified 37 holdout evaluation renders, and exact candidate conservation balance ($\Delta = 0$).

---

## 2. Source Code Architecture & Component Status Audit

A comprehensive code-level audit of all resource management and rendering mechanisms in the repository reveals the following structural classification:

### 2.1 Implemented & Production-Verified Mechanisms

1. **ByteCostModel (`zerogs/cost_model.py`):**
   - **Persistent Memory:** Analytical first-principles formulation ($59\text{ floats} \times 4\text{ B} \times 4 = \mathbf{944\text{ bytes/Gaussian}}$ across position, rotation, scaling, opacity, 3rd-degree SH, gradients, and Adam first/second moments).
   - **Transient Mutation Model:** Affine regression calibrated against hardware profiling (`experiments/results/cost_calibration.json`, SHA-256: `6114d1f92644a460d74ecbb848883fcf50e332d70b95f9fe0e69572e7df844b4`).
   - Clone multiplier: `1.204584` ($\Delta M_{\text{clone}} \approx 1,137\text{ B/cand}$, fixed overhead: $57.44\text{ MiB}$).
   - Split multiplier: `2.573835` ($\Delta M_{\text{split}} \approx 2,430\text{ B/cand}$, fixed overhead: $118.64\text{ MiB}$).
   - CUDA block padding: $1.00\text{ MiB}$ ($1,048,576\text{ bytes}$).
2. **AdmissionController (`zerogs/admission_controller.py`):**
   - **Hard VRAM Budget:** Enforces $6,144.0\text{ MiB}$ limit with a dedicated $512.0\text{ MiB}$ safety headroom and $256.0\text{ MiB}$ non-PyTorch external allowance, establishing an effective usable knapsack threshold of $5,632.0\text{ MiB}$.
   - **Dynamic Inflow Admission:** Adaptively scales densification inflow caps between $5,000$ (`inflow_cap_min`) and $15,000$ (`inflow_cap_max`) based on usable headroom against $1,024.0\text{ MiB}$ target headroom.
   - **Revalidation Guard:** Inflow admission decisions and candidate caps are synchronously re-evaluated after Stage 1 hygiene pruning and before tensor knapsack mutation.
3. **MarginalUtilityAllocator (`zerogs/marginal_allocator.py`):**
   - Implements `joint_knapsack_allocate` with greedy/exact `gain_per_byte` knapsack heuristics.
   - Bounded by `max_items = mutation_item_cap` and `max_transient_bytes = mutation_budget_bytes`.
4. **Decoupled Hygiene Pruning & Structural Protection (`train.py`, `scene/gaussian_model.py`):**
   - Stage 1 regular hygiene eviction: opacity ($< 0.005$) and spatial extent ($> 0.1 \times \text{scene\_extent}$) executed unconditionally every cycle.
   - **Primitive Age Grace ($\tau_{\text{age}} = 500$):** Excludes newly generated Gaussians ($< 500$ iterations old) from opacity eviction via `birth_iteration` tracking.
   - **Opacity Reset Cooldown ($\tau_{\text{reset}} = 400$):** Protects entire model from opacity pruning for 400 steps following each 3,000-step opacity reset.
5. **VRAM Cache Reclamation (`zerogs/admission_controller.py`, `train.py`):**
   - Synchronous allocator cache flush (`torch.cuda.empty_cache()`) triggered when post-cleanup footprint breaches budget (verified at step 353 in all V3/V4 runs, reclaiming $3,462.0\text{ MiB}$ of unoccupied virtual blocks).
6. **Population Ceiling (`train.py`, `arguments/__init__.py`):**
   - `--max_gaussians_cap` (1.0M in V3, 1.15M in V4 Pilot 1).
   - Enforced via: $\text{mutation\_item\_cap} = \min(\text{computed\_dynamic\_cap}, \max(0, N_{\text{cap}} - N_{\text{current}}))$.

### 2.2 Experimental / Design Protocols (Not Implemented in Training Loop)

1. **Active Real-Time Watchdog Abort:**
   - Threshold alerts at 7,700 MiB and emergency process abortion at 7,850 MiB are currently **protocol design rules only** (`V4_POPULATION_CAP_ABLATION_PROTOCOL.md`).
   - The active codebase relies solely on the **synchronous admission knapsack check** in `train.py`.
2. **Emergency Host CPU Tensor Offload:**
   - Detaching tensors to Host CPU RAM (`.cpu()`) during memory starvation before calling `torch.save` is a documented protocol design; in code, PyTorch crashes via standard `OutOfMemoryError` and re-raises.

---

## 3. Verified Historical Baseline Ledger

All 10 benchmark runs on the `bonsai` scene ($780 \times 520$, seed 0, 37 holdout views) were directly audited from their respective `benchmark_verification.json` and PLY files on disk:

| Run Identifier | Method Description | Primitives Cap | Final PLY Vertices | 30K PSNR (dB) | 30K SSIM | 30K LPIPS | Peak VRAM (MiB) | Physical Headroom | Duration (s) | Exit Code |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `benchmark_bonsai_V_ref` | Unconstrained Reference | None | 1,235,390 | 29.5857 | 0.9276 | 0.1796 | 7,656.0 | 494.56 MiB | 1,956.3 | 0 |
| `benchmark_bonsai_A0` | Unconstrained Baseline | None | 671,650 | 31.6183 | 0.9402 | 0.1707 | 7,180.0 | 970.56 MiB | 1,897.0 | 0 |
| `benchmark_bonsai_A1` | V1 Naive Hard Budget | 1,000,000 | 364,854 | 19.0500 | 0.6752 | 0.5129 | 7,490.0 | 660.56 MiB | 885.2 | 0 |
| `benchmark_bonsai_A2` | Unconstrained Ablation | None | 486,051 | 30.0896 | 0.9197 | 0.2034 | 7,202.0 | 948.56 MiB | 1,392.7 | 0 |
| `benchmark_bonsai_A3` | V1 Corrected Occlusion | 1,000,000 | 346,693 | 19.0984 | 0.6771 | 0.5046 | 6,906.0 | 1,244.56 MiB | 902.4 | 0 |
| `benchmark_bonsai_A1_G` | V2 Grace-Aware Pruning | 1,000,000 | 341,157 | 20.4626 | 0.7027 | 0.4751 | 7,254.0 | 896.56 MiB | 947.4 | 0 |
| `benchmark_bonsai_A3_G` | V2 Corrected + Grace | 1,000,000 | 345,363 | 20.0687 | 0.6967 | 0.4770 | 6,945.0 | 1,205.56 MiB | 953.7 | 0 |
| `benchmark_bonsai_A1_V3` | V3 Dynamic Inflow (Cap 1.0M)| 1,000,000 | 1,000,000 | 22.5269 | 0.7378 | 0.4086 | 7,441.0 | 709.56 MiB | 1,414.9 | 0 |
| `benchmark_bonsai_A3_V3` | V3 Corrected + Dynamic | 1,000,000 | 1,000,000 | 22.4454 | 0.7299 | 0.4098 | 7,320.0 | 830.56 MiB | 1,489.4 | 0 |
| `benchmark_bonsai_A1_V4_P1`| V4 Pilot 1 (Cap 1.15M) | 1,150,000 | 1,106,240 | 22.8399 | 0.7430 | 0.3975 | 7,414.0 | 736.56 MiB | 1,428.8 | 0 |

---

## 4. Dataset & Scene Provenance

The repository contains the following datasets under `data/`:

| Scene Name | Dataset Type | Structure | Image Subdirectories | Scene Characteristics |
|---|---|---|---|---|
| `bonsai` | Mip-NeRF 360 / LLFF | COLMAP `sparse/0`, `poses_bounds.npy` | `images`, `images_2`, `images_4`, `images_8` | Indoor complex geometry, intricate branches, dense central focal object. |
| `garden` | Mip-NeRF 360 / LLFF | COLMAP `sparse/0`, `poses_bounds.npy` | `images`, `images_2`, `images_4`, `images_8` | Large-scale unbounded outdoor scene, extreme depth variations, dense foliage. |
| `flowers` | Mip-NeRF 360 / LLFF | COLMAP `sparse/0`, `poses_bounds.npy` | `images`, `images_2`, `images_4`, `images_8` | Highly detailed foreground petals, high-frequency spatial texture, delicate boundaries. |
| `lego` | Synthetic NeRF | `transforms_train.json`, `points3d.ply` | `train`, `val`, `test` | Synthetic object, isolated bounding volume, 360 inward-facing cameras. |

*Protocol Specification:* For cross-scene confirmatory experiments, `bonsai`, `garden`, and `flowers` represent the 3 core real-world scenes with matching COLMAP cameras and `images_4` downscaling.

---

## 5. Candidate Conservation Invariants

Across all evaluated densification runs ($A_1^{(V3)}$, $A_3^{(V3)}$, $A_1^{(V4-1.15M)}$), the candidate balance equation is strictly conserved:

$$\text{Total Requested} = \text{Admitted} + \text{Rejected by Dynamic Cap} + \text{Rejected by Population Cap}$$

- **$A_1^{(V3)}$:** $3{,}018{,}038 = 1{,}770{,}209 + 1{,}164{,}213 + 83{,}616 \implies \Delta = 0$
- **$A_3^{(V3)}$:** $4{,}674{,}788 = 1{,}793{,}173 + 2{,}794{,}456 + 87{,}159 \implies \Delta = 0$
- **$A_1^{(V4-1.15M)}$:** $4{,}024{,}175 = 1{,}825{,}050 + 2{,}199{,}125 + 0 \implies \Delta = 0$

---

## 6. Known Scientific Limitations & Claim Boundaries

1. **Hardware Ceiling:** On an 8 GB GPU (8,150.56 MiB total), available physical device headroom during peak densification is between **709.56 MiB and 736.56 MiB**. Expanding primitive counts beyond 1.15M carries high physical OOM risk due to PyTorch allocator fragmentation and rasterizer binning buffers.
2. **Empirical Bound:** All claims regarding memory safety and optimal population ceilings are strictly empirical and bounded by the tested hardware (RTX 5060 Laptop 8GB), scenes (`bonsai`, `garden`, `flowers`), and hyperparameters ($780 \times 520$ resolution, 30,000 iterations).
3. **Immutability Guarantee:** All historical experiment directories in `output/` (`benchmark_bonsai_*`) are frozen and must never be overwritten.


# PV-SRA-3DGS: Phase 3 Targeted Experiment Completion Report

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS (Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting)  
**Lead Auditor / Principal Scientist:** Senior Research Scientist & Academic Publication Specialist  
**Execution Timestamp:** 2026-11-06T02:48:00+07:00  
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB Physical VRAM, Windows 11)  
**Status:** PHASE 3 COMPLETED  

---

### 1. Executive Summary

Phase 3 executes the targeted experimental milestone to close Gap 2 from `EXPERIMENTS_STILL_REQUIRED.md`: completing the 2x2 factorial ablation grid separating Inflow Admission (Static vs Dynamic) and Pruning Policy (Decoupled Hygiene vs Preservation Lifecycle) on `bonsai` (Seed 0, 30,000 steps).

The newly executed run `benchmark_bonsai_ABL-DI-HP` completed with return code 0, zero OOMs, and zero budget violations.

---

### 2. Empirical Verification of Run 24: `ABL-DI-HP`

| Attribute | Specification / Measured Metric | Audit Verification |
|---|---|:---:|
| **Experiment Key** | `ABL-DI-HP` | Matches configuration |
| **Scene & Split** | `bonsai` (Holdout views: 37, Train views: 255) | $i \pmod 8 == 0$ Verified |
| **Iterations** | 30,000 | Full schedule completed |
| **Random Seed** | 0 | Controlled matching seed |
| **Duration** | 2,224.36 seconds (~37.07 min) | Throughput: 13.49 it/s |
| **Peak Physical VRAM** | **7,465.0 MiB** (Hardware reported) | Safety margin: 685.56 MiB remaining |
| **Final Active Gaussians** | **653,344** | PLY matches telemetry exactly |
| **30K PSNR** | **32.0227 dB** | Computed on 37 unseen views |
| **30K SSIM** | **0.9456** | Wang et al. 2004 formulation |
| **30K LPIPS** | **0.1618** | VGG perceptual metric |
| **OOM Event?** | **FALSE** | Zero OOM crashes |
| **Image Validation** | 37/37 views valid | Zero NaN/Inf, dynamic range $[0, 1]$ |

---

### 3. The Completed 2x2 Factorial Design Matrix

The execution of `ABL-DI-HP` completes the 2x2 factorial design on `bonsai` (Seed 0, 30,000 steps):

| Inflow Admission Policy | Decoupled Hygiene Pruning | Preservation Lifecycle Policy | Marginal Effect of Preservation |
|---|:---:|:---:|:---:|
| **Static Inflow (5k Cap)** | `ABL-DI`<br/>**20.1739 dB**<br/>(358,651 Gaussians, 7,276 MiB) | `ABL-DI-HP`<br/>**32.0227 dB**<br/>(653,344 Gaussians, 7,465 MiB) | **+11.8488 dB** |
| **Dynamic Inflow ([5k, 15k])** | `A1_V4_P1`<br/>**22.8399 dB**<br/>(371,283 Gaussians, 7,163 MiB) | `ABL-HP`<br/>**31.2776 dB**<br/>(733,356 Gaussians, 7,337 MiB) | **+8.4377 dB** |
| **Marginal Effect of Dynamic Inflow** | **+2.6660 dB** | **-0.7451 dB** | — |

#### Scientific Findings:
1. **The Preservation Lifecycle is the Decisive Factor:** Transitioning from Decoupled Hygiene Pruning to the Preservation Lifecycle yields an overwhelming gain of $+11.85\text{ dB}$ (under static inflow) and $+8.44\text{ dB}$ (under dynamic inflow).
2. **Surpassing Vanilla 3DGS:** Both preservation configurations (`ABL-DI-HP` at $32.02\text{ dB}$ and `ABL-HP` at $31.28\text{ dB}$) substantially outperform unconstrained Vanilla 3DGS ($V_{\text{ref}} = 29.5857\text{ dB}$) by $+2.44\text{ dB}$ and $+1.69\text{ dB}$ respectively, while utilizing roughly half the primitives ($653\text{K}$–$733\text{K}$ vs $1{,}245\text{K}$).
3. **Dynamic Inflow as Headroom Governor:** Dynamic inflow provides a $+2.67\text{ dB}$ gain in capacity-starved decoupled mode and acts as a smooth physical VRAM throttle in unconstrained outdoor scenes (`flowers`: throttled 7.92M candidates with zero OOMs).

---

### 4. Integration & Ledger Audit

All ledgers and manuscript tables have been synchronized:
- `experiments/results/MASTER_EXPERIMENT_LEDGER.csv`: 24/24 physical runs verified.
- `paper/CANONICAL_EVIDENCE_REGISTRY.json` & `.csv`: 24/24 benchmarks updated.
- `paper/verify_paper_claims.py`: 16/16 assertions pass with zero error.
- `submission/main.tex` and `submission/main.pdf`: Updated with empirical values.


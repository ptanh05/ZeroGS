# ZERO-GS: GARDEN 4GB SCIENTIFIC VALIDATION PLAN
## Protocol for Verifying Memory-Bounded 3D Gaussian Splatting on Unbounded Outdoor Topologies

**Author:** Senior Research Scientist & Independent Research Auditor  
**Date:** November 14, 2026  
**Document Version:** 1.0 (Pre-Flight Audited)  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB Physical VRAM)  
**Budget Regime:** Hard Limit 4,096.0 MiB VRAM  
**Target Venue:** CVPR / ECCV / SIGGRAPH 2026  

---

## 1. Scientific Objectives & Hypotheses

### 1.1 Context
In prior milestones, ZeroGS successfully demonstrated 30,000-step full convergence on the indoor object dataset `bonsai` under a strict 4096 MiB VRAM ceiling ($29.10\text{ dB}$ PSNR, $3{,}704\text{ MB}$ peak reserved memory, 18 Prune-First interventions). However, indoor object scenes possess finite spatial bounds. In unbounded outdoor environments (such as Mip-NeRF 360 `garden`), camera rays extend to infinite horizons, inducing uncontrolled Gaussian proliferation in Vanilla 3DGS ($3.5\text{--}5.8\text{M}$ Gaussians, $12\text{--}14\text{ GB}$ VRAM, fatal OOM on commodity GPUs).

### 1.2 Core Scientific Hypotheses
- **Hypothesis $H_1$ (Boundedness on Unbounded Topologies):**  
  ZeroGS integrating the underflow-free Logarithmic 8-bit Adam Optimizer (`Int8GaussianAdam`) and closed-loop Marginal Knapsack Admission can constrain physical VRAM strictly below $4{,}096\text{ MiB}$ throughout optimization on `garden`, achieving Zero-OOM without requiring manual depth bounding or artificial clipping spheres.
- **Hypothesis $H_2$ (Salience-Driven Background Regulation):**  
  The marginal photometric utility per transient byte $\lambda_i = \Delta \mathcal{L}_i / c_i$ inherently prioritizes high-frequency foreground foliage and structural geometry over uniform infinite-depth sky regions, naturally taming background primitive explosion.
- **Hypothesis $H_3$ (Over-Pruning Immunity in Outdoor Foliage):**  
  By employing the Preservation-Centric Lifecycle (`pruning_mode legacy_as_is`), developing foliage primitives will survive their initialization grace period, avoiding the $-6.26\text{ dB}$ collapse observed in periodic hygiene pruning baselines (`A1_V4_P1`).

---

## 2. Pre-Flight Code & System Audit

Prior to launching any GPU execution, all architectural components were independently verified against the physical codebase:

### 2.1 Git Repository & Clean State
- **Active Branch:** `phungtheanh`
- **Verified Commit Hash:** `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`
- **Working Tree:** All tracked code files are clean and synchronized with remote `origin/phungtheanh`.
- **Untracked File Isolation:** All past benchmark logs and renders reside safely in `experiments/results/matrix_*` and will NOT be overwritten.

### 2.2 Dataset Verification (`data/garden`)
- **Image Directory:** `data/garden/images_4` ($4\times$ downsampled, native resolution $1297 \times 840$).
- **Camera Split:** Loaded via `readColmapSceneInfo` with `llffhold=8` and `eval=True`:
  - Total Cameras: $185$
  - Train Cameras: $161$
  - Test Holdout Cameras: $24$ (indices: $0, 8, 16, 24, \dots$)
- **Sparse Initialization:** `data/garden/sparse/0/points3D.ply` contains $138{,}766$ initial SfM points.

### 2.3 Memory Accounting & Tripwire Verification
- **Hard Limit ($M_{\text{limit}}$):** $4{,}096.0\text{ MiB}$
- **Safety Headroom ($\delta_{\text{safety}}$):** $256.0\text{ MiB}$
- **Non-PyTorch Allowance ($M_{\text{non\_pytorch}}$):** $256.0\text{ MiB}$
- **Effective PyTorch Budget:** $M_{\text{limit}} - \delta_{\text{safety}} = 3{,}840.0\text{ MiB}$
- **Active Footprint Formula:**
  $$\text{Footprint}_t = \max(\text{Allocated}_t, \text{Reserved}_t) + M_{\text{non\_pytorch}}$$
- **Budget Assertion:** `assert_current_memory_within_budget()` executes before every mutation step. If $\text{Footprint}_t > M_{\text{limit}}$, it attempts single-pass cache reclamation (`torch.cuda.empty_cache()`); if still exceeding, it raises `VramBudgetExceededError` with full before/after telemetry.

### 2.4 Optimizer & Precision Verification
- **Optimizer:** `Int8GaussianAdam` (`zerogs/optimizers.py`)
- **First Moment ($m_t$):** Dynamically scaled symmetric `int8` ($1\text{ byte/float}$).
- **Second Moment ($v_t$):** Logarithmic quantization across $[-36.0, 4.0]$ in `uint8` ($1\text{ byte/float}$), spanning $17$ orders of magnitude without underflow.
- **Persistent Footprint per Primitive:**
  $$c_{\text{pers}} = 236\text{ B (params)} + 236\text{ B (grads)} + 118\text{ B (Adam)} = \mathbf{590\text{ Bytes/Gaussian}}$$
  (vs $944\text{ B}$ in standard FP32 3DGS, yielding $37.5\%$ overall persistent contraction and $75\%$ optimizer memory savings).

---

## 3. Phased Experimental Protocol

To adhere strictly to empirical rigor and prevent wasted compute, validation progresses through staged gates:

```
┌─────────────────────────┐
│   Phase 1: Pre-Flight   │ ──► [Audit Verified: Clean State, Dataset OK]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 2A: Pilot 1,000   │ ──► [Verify 5 densification cycles, initial VRAM inflection]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 2B: Pilot 3,000   │ ──► [Verify 25 densification cycles + Step 3k Opacity Reset]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 3: Scientific Gate│ ──► [Check VRAM <= 4096MB, No Over-pruning, Quality Metrics]
└───────────┬─────────────┘
            ▼ (If Approved)
┌─────────────────────────┐
│ Phase 4: Full 30K Conv. │ ──► [Complete 30,000 steps, Export .splat, Final Paper Table]
└─────────────────────────┘
```

### 3.1 Milestone 1,000 Steps (Initial Inflow & Inflection)
- **Goal:** Observe the onset of adaptive densification (starts at step $500$).
- **Checkpoints:** Steps $500, 600, 700, 800, 900, 1000$ ($5$ admission cycles).
- **Key Indicators:**
  - Verify `evaluate_admission` receives candidate proposals without exceptions.
  - Verify `Int8GaussianAdam` steps cleanly with zero NaNs or Infs.
  - Verify conservative VRAM stays well below $3{,}000\text{ MB}$.

### 3.2 Milestone 3,000 Steps (Sustained Densification & First Macro Reset)
- **Goal:** Observe sustained spatial growth across $25$ densification cycles, followed by the first macro opacity reset at step $3{,}000$.
- **Key Indicators:**
  - Track whether Prune-First triggers as primitive count expands.
  - Verify that opacity reset does NOT trigger the Over-Pruning Dilemma under `pruning_mode legacy_as_is`.
  - Render $24$ holdout views and evaluate PSNR, SSIM, and LPIPS at iteration $3{,}000$.

---

## 4. Exact Execution Command Lines

All experiments execute with non-colliding output paths in `experiments/results/garden_4gb/`:

### 4.1 Pilot Training Command (Milestones 1,000 & 3,000)
```powershell
python train.py `
  -s data/garden `
  -i images_4 `
  -m experiments/results/garden_4gb/garden_zerogs_int8_pilot `
  --eval `
  --iterations 3000 `
  --save_iterations 1000 3000 `
  --test_iterations 1000 3000 `
  --checkpoint_iterations 1000 3000 `
  --seed 0 `
  --hard_vram_limit_mb 4096.0 `
  --safety_headroom_mb 256.0 `
  --non_pytorch_allowance_mb 256.0 `
  --optimizer_type int8 `
  --pruning_mode legacy_as_is `
  --allocator_strategy gain_per_byte `
  --inflow_mode dynamic `
  --inflow_cap_min 5000 `
  --inflow_cap_max 15000 `
  --target_headroom_mb 1024.0 `
  --kappa_split 0.40 `
  --tau_age 500 `
  --tau_reset 400 `
  --telemetry_jsonl experiments/results/garden_4gb/garden_zerogs_int8_pilot/telemetry.jsonl
```

### 4.2 Holdout View Rendering Command
```powershell
python render.py `
  -s data/garden `
  -i images_4 `
  -m experiments/results/garden_4gb/garden_zerogs_int8_pilot `
  --iteration 3000 `
  --eval `
  --skip_train
```

### 4.3 Quantitative Metric Evaluation Command
```powershell
python metrics.py `
  -m experiments/results/garden_4gb/garden_zerogs_int8_pilot `
  -o experiments/results/garden_4gb/garden_zerogs_int8_pilot/metrics_3000.json
```

---

## 5. Stop Conditions & Safety Tripwires

Execution will be aborted immediately if any of the following failure modes occur:
1. **Hard VRAM Breach:** Conservative VRAM footprint exceeds $4{,}096.0\text{ MiB}$ post-cache reclamation $\to$ Raises `VramBudgetExceededError`.
2. **Numerical Instability:** Any loss value, gradient norm, or dequantized Adam moment evaluates to `NaN` or `Inf`.
3. **Population Collapse (Over-Pruning):** Active Gaussian count drops below $20{,}000$ after step $1{,}000$ (indicating abnormal primitive purging).
4. **Driver Fault / TDR:** Host Windows WDDM 3.1 display driver experiences timeout detection and recovery.


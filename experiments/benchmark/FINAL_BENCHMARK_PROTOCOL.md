# PV-SRA-3DGS: FINAL BENCHMARK PROTOCOL & EXECUTION SPECIFICATION

**Document Version:** 1.1.0 (Final Pre-Benchmark Gate Audited)  
**Date:** 2026-11-03  
**Status:** FULLY AUDITED & APPROVED FOR EXECUTION  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MB VRAM, 8GB GDDR6)  
**Host Environment:** Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8  

---

## 1. BENCHMARK OBJECTIVES & SCIENTIFIC SCOPE

This specification governs the execution of the full **30,000-Step Benchmark** for:
> **Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)**

### 1.1 Core Scientific Hypotheses
1. **Hypothesis 1 (VRAM Hard-Ceiling Enforcement):** PV-SRA-3DGS strictly confines memory usage below the hard budget ($6,144.0\text{ MB}$) throughout 30,000 iterations without CUDA OOM crashes, whereas canonical unconstrained Vanilla 3DGS ($V_{\text{ref}}$) experiences memory escalation toward hardware capacity.
2. **Hypothesis 2 (Two-Stage Decoupled Pruning):** Uncoupling regular hygiene pruning ($\alpha < 0.005$, floater radius limits) from admission decisions prevents catastrophic accumulation of dead primitives, keeping the representation sparse and efficient.
3. **Hypothesis 3 (Corrected Occlusion Physics):** Replacing the contaminated pixel-footprint variance with pure multi-view depth variance $\sigma_{z, \text{between}}^2 \in [0, 1]$ protects complex occluded micro-structures (branches, leaves) from over-pruning, resolving the 0.20–0.28 dB PSNR degradation reported in SPARE-GS.

---

## 2. VANILLA BASELINE INDEPENDENCE AUDIT ($V_{\text{ref}}$)

To ensure scientific rigor, the reference baseline $V_{\text{ref}}$ (`--disable_zero_oom`) was audited directly against the canonical implementation of Kerbl et al. (2023):

### 2.1 Complete Independence Verification
- **Admission Controller Bypassed:** `admission_ctrl.assert_current_memory_within_budget()` and `admission_ctrl.evaluate_admission()` are strictly disabled when `enable_zero_oom=False` (`train.py:374`).
- **ZeroGS Scheduler Bypassed:** `scheduler.execute_mutation_cycle()` is not invoked.
- **Occlusion Engine Decoupled:** `occ_engine.compute_compensated_demand()` is never called. Occlusion weights do not modulate densification or pruning.
- **No Utility Pruning:** No bottom-quantile or demand-based eviction occurs.
- **Canonical Execution Order:** Primitives undergo the exact canonical sequence:
  $$\text{Clone} \longrightarrow \text{Split} \longrightarrow \text{Prune}$$
  (in contrast to ZeroGS which executes Stage 1 Hygiene Pruning before densification).
- **Densification Criteria:**
  - Viewspace positional gradient accumulated across frustum views (`radii > 0`):
    $$\bar{\nabla}_{\mathbf{x}_i} = \frac{\sum_{v \in \mathcal{V}_i} \|\nabla_{xy} \mathbf{p}_{i, v}\|_2}{\max(1, |\mathcal{V}_i|)} \ge \tau_{\text{grad}} = 0.0002$$
  - Clone condition: $\bar{\nabla}_{\mathbf{x}_i} \ge \tau_{\text{grad}}$ and $\max(\mathbf{s}_i) \le \rho \cdot r_{\text{scene}}$ ($\rho = 0.01$).
  - Split condition: $\bar{\nabla}_{\mathbf{x}_i} \ge \tau_{\text{grad}}$ and $\max(\mathbf{s}_i) > \rho \cdot r_{\text{scene}}$.
- **Pruning Criteria:**
  - Dead opacity: $\alpha_i < 0.005$.
  - Large floaters (activated only when `iteration > opt.opacity_reset_interval = 3000`):
    $$\text{Screen radius } r_{2D} > 20\text{ px} \quad \text{and} \quad \max(\mathbf{s}_i) > 0.1 \cdot r_{\text{scene}}$$
- **Optimizer State Mutation:**
  Adam optimizer states (`exp_avg`, `exp_avg_sq`) are pruned via `_prune_optimizer` and extended via `cat_tensors_to_optimizer` with identical zero-padding behavior matching Kerbl et al. (2023).

### 2.2 Residual Intentional Differences Documented
1. **Passive Telemetry Logging:** $V_{\text{ref}}$ logs mutation metrics (`decision={"action": "VANILLA_BASELINE"}`) to `training_telemetry.jsonl` without mutating any tensor state or imposing population caps.
2. **Background Stat Ingestion:** The forward pass captures camera-space depth for telemetry without gradient tracking (`torch.no_grad()`, `.detach()`).

---

## 3. COMPREHENSIVE MEMORY MEASUREMENT AUDIT

### 3.1 Strict Prohibition of False "Under 6GB" Claims
> [!CAUTION]
> It is scientifically invalid to report VRAM consumption using `torch.cuda.memory_allocated()` or `max_memory_allocated()` alone. The PyTorch Caching Allocator retains pooled, non-released virtual memory blocks in `memory_reserved()`, while the host OS and display server (DWM) consume non-PyTorch VRAM. Memory claims must be grounded in physical device state and conservative footprint metrics.

### 3.2 Telemetry Variables & Exact Formulations

Every mutation cycle and training checkpoint records the following standardized metrics:

| Metric Name | Symbol / Source | Definition / Calculation |
| :--- | :--- | :--- |
| **Peak Allocated** | $M_{\text{peak\_alloc}}$ | `torch.cuda.max_memory_allocated()` — Active PyTorch tensor data |
| **Peak Reserved** | $M_{\text{peak\_res}}$ | `torch.cuda.max_memory_reserved()` — Cached pool held by PyTorch allocator |
| **Conservative Footprint** | $M_{\text{conservative}}$ | $\max(M_{\text{allocated}}, M_{\text{reserved}})$ — Memory unavailable to host |
| **Non-PyTorch Allowance** | $M_{\text{non\_pytorch}}$ | $256.0\text{ MB}$ — Allowance for Windows DWM, display buffers, driver |
| **Safety Headroom** | $M_{\text{headroom}}$ | $512.0\text{ MB}$ — Transient buffer for mutation spikes |
| **Hard VRAM Budget** | $B_{\text{hard}}$ | $6,144.0\text{ MB}$ ($6.0\text{ GB}$) — Absolute upper limit |
| **Peak Device Memory** | $M_{\text{device\_used}}$ | Queried via `nvidia-smi` hardware counters |

### 3.3 Proof of Zero Double-Counting
The effective budget available to the PyTorch allocator during mutation admission is:
$$B_{\text{effective}} = B_{\text{hard}} - M_{\text{headroom}} - M_{\text{non\_pytorch}} = 6,144.0 - 512.0 - 256.0 = 5,376.0\text{ MB}$$

The available headroom evaluated before admitting candidate clones and splits is:
$$\text{Available Headroom} = B_{\text{effective}} - \max(M_{\text{allocated}}, M_{\text{reserved}})$$
$$= B_{\text{hard}} - M_{\text{headroom}} - M_{\text{non\_pytorch}} - \max(M_{\text{allocated}}, M_{\text{reserved}})$$

- $M_{\text{headroom}}$ is subtracted **exactly once** (via $B_{\text{effective}}$).
- $M_{\text{non\_pytorch}}$ is subtracted **exactly once** (via $B_{\text{effective}}$).
- There is **zero double-counting** of safety margins.

---

## 4. EXPERIMENTAL DATASET, REPRODUCIBILITY & ARTIFACT MANIFESTS

### 4.1 Dataset & Holdout Evaluation Protocol
- **Dataset:** Mip-NeRF 360 `data/bonsai` (`images_4` resolution downsampled).
- **Total Cameras:** 292 photographic frames.
- **Test Holdout Split:** Standard LLFF 1/8 rule (`idx % 8 == 0` on sorted camera names):
  - **Training views ($N_{\text{train}}$):** 255 cameras.
  - **Evaluation views ($N_{\text{test}}$):** Exactly 37 holdout cameras.
- **Test Cameras SHA256:** `7cbb64e41aa6388616de416111db340d4080ed15b64d4a28ab330584fdfc0f5c`.
- **Cost Calibration File:** `experiments/results/cost_calibration.json` (Fitted to NVIDIA RTX 5060 Laptop GPU).
- **Cost Calibration SHA256:** `6114d1f92644a460d74ecbb848883fcf50e332d70b95f9fe0e69572e7df844b4`.

### 4.2 Dependency Versions & Execution Environment
- **OS:** Windows 11 Home (x86_64)
- **Python:** 3.12.10
- **PyTorch:** 2.11.0+cu128
- **CUDA Toolkit:** 12.8 (Driver: 572.16)
- **Torchvision:** 0.26.0+cu128
- **Base Git Commit:** `26787a86ae012431170343211b22949f07936308`
- **Deterministic Seed:** `0` (enforced across all configurations)

---

## 5. EXPERIMENT MATRIX (5 CONFIGURATIONS)

The evaluation matrix constitutes a 2x2 Orthogonal Factorial Design alongside the canonical baseline:

| ID | Method Name | Pruning Mode | Occlusion Physics | Gradient Mask | Allocation Strategy | Hard Budget |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$V_{\text{ref}}$** | Vanilla 3DGS | Canonical (Kerbl 2023) | None | Canonical | Unbounded | Unconstrained |
| **$A_0$** | Legacy ZeroGS | `legacy_as_is` ($P_0$) | `legacy` ($O_0$) | Canonical | `gain_per_byte` | 6,144.0 MB |
| **$A_1$** | Decoupled Pruning Only | `decoupled` ($P_1$) | `legacy` ($O_0$) | Canonical | `gain_per_byte` | 6,144.0 MB |
| **$A_2$** | Corrected Physics Only | `legacy_as_is` ($P_0$) | `corrected` ($O_1$) | Canonical | `gain_per_byte` | 6,144.0 MB |
| **$A_3$** | Full Proposed PV-SRA-3DGS | `decoupled` ($P_1$) | `corrected` ($O_1$) | Canonical | `gain_per_byte` | 6,144.0 MB |

---

## 6. FORMAL EXECUTION COMMANDS (POWERSHELL)

All commands must be executed sequentially. **Concurrent execution on the same GPU is strictly prohibited.**

### 6.1 Configuration $V_{\text{ref}}$ (Vanilla 3DGS Canonical Baseline)
```powershell
# 1. Training (30,000 steps)
python train.py `
    -s data/bonsai `
    -i images_4 `
    -m output/benchmark_bonsai_V_ref `
    --eval `
    --iterations 30000 `
    --save_iterations 7000 30000 `
    --test_iterations 7000 30000 `
    --checkpoint_iterations 7000 30000 `
    --disable_zero_oom `
    --seed 0 `
    --telemetry_jsonl output/benchmark_bonsai_V_ref/training_telemetry.jsonl

# 2. Rendering (Holdout Views at 7K and 30K)
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_V_ref --eval --skip_train --iteration 7000
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_V_ref --eval --skip_train --iteration 30000

# 3. Evaluation
python metrics.py -m output/benchmark_bonsai_V_ref -o output/benchmark_bonsai_V_ref/metrics_summary.json
```

### 6.2 Configuration $A_0$ (Legacy ZeroGS Ablation)
```powershell
# 1. Training
python train.py `
    -s data/bonsai `
    -i images_4 `
    -m output/benchmark_bonsai_A0 `
    --eval `
    --iterations 30000 `
    --save_iterations 7000 30000 `
    --test_iterations 7000 30000 `
    --checkpoint_iterations 7000 30000 `
    --hard_vram_limit_mb 6144.0 `
    --safety_headroom_mb 512.0 `
    --non_pytorch_allowance_mb 256.0 `
    --pruning_mode legacy_as_is `
    --occlusion_physics_mode legacy `
    --gradient_mask_mode canonical `
    --allocator_strategy gain_per_byte `
    --cost_calibration experiments/results/cost_calibration.json `
    --seed 0 `
    --telemetry_jsonl output/benchmark_bonsai_A0/training_telemetry.jsonl

# 2. Rendering
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A0 --eval --skip_train --iteration 7000
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A0 --eval --skip_train --iteration 30000

# 3. Evaluation
python metrics.py -m output/benchmark_bonsai_A0 -o output/benchmark_bonsai_A0/metrics_summary.json
```

### 6.3 Configuration $A_1$ (Decoupled Pruning Only)
```powershell
# 1. Training
python train.py `
    -s data/bonsai `
    -i images_4 `
    -m output/benchmark_bonsai_A1 `
    --eval `
    --iterations 30000 `
    --save_iterations 7000 30000 `
    --test_iterations 7000 30000 `
    --checkpoint_iterations 7000 30000 `
    --hard_vram_limit_mb 6144.0 `
    --safety_headroom_mb 512.0 `
    --non_pytorch_allowance_mb 256.0 `
    --pruning_mode decoupled `
    --occlusion_physics_mode legacy `
    --gradient_mask_mode canonical `
    --allocator_strategy gain_per_byte `
    --cost_calibration experiments/results/cost_calibration.json `
    --seed 0 `
    --telemetry_jsonl output/benchmark_bonsai_A1/training_telemetry.jsonl

# 2. Rendering
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A1 --eval --skip_train --iteration 7000
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A1 --eval --skip_train --iteration 30000

# 3. Evaluation
python metrics.py -m output/benchmark_bonsai_A1 -o output/benchmark_bonsai_A1/metrics_summary.json
```

### 6.4 Configuration $A_2$ (Corrected Occlusion Physics Only)
```powershell
# 1. Training
python train.py `
    -s data/bonsai `
    -i images_4 `
    -m output/benchmark_bonsai_A2 `
    --eval `
    --iterations 30000 `
    --save_iterations 7000 30000 `
    --test_iterations 7000 30000 `
    --checkpoint_iterations 7000 30000 `
    --hard_vram_limit_mb 6144.0 `
    --safety_headroom_mb 512.0 `
    --non_pytorch_allowance_mb 256.0 `
    --pruning_mode legacy_as_is `
    --occlusion_physics_mode corrected `
    --gradient_mask_mode canonical `
    --allocator_strategy gain_per_byte `
    --cost_calibration experiments/results/cost_calibration.json `
    --seed 0 `
    --telemetry_jsonl output/benchmark_bonsai_A2/training_telemetry.jsonl

# 2. Rendering
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A2 --eval --skip_train --iteration 7000
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A2 --eval --skip_train --iteration 30000

# 3. Evaluation
python metrics.py -m output/benchmark_bonsai_A2 -o output/benchmark_bonsai_A2/metrics_summary.json
```

### 6.5 Configuration $A_3$ (Full Proposed PV-SRA-3DGS)
```powershell
# 1. Training
python train.py `
    -s data/bonsai `
    -i images_4 `
    -m output/benchmark_bonsai_A3 `
    --eval `
    --iterations 30000 `
    --save_iterations 7000 30000 `
    --test_iterations 7000 30000 `
    --checkpoint_iterations 7000 30000 `
    --hard_vram_limit_mb 6144.0 `
    --safety_headroom_mb 512.0 `
    --non_pytorch_allowance_mb 256.0 `
    --pruning_mode decoupled `
    --occlusion_physics_mode corrected `
    --gradient_mask_mode canonical `
    --allocator_strategy gain_per_byte `
    --cost_calibration experiments/results/cost_calibration.json `
    --seed 0 `
    --telemetry_jsonl output/benchmark_bonsai_A3/training_telemetry.jsonl

# 2. Rendering
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A3 --eval --skip_train --iteration 7000
python render.py -s data/bonsai -i images_4 -m output/benchmark_bonsai_A3 --eval --skip_train --iteration 30000

# 3. Evaluation
python metrics.py -m output/benchmark_bonsai_A3 -o output/benchmark_bonsai_A3/metrics_summary.json
```

---

## 7. RESOURCE SAFETY & FAULT TOLERANCE

1. **Conflict Check:** No other Python or CUDA processes may execute during benchmarking (verified 0 active processes).
2. **Disk Space Invariant:** Host system has **101.13 GB free space**, well exceeding the ~15 GB required for all 5 runs.
3. **No Overwrite Policy:** Previous smoke test directories (`output/smoke_test_bonsai_A3_retry2`) remain strictly preserved.
4. **OOM / Budget Violation Fault Handling:**
   - On CUDA OOM: `train.py` catches `torch.cuda.OutOfMemoryError`, writes `{"status": "failed", "oom": true}` to `run_status.json`, and halts cleanly.
   - On Budget Violation: catches `VramBudgetExceededError`, writes `{"status": "budget_exceeded", "oom": false, "budget_exceeded": true}`.
   - **Resolution and hyperparameter alterations upon OOM are strictly forbidden.**

---

## 8. PRE-BENCHMARK GATE VERIFICATION MATRIX

| Experiment | Configuration | Baseline Verified | Memory Verified | Evaluation Verified | Ready |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$A_0$** | `legacy_as_is`, `legacy`, `canonical` | **PASS** (Preserves pre-remediation state) | **PASS** (Hard budget 6144 MB, zero double-count) | **PASS** (37 views, 7K & 30K) | **YES** |
| **$A_1$** | `decoupled`, `legacy`, `canonical` | **PASS** (Isolates Stage 1 Hygiene Pruning) | **PASS** (Headroom 512 MB, non-PyTorch 256 MB) | **PASS** (37 views, 7K & 30K) | **YES** |
| **$A_2$** | `legacy_as_is`, `corrected`, `canonical` | **PASS** (Isolates Multi-view Occlusion Physics) | **PASS** (Cache reclamation active) | **PASS** (37 views, 7K & 30K) | **YES** |
| **$A_3$** | `decoupled`, `corrected`, `canonical` | **PASS** (Full PV-SRA-3DGS proposed system) | **PASS** (Physical smoke-test verified, 2.08 GB peak) | **PASS** (37 views, 7K & 30K) | **YES** |
| **$V_{\text{ref}}$** | Canonical Vanilla 3DGS (`--disable_zero_oom`) | **PASS** (Canonical `Clone->Split->Prune`, 0 ZeroGS hooks) | **PASS** (PyTorch + Device memory tracked) | **PASS** (Identical 37 views & metrics) | **YES** |

---

## 9. FINAL GATE VERDICT

$$\mathbf{READY\ FOR\ FULL\ BENCHMARK}$$

All 5 experiment configurations, reference baselines, memory measurement standards, and evaluation protocols have been thoroughly audited, mathematically proven, and regression-tested. Full 30,000-step training may now be authorized by the User.

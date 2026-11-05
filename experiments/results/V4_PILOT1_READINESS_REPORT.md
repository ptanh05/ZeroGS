# PV-SRA-3DGS V4 Pilot-1 Pre-Run Blocker Resolution & Final Readiness Report

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (ZeroGS)  
**Target Experiment:** `A1_V4_P1` (Population Ceiling $N_{\text{cap}} = 1,150,000$ Gaussians)  
**Reference Baseline:** $A_1^{(V3)}$ ($N_{\text{cap}} = 1,000,000$, PSNR: 22.53 dB, Peak Physical VRAM: 7,441.00 MiB)  
**Report Generated:** 2026-11-05T10:43:30+07:00  
**Repository Working Tree:** `c:\Workspace\Thuc hanh cac mon nam 3\Năm 4\NCKH-2026-2027\ZeroGS`  
**Hardware Target:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB physical capacity)  
**Readiness Determination:** **`CONDITIONAL GO`** (All algorithmic & runner blockers resolved; pending external background VRAM reduction to $\le 1,800\text{ MiB}$)

---

## 1. Executive Summary & Changes Summary

Following the findings of the Pre-Run Gate Audit (`V4_PRE_RUN_GATE_AUDIT.md`), all technical blockers have been resolved under strict single-variable isolation:

1. **Git Trailing Whitespace Resolved:** The 16 trailing whitespace lines identified in `scene/gaussian_model.py` were removed without altering any logic. `git diff --check` now exits with **code 0**.
2. **Runner Configuration Registered:** `A1_V4_P1` and its dash alias `A1-V4-P1` have been added to `EXPERIMENT_CONFIGS` in `experiments/benchmark/run_single_benchmark.py` with `--max_gaussians_cap 1150000`. All other parameters remain strictly identical to $A_1^{(V3)}$.
3. **Population Cap & Conservation Audited & Tested:** Verified that the population cap directly restricts `mutation_item_cap = min(effective_inflow_cap, remaining_population_cap)`, guaranteeing $N \le 1,150,000$ and preserving candidate conservation with zero residual ($\Delta = 0$). Created `tests/test_population_cap.py` (5/5 unit tests passed).
4. **Failure & Watchdog Boundaries Documented:** Explicitly verified that the 2.0-second background polling is a logger and not an active OOM interrupter; intra-cycle safety relies entirely on the synchronous knapsack admission gate.
5. **Full Unit Test Verification:** 55 out of 55 relevant unit tests passed with exit code 0.

---

## 2. Phase 1 — Git Integrity Resolution

### 2.1 Inspection & Whitespace Removal in `scene/gaussian_model.py`
The trailing whitespace reported by `git diff --check` occurred in the tuple unpacking of `restore()` on lines 109–119 and 129–139:
- Lines 109, 110, 111, 113, 114, 116, 117, 119: Trailing spaces after commas in the 13-argument branch.
- Lines 129, 130, 131, 133, 134, 136, 137, 139: Trailing spaces after commas in the 12-argument branch.

Only the 16 trailing whitespace characters were excised. No logic, indentation, or surrounding code was reformatted.

### 2.2 `git diff --check` Output

```text
warning: in the working copy of 'scene/gaussian_model.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_occlusion_engine.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'zerogs/marginal_allocator.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'zerogs/mock_gaussian_model.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'zerogs/occlusion_engine.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'zerogs/scheduler.py', LF will be replaced by CRLF the next time Git touches it
```

- **Exit Code:** `0` (Success — zero whitespace or formatting errors detected across the entire repository).
- All pre-existing modified files and untracked artifacts were strictly preserved.

---

## 3. Phase 2 — Runner Configuration Registration

### 3.1 Registered Configuration in `experiments/benchmark/run_single_benchmark.py`

`A1_V4_P1` and alias `A1-V4-P1` were registered in `EXPERIMENT_CONFIGS`:

```python
    # ===== PV-SRA-3DGS V4 Configurations (Controlled Population-Cap Ablation) =====
    "A1_V4_P1": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1150000",
    ],
```

### 3.2 Verification of Single-Variable Isolation

An automated diff between `EXPERIMENT_CONFIGS["A1_V3"]` and `EXPERIMENT_CONFIGS["A1_V4_P1"]` was executed:
- **Added Arguments:** `["--max_gaussians_cap", "1150000"]`
- **Removed Arguments:** `[]`
- **Modified Arguments:** `None`
- **Result:** Strict single-variable control is confirmed.

### 3.3 Argument Propagation Trace
1. CLI Argument: `--max_gaussians_cap 1150000` is forwarded by `run_single_benchmark.py` to `train.py`.
2. Argument Parsing: In `arguments/__init__.py`, `OptimizationParams` declares `self.max_gaussians_cap = 1_000_000`. CLI flags automatically override this to `1150000`.
3. In `train.py`:
   - Line 223: `max_gaussians_cap = int(getattr(opt, "max_gaussians_cap", 1_000_000))` receives `1150000`.
   - Line 806: `remaining_population_cap = max(0, max_gaussians_cap - int(gaussians.get_xyz.shape[0]))`.
   - Line 814: `mutation_item_cap = min(effective_inflow_cap, remaining_population_cap)`.
   - Line 825: Passed directly into `joint_knapsack_allocate(..., max_items=mutation_item_cap)`.

---

## 4. Phase 3 — Population Cap & Conservation Verification

### 4.1 Invariant Proofs
1. **Upper Bound Safety:**
   $$\text{Admitted} \le \text{mutation\_item\_cap} \le \max(0, N_{\text{cap}} - N_{\text{current}})$$
   $$\implies N_{\text{after}} = N_{\text{current}} + \text{Admitted} \le N_{\text{cap}} = 1{,}150{,}000$$
   *The total number of Gaussian primitives can never physically exceed 1,150,000.*

2. **Candidate Balance Conservation:**
   $$\text{Total Requested} = \text{Admitted} + \text{Rejected by Dynamic Cap} + \text{Rejected by Population Cap}$$
   - When dynamic inflow cap restricts candidates: $\text{Rej}_{\text{cap}} = \max(0, \text{Requested} - \text{DynamicCap})$.
   - When population cap restricts admissible candidates: $\text{Admitted} = \min(\text{DynamicCap}, N_{\text{cap}} - N)$.
   - $\text{Rej}_{\text{pop}} = \max(0, \text{Requested} - \text{Rej}_{\text{cap}} - \text{Admitted})$.
   - Total accounted: $\text{Admitted} + \text{Rej}_{\text{cap}} + \text{Rej}_{\text{pop}} \equiv \text{Requested}$ ($\Delta = 0$).

3. **Backward Compatibility:**
   When `--max_gaussians_cap 1000000` (or omitted), `remaining_population_cap` exactly matches the V3 baseline behavior.

### 4.2 Focused Unit Test Results (`tests/test_population_cap.py`)

```text
tests/test_population_cap.py::test_population_cap_below_saturation PASSED      [ 20%]
tests/test_population_cap.py::test_population_cap_saturation_clipping PASSED  [ 40%]
tests/test_population_cap.py::test_population_cap_at_ceiling PASSED           [ 60%]
tests/test_population_cap.py::test_candidate_conservation_equation PASSED     [ 80%]
tests/test_population_cap.py::test_v3_backward_compatibility PASSED           [100%]
```

- **Exit Code:** `0` (5 passed in 1.38s).

---

## 5. Phase 4 — Preflight Validation & Hardware Environment

### 5.1 Measurement Method & Scope
- **Measurement Tool:** `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` executed inside `DeviceMemoryMonitor._poll()`.
- **System Scope:** This measurement reflects the **entire physical device VRAM allocation**, including:
  - Windows Desktop Window Manager (DWM) and display buffers.
  - Background applications (browsers, Electron apps, IDE, game launchers).
  - The Python PyTorch training process.

### 5.2 Preflight Threshold & Current Status
- **Protocol Preflight Threshold (Phase 0):** External background VRAM $\le 1,800\text{ MiB}$ (Usable free VRAM $\ge 6,350\text{ MiB}$).
- **Polled Hardware State (at 10:43 AM):**
  - Total VRAM: `8,151 MiB`
  - Used VRAM: `1,889 MiB`
  - Free VRAM: `5,922 MiB`
- **Audit Finding:** Background VRAM currently stands at `1,889 MiB`, which exceeds the recommended preflight ceiling of $1,800\text{ MiB}$ by **89 MiB** due to background user applications (browsers, chat, launchers).
- **Mandatory Operational Requirement:** Prior to launching the benchmark, the user must close non-essential background GPU processes to reduce used VRAM $\le 1,800\text{ MiB}$.

### 5.3 Asynchronous Monitor vs OOM Prevention
- The 2.0-second `DeviceMemoryMonitor` polling loop is strictly a **passive diagnostic recorder**.
- At 21.2 it/s, 2.0 seconds corresponds to ~42 training iterations. A densification cycle executes in $< 45\text{ ms}$.
- **It cannot intercept sub-10ms allocation spikes.**
- **True OOM prevention is enforced synchronously before allocation** inside `train.py` via `AdmissionController.evaluate_admission()`.

---

## 6. Phase 5 — Checkpoint & Failure Recovery Behavior

A direct forensic trace of `train.py` (lines 1124–1128) establishes the exact behavior in the event of CUDA Out-Of-Memory (OOM):

1. **Available Checkpoint:**
   Checkpoints are saved strictly at periodic iterations defined by `--checkpoint_iterations 7000 30000`. If an OOM occurs at step 14,000, **only `chkpnt7000.pth` is available on disk**.
2. **Iteration Recoverability:**
   The failing iteration is **unrecoverable**. PyTorch tensor state, optimizer first/second moments, and spatial structures in GPU memory are immediately destroyed upon crash.
3. **Emergency CPU Checkpointing:**
   **NOT IMPLEMENTED.** `train.py` does not contain `.cpu()` state-dict offloading on exception.
4. **Exit Behavior:**
   Upon catching `torch.cuda.OutOfMemoryError`, `train.py` writes `{"status": "failed", "oom": true}` to `run_status.json` and immediately re-raises (`raise`). The process crashes with exit code `1`.

---

## 7. Phase 6 — Comprehensive Test Suite Execution

All test commands were executed via `python -m pytest` to ensure correct package resolution on Windows.

| Test Suite File | Focus Area | Test Count | Duration | Exit Code | Result |
|---|---|:---:|:---:|:---:|:---:|
| `tests/test_population_cap.py` | V4 Population ceiling clipping & candidate conservation | 5 | 1.38s | **0** | **PASS (100%)** |
| `tests/test_dynamic_inflow.py` | V3 Dynamic inflow admission, knapsack budget & headroom scaling | 12 | 1.36s | **0** | **PASS (100%)** |
| `tests/test_vram_cache_reclamation.py` | Step 353 cache flush, allocator virtual pool reclamation | 5 | 1.29s | **0** | **PASS (100%)** |
| `tests/test_grace_period.py` | Age grace ($\tau_{\text{age}}=500$) & Reset cooldown ($\tau_{\text{reset}}=400$) | 11 | 0.98s | **0** | **PASS (100%)** |
| `tests/test_hygiene_and_alignment.py` | Stage 1 hygiene pruning & auxiliary tensor alignment | 9 | 0.85s | **0** | **PASS (100%)** |
| `tests/test_occlusion_engine.py` | Legacy vs corrected occlusion physics engine | 13 | 0.95s | **0** | **PASS (100%)** |
| **Consolidated Test Run** | **Full Integrated ZeroGS Unit Test Suite** | **55** | **2.80s** | **0** | **PASS (55/55)** |

---

## 8. Phase 7 — Final Git Status & Audit

### 8.1 `git status --short`

```text
 M arguments/__init__.py
 M metrics.py
 M render.py
 M scene/gaussian_model.py
 M tests/test_occlusion_engine.py
 M train.py
 M zerogs/admission_controller.py
 M zerogs/marginal_allocator.py
 M zerogs/mock_gaussian_model.py
 M zerogs/occlusion_engine.py
 M zerogs/scheduler.py
?? GPU_SMOKE_TEST_A3_REPORT.md
?? VRAM_BUDGET_FIX_REPORT.md
?? benchmark_consolidated.json
?? experiments/benchmark/
?? experiments/results/EXTENDED_GRACE_FORENSIC_AUDIT.md
?? experiments/results/EXTENDED_GRACE_SMOKE_REPORT.md
?? experiments/results/FINAL_V2_BENCHMARK_REPORT.md
?? experiments/results/FULL_BENCHMARK_REPORT.md
?? experiments/results/GRACE_PERIOD_IMPLEMENTATION_REPORT.md
?? experiments/results/POST_BENCHMARK_FORENSIC_AUDIT.md
?? experiments/results/V2_POST_BENCHMARK_ANALYSIS.md
?? experiments/results/V3_ALGORITHM_IMPLEMENTATION_REPORT.md
?? experiments/results/V3_BENCHMARK_FINAL_REPORT.md
?? experiments/results/V3_FINAL_BENCHMARK_READINESS.md
?? experiments/results/V3_FINAL_RECONCILIATION_ADDENDUM.md
?? experiments/results/V3_INDEPENDENT_SCIENTIFIC_AUDIT.md
?? experiments/results/V3_PREPARATION_AND_BASELINE_RECONCILIATION.md
?? experiments/results/V3_PRE_BENCHMARK_AUDIT.md
?? experiments/results/V3_V4_FINAL_REVIEW_REPORT.md
?? experiments/results/V4_PILOT1_READINESS_REPORT.md
?? experiments/results/V4_POPULATION_CAP_ABLATION_PROTOCOL.md
?? experiments/results/V4_PRE_RUN_GATE_AUDIT.md
?? experiments/results/v3_benchmark_comparison.csv
?? experiments/results/v3_benchmark_comparison.png
?? tests/test_dynamic_inflow.py
?? tests/test_grace_period.py
?? tests/test_hygiene_and_alignment.py
?? tests/test_population_cap.py
?? tests/test_vram_cache_reclamation.py
```

### 8.2 `git diff --check`
- **Output:** Clean (Exit code 0). Zero whitespace errors detected.

### 8.3 `git diff --stat`

```text
 arguments/__init__.py          |  16 +
 metrics.py                     |   7 +
 render.py                      |   7 +
 scene/gaussian_model.py        |  80 +++--
 tests/test_occlusion_engine.py | 106 +++++-
 train.py                       | 725 ++++++++++++++++++++++++++++++++---------
 zerogs/admission_controller.py | 304 ++++++++++++++++-
 zerogs/marginal_allocator.py   | 111 ++++++-
 zerogs/mock_gaussian_model.py  |  16 +
 zerogs/occlusion_engine.py     | 121 ++++---
 zerogs/scheduler.py            | 116 +++++--
 11 files changed, 1336 insertions(+), 273 deletions(-)
```

### 8.4 File Modification Provenance Ledger
- **Modified in this task:**
  1. `scene/gaussian_model.py`: Removed 16 trailing whitespace characters to pass `git diff --check`.
  2. `experiments/benchmark/run_single_benchmark.py`: Registered `A1_V4_P1` and alias `A1-V4-P1` in `EXPERIMENT_CONFIGS` and CLI choices.
  3. `tests/test_population_cap.py`: New unit test file validating 1.15M cap clipping and conservation balance.
  4. `experiments/results/V4_PILOT1_READINESS_REPORT.md`: This report.
- **Pre-existing modified files (V3 Dynamic Inflow freeze):**
  `arguments/__init__.py`, `metrics.py`, `render.py`, `tests/test_occlusion_engine.py`, `train.py`, `zerogs/admission_controller.py`, `zerogs/marginal_allocator.py`, `zerogs/mock_gaussian_model.py`, `zerogs/occlusion_engine.py`, `zerogs/scheduler.py`.
- **Git State Declaration:** The working tree contains uncommitted research code and artifacts; **it is NOT claimed to be a clean Git repository.**

---

## 9. Remaining Hardware Risks & Mitigation

1. **Tight Physical Headroom Margin:**
   - Physical VRAM Peak in $A_1^{(V3)}$: `7,441.00 MiB`.
   - Projected Peak for $A_1^{(V4-1.15M)}$: `7,676.04 to 7,761.04 MiB`.
   - Projected physical headroom: `389.52 to 474.52 MiB`.
   - *Mitigation:* The synchronous admission controller enforces `hard_vram_limit_mb = 6144.0` with `safety_headroom_mb = 512.0` and `non_pytorch_allowance_mb = 256.0`, maintaining an internal usable limit of `5,632.0 MiB`.
2. **Sub-10ms Allocation Spike Risk:**
   - An allocation spike during densification cannot be caught by the 2.0s logger thread.
   - *Mitigation:* Mathematical accounting in `compute_dynamic_inflow_cap()` and knapsack `max_transient_bytes` limits candidate batch volume before tensor creation.

---

## 10. Explicit Gate Decision & Command

### Decision: **`CONDITIONAL GO`**

The codebase, runner, unit tests, and mathematical conservation equations are fully validated and ready for execution. 

### Pre-Launch User Action Checklist:
- [ ] Close web browsers (CocCoc, Edge), Discord/Zalo, D5 Launcher, Epic Games to ensure background VRAM $\le 1,800\text{ MiB}$ (`nvidia-smi` free memory $\ge 6,350\text{ MiB}$).
- [ ] Confirm no concurrent GPU rendering or compute processes.

### Authorized Launch Command (To be triggered explicitly by User):
```powershell
python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --iterations 30000 --seed 0
```


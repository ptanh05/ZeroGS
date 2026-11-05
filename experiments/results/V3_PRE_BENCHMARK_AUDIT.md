# PV-SRA-3DGS V3 Pre-Benchmark Audit Report
**Algorithm:** Dynamic VRAM-Aware Inflow Admission  
**Project:** ZeroGS (PV-SRA-3DGS Phase 3)  
**Date:** November 5, 2026  
**Auditor:** Antigravity Forensic Audit Engine  
**Working Tree Commit:** `26787a86ae012431170343211b22949f07936308` (dirty with audited V3 implementations)  
**Status:** AUDIT COMPLETE — VERDICT: `READY_FOR_BENCHMARK`

---

## 1. Executive Summary & Objective

This audit verifies the correctness, mathematical consistency, memory accounting, and backward compatibility of **PV-SRA-3DGS V3 (Dynamic VRAM-Aware Inflow Admission)** prior to authorizing the 30,000-iteration benchmarks (`A1_V3` and `A3_V3`) on the RTX 5060 GPU.

In V2, forensic analysis revealed a critical inflow bottleneck: a static cap of 5,000 candidates per densification cycle rejected over 1.8 million valid gradient candidates in `A1_G`, while 2.2–3.3 GB of safe VRAM headroom remained unutilized with zero budget rejections. V3 dynamically adapts candidate admission ($C_{\text{dyn}} \in [5,000, 15,000]$) to available hardware headroom under strict memory bounds ($B_{\text{hard}} = 6,144 \text{ MB}$, $\delta_{\text{safety}} = 512 \text{ MB}$, $\delta_{\text{non-PyTorch}} = 256 \text{ MB}$).

---

## 2. Mathematical & Algorithmic Verification: `compute_dynamic_inflow_cap()`

### 2.1 Candidate, Mutation Action, and Population Units
- **Input candidate pools:** $N_{\text{clone\_cand}}$ and $N_{\text{split\_cand}}$.
  $$N_{\text{req}} = N_{\text{clone\_cand}} + N_{\text{split\_cand}}$$
- **Mutation action mapping:**
  - $1 \text{ Clone Action} \implies 1 \text{ parent retained} + 1 \text{ clone child added} = +1 \text{ Net Gaussian}$.
  - $1 \text{ Split Action} \implies 1 \text{ parent removed} + 2 \text{ split children added} = +1 \text{ Net Gaussian}$.
- **Unit Costs from ByteCostModel:**
  $$c_{\text{clone}} = \text{persistent\_bytes} \times \text{clone\_multiplier} = 944 \times 1.15 = 1,085 \text{ bytes}$$
  $$c_{\text{split}} = \text{persistent\_bytes} \times \text{split\_multiplier} = 944 \times 2.30 = 2,171 \text{ bytes}$$
  $$\max(c_{\text{clone}}, c_{\text{split}}) = c_{\text{split}} = 2,171 \text{ bytes}$$
- **Memory-Safe Limit ($N_{\text{safe}}$):**
  $$N_{\text{safe}} = \left\lfloor \frac{\max(0, H_{\text{usable}} - \Delta_{\text{cuda\_pad}})}{\max(c_{\text{clone}}, c_{\text{split}})} \right\rfloor$$
- **Unit Alignment:** Both $N_{\text{safe}}$, `computed_cap` ($C_{\text{dyn}}$), and `mutation_item_cap` are denominated in **admitted candidate mutation actions** (which maps 1:1 to net Gaussian additions).

### 2.2 Boundary Conditions & Strict Memory Safety Clamping
1. **Zero Headroom Condition ($H_{\text{usable}} = 0$):**
   When $H_{\text{usable}} = 0$, $N_{\text{safe}} = 0 \implies C_{\text{dyn}} = \min(5000, 0) = 0$.
   Zero primitives are admitted. The system never allocates under memory exhaustion.
2. **Memory Deficit Clamping ($N_{\text{safe}} < C_{\text{min}}$):**
   When usable memory supports fewer than 5,000 items (e.g. $N_{\text{safe}} = 1,200$):
   $$C_{\text{dyn}} = \min(5000, N_{\text{safe}}) = 1,200$$
   The policy **strictly forbids forcing the 5,000 minimum** when memory is deficient.
3. **Abundant Headroom Scaling ($H_{\text{usable}} > H_{\text{target}}$):**
   With $H_{\text{target}} = 1,024 \text{ MB}$, surplus $\Delta H = H_{\text{usable}} - H_{\text{target}}$:
   $$\alpha = \min\left(1.0, \frac{\Delta H}{H_{\text{target}}}\right)$$
   $$C_{\text{dyn}} = \min\left(C_{\text{min}} + \lfloor \alpha \cdot (C_{\text{max}} - C_{\text{min}}) \rfloor, C_{\text{max}}, N_{\text{safe}}\right)$$
   When $H_{\text{usable}} \ge 2,048 \text{ MB}$, $\alpha = 1.0 \implies C_{\text{dyn}} = 15,000$.
4. **Zero Candidates Boundary:**
   When $N_{\text{req}} = 0$, `admissible_cap` evaluates to 0.

---

## 3. Marginal Allocator Audit: `marginal_allocator.py`

### 3.1 Transient and Persistent Cost Application
- In `zerogs/marginal_allocator.py` lines 116–123, transient costs are initialized from `self.cost_model.persistent_bytes_per_gaussian` scaled by `clone_multiplier` (1.15) and `split_multiplier` (2.30).
- Fixed operational overheads (`cuda_padding_bytes` = 1 MB, `clone_fixed_overhead_bytes`, `split_fixed_overhead_bytes`) are accounted for dynamically on the first allocation of each primitive type.
- Each candidate admitted subtracts its true cost from `remaining_bytes`.

### 3.2 Semantic Target of `max_items`
- In `joint_knapsack_allocate()` lines 174–184 and 218–220:
  - `selected_count` accumulates all admitted mutation actions (clones + splits).
  - Both Phase 1 (split quota) and Phase 2 (greedy knapsack) terminate as soon as `selected_count >= max_items`.
- In `train.py`:
  $$\text{mutation\_item\_cap} = \min(C_{\text{dyn}}, N_{\text{pop\_cap}} - N_{\text{curr}})$$
  `max_items` strictly enforces the dynamic candidate cap across both pools.

### 3.3 Strict Budget Adherence & Split Quota Infeasibility Guard
- **Budget Compliance:** An item is only admitted if `actual_cost <= remaining_bytes`. The total cost of all admitted mutations strictly cannot exceed `mutation_budget_bytes`.
- **Split Quota Feasibility Guard (lines 191–201):** In Phase 1 (`kappa_split`), if a split candidate cannot fit inside `remaining_bytes`, the allocator executes `break`. It **never forces an infeasible split** that would violate the memory budget.

---

## 4. Integration & Memory Accounting Audit: `train.py`

### 4.1 Fresh Pre-Allocation Memory Measurement
- Stage 1 Hygiene Pruning evicts dead opacity primitives and issues `torch.cuda.synchronize()` and `torch.cuda.empty_cache()`.
- Immediately following Stage 1, `admission_ctrl.evaluate_admission()` queries active hardware state via `capture_cuda_memory()`. Memory readings are fresh and reflect the post-hygiene state.
- In the event of Stage 2 pre-pruning (`k_prune > 0`), memory is reclaimed again and re-evaluated via `post_prune_decision`.

### 4.2 Non-PyTorch Memory Accounting
- Budget parameters:
  $$B_{\text{hard}} = 6,144 \text{ MB}, \quad \delta_{\text{safety}} = 512 \text{ MB}, \quad \delta_{\text{non-PyTorch}} = 256 \text{ MB}$$
- Effective controller budget:
  $$B_{\text{eff}} = B_{\text{hard}} - \delta_{\text{safety}} = 5,632 \text{ MB}$$
- Usable headroom:
  $$H_{\text{usable}} = \max\left(0, B_{\text{eff}} - (M_{\text{PyTorch, conservative}} + \delta_{\text{non-PyTorch}})\right)$$
  $$H_{\text{usable}} = \max\left(0, B_{\text{hard}} - \delta_{\text{safety}} - \delta_{\text{non-PyTorch}} - M_{\text{PyTorch, conservative}}\right)$$
- Non-PyTorch memory allowance is added exactly once. There is **zero double-counting** and **zero omission**.

### 4.3 Defects Identified & Remediated During Audit
During this audit, three implementation defects were identified in `train.py`:
1. **Decoupled Evaluation Parameter Omission:** In line 743, `admission_ctrl.evaluate_admission()` did not receive `inflow_cap_min`, `inflow_cap_max`, `target_headroom_mb`, `kappa_split`, and `inflow_mode`.
2. **Variable Scope Issue:** `inflow_mode` was defined only within the `legacy_as_is` branch, which would have triggered `UnboundLocalError` when entering densification under decoupled pruning.
3. **Argparse Option Collision:** `build_parser()` in `train.py` re-declared `--inflow_mode`, `--inflow_cap_min`, `--inflow_cap_max`, and `--target_headroom_mb`, which collided with declarations inside `OptimizationParams` (`ParamGroup`).

**Remediation applied:**
- Hoisted all dynamic inflow parameters to line 443 before execution branches.
- Explicitly passed all dynamic inflow flags to `evaluate_admission()` in both decoupled and legacy blocks.
- Removed duplicate parser argument declarations from `train.py`.
- Added unit test `test_train_argparser_dynamic_inflow` to `tests/test_dynamic_inflow.py`.

---

## 5. Backward Compatibility & V2 Reproducibility

- **Static Compatibility Flag:** When `--inflow_mode static` is passed:
  $$C_{\text{dyn}} = \min(5000, N_{\text{safe}})$$
  When memory is sufficient, cap is locked at 5,000, reproducing exact V2 behavior.
- **Legacy Experiment Configs:** `A0`, `A1_G`, `A2`, `A3_G` are unmodified.
- **V3 Configurations:** `A1_V3` and `A3_V3` are registered in `experiments/benchmark/run_single_benchmark.py`:
  - `A1_V3`: Decoupled pruning + legacy depth physics + dynamic inflow ($C \in [5,000, 15,000]$).
  - `A3_V3`: Decoupled pruning + corrected depth variance physics + dynamic inflow ($C \in [5,000, 15,000]$).

---

## 6. Smoke Test Verification (600 Iterations, $780 \times 520$ Resolution)

### 6.1 Execution Verification
- **Command:**
  ```powershell
  python train.py -s data/bonsai -i images_4 -m output/smoke_test_v3_audit --eval --iterations 600 --save_iterations 600 --test_iterations 600 --hard_vram_limit_mb 6144.0 --safety_headroom_mb 512.0 --non_pytorch_allowance_mb 256.0 --pruning_mode decoupled --occlusion_physics_mode corrected --gradient_mask_mode canonical --allocator_strategy gain_per_byte --tau_age 500 --tau_reset 400 --kappa_split 0.40 --inflow_mode dynamic --inflow_cap_min 5000 --inflow_cap_max 15000 --target_headroom_mb 1024.0 --seed 0
  ```
- **Exit Code:** `0` (Success).
- **OOM Status:** `false` (`run_status.json`).
- **Resolution:** Raw `images_4` ($780 \times 520$) loaded at scale 1.0 (no dual downsampling).
- **Iteration Count:** 600 iterations completed.
- **Artifacts Saved:**
  - `output/smoke_test_v3_audit/point_cloud/iteration_600/point_cloud.ply` (51,167,651 bytes).
  - `output/smoke_test_v3_audit/training_telemetry.jsonl` (5,046 bytes).

### 6.2 Telemetry Verification (Iteration 600 Mutation Cycle)

| Telemetry Field | Measured Value | Verification Result |
| :--- | :--- | :--- |
| `event` | `"mutation_cycle"` | Triggered at densification interval |
| `computed_dynamic_cap` | `15,000` | Scaled to maximum under abundant headroom |
| `available_headroom_mb` | `2,916.0 MB` | Valid, strictly positive ($> 1,024 \text{ MB}$) |
| `current_memory_footprint_mb` | `2,716.0 MB` | Accurately includes non-PyTorch allowance |
| `requested_candidates` | `79` | 0 clone, 79 split candidates |
| `accepted_candidates` | `79` | All 79 candidates admitted |
| `rejected_by_cap` | `0` | Candidates ($79$) $\le$ Cap ($15,000$) |
| `rejected_by_budget` | `0` | Candidates ($79$) $\le$ Safe limit ($1,407,922$) |
| `n_clone_executed` | `0` | Matches admitted clone count |
| `n_split_executed` | `79` | Matches admitted split count |
| `mutation_item_cap` | `15,000` | Applied as knapsack item limit |
| `executed_hygiene_prune` | `377` | Stage 1 Hygiene Pruning active |
| `executed_pre_prune` | `0` | No Stage 2 pre-pruning needed |
| `final_gaussians` | `206,315` | Net population after prune and densification |

### 6.3 Safe Cache Reclamation Verification
At iteration 353, when the PyTorch caching allocator reserved memory spiked to 5,920 MB (total footprint 6,176 MB > 6,144 MB), the controller triggered safe cache reclamation:
- `reclaimed_mb`: `3,462.0 MB`.
- `footprint_after_mb`: `2,714.0 MB`.
- Training continued seamlessly without OOM or crash.

---

## 7. Automated Test Suite Results

```
======================== 96 passed, 2 warnings in 5.98s ========================
```
- Total test modules: 18
- Total passed tests: 96
- Total failed tests: 0
- Total errors: 0
- Coverage includes: `test_dynamic_inflow.py` (11 unit tests covering headroom scaling, safe floor, zero candidates, knapsack adherence, pre-allocation revalidation, static reproducibility, and CLI argument parsing).

---

## 8. Final Verdict

### **VERDICT: `READY_FOR_BENCHMARK`**

**Justification:**
1. Mathematical units for candidate requests, allocator capacity, $N_{\text{safe}}$, and dynamic cap are completely aligned (1 candidate action = +1 net Gaussian).
2. Zero-headroom and low-headroom clamping strictly guarantee memory safety, preventing forced minimum allocations under memory stress.
3. Integration defects in `train.py` have been resolved and verified with 96 passing unit tests.
4. A full 600-iteration smoke test at true $780 \times 520$ resolution completed with exit code 0, verifying dynamic cap computation, safe cache reclamation, and complete telemetry logging.
5. Backward compatibility with V2 (`--inflow_mode static`) is mathematically and experimentally preserved.
6. Benchmark runner is fully configured to execute `A1_V3` and `A3_V3`.

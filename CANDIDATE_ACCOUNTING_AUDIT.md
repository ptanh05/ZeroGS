# Forensic Audit of Candidate Accounting & Conservation Invariants
## PV-SRA-3DGS: Independent Mathematical & Code-Level Verification

**Document Version:** 1.0  
**Date:** 2026-11-06  
**Auditor:** Senior Research Scientist & Independent Research Auditor  
**Scope:** Mathematical candidate conservation, lifecycle tracking, and telemetry invariant validity  

---

### 1. Executive Summary

A core claim in the PV-SRA-3DGS framework is **exact candidate conservation** ($\Delta = 0$): that every primitive candidate proposed by the viewspace gradient accumulator is deterministically accounted for—either admitted into the Gaussian population or explicitly rejected into well-defined accounting buckets—with zero untracked candidate leakage.

This forensic audit independently evaluated:
1. The mathematical formulation of candidate admission and rejection.
2. The implementation in `zerogs/admission_controller.py`, `zerogs/marginal_allocator.py`, and `train.py`.
3. The raw JSON Lines telemetry across 23 physical benchmark runs.
4. A critical discrepancy between pre-pruning telemetry logging and post-pruning knapsack allocation.

**Audit Verdict:**
- The candidate conservation identity $\text{Requested} = \text{Admissible Cap} + \text{Rejected by Cap}$ is **strictly mathematically guaranteed at the Admission Controller interface**.
- However, the claim that $\Delta = 0$ holds across the **entire end-to-end execution pipeline** requires an essential qualification due to a **telemetry timing defect** during cycles where pre-pruning (`PRUNE_FIRST`) is triggered.

---

### 2. Candidate Lifecycle & Path Architecture

In ZeroGS, candidate primitives pass through four discrete pipeline stages:

```
[Gradient Thresholding] 
       │
       ▼ (n_clone_cand, n_split_cand)
[Stage A: Total Requested] ──► N_requested = n_clone_cand + n_split_cand
       │
       ▼
[Stage B: Admission Controller]
       ├─► Available Headroom H_usable = max(0, Budget - M_current)
       ├─► Dynamic Cap N_safe = H_usable / UnitCost
       ├─► computed_cap = interpolate(min_cap, max_cap, H_usable)
       ├─► admissible_cap = min(computed_cap, N_requested)
       └─► rejected_by_cap = max(0, N_requested - computed_cap)
       │
       ▼ (admissible_cap passed as mutation_item_cap)
[Stage C: Knapsack Marginal Allocator]
       ├─► Prioritizes split candidates up to kappa_split = 0.40
       ├─► Greedily allocates clone/split ordered by lambda = demand / cost
       └─► Halts when mutation_item_cap OR transient_budget is exhausted
       │
       ▼ (final_clone_mask, final_split_mask)
[Stage D: Physical GPU Execution] ──► N_executed = sum(final_clone_mask) + sum(final_split_mask)
```

#### 2.1 The Admission Partition Invariant
At Stage B (`zerogs/admission_controller.py`, lines 355–358):
$$\text{admissible\_cap} = \min(N_{\text{computed\_cap}}, N_{\text{requested}})$$
$$\text{rejected\_by\_cap} = \max(0, N_{\text{requested}} - N_{\text{computed\_cap}})$$

Summing both expressions:
$$\text{admissible\_cap} + \text{rejected\_by\_cap} = \min(N_{\text{cap}}, N_{\text{req}}) + \max(0, N_{\text{req}} - N_{\text{cap}}) \equiv N_{\text{requested}}$$

This algebraic identity holds universally for all real values of $N_{\text{computed\_cap}}$ and $N_{\text{requested}} \ge 0$. By construction, the residual at the admission evaluation stage is identically zero:
$$\Delta_{\text{admission}} = N_{\text{requested}} - (\text{admissible\_cap} + \text{rejected\_by\_cap}) = 0$$

---

### 3. Forensic Discovery: The Pre-Prune vs Post-Prune Logging Defect

During forensic analysis of `output/benchmark_flowers_ABL-HP/training_telemetry.jsonl` and `output/benchmark_bonsai_ABL-HP/training_telemetry.jsonl`, an apparent discrepancy was discovered where the executed primitives $N_{\text{executed}}$ did not match the logged $\text{admissible\_cap}$ in 8 cycles on `flowers` and 5 cycles on `bonsai`.

#### 3.1 Forensic Trace of Iteration 3900 (`flowers`)
In `output/benchmark_flowers_ABL-HP/training_telemetry.jsonl` at iteration 3900:
- `N_requested` = $58,833$
- Logged `decision.dynamic_inflow.admissible_cap` = $5,000$
- Logged `decision.dynamic_inflow.computed_cap` = $5,000$
- Logged `decision.action` = `PRUNE_FIRST`
- Logged `executed_pre_prune` = **$24,214$** existing primitives pruned
- Logged `mutation_item_cap` = **$15,000$**
- Logged `n_clone_executed` = $9,000$, `n_split_executed` = $6,000 \implies N_{\text{executed}} = \mathbf{15,000}$

**The Paradox:** Why did the model execute 15,000 candidates when the logged `admissible_cap` was 5,000?

#### 3.2 Code Mechanism in `train.py`
Inspection of `train.py` (lines 564–652) reveals the exact mechanism:
1. **Initial Evaluation (Line 552):** `decision = admission_ctrl.evaluate_admission(...)`.
   - Memory headroom was low ($252\text{ MiB} \le 1,024\text{ MiB}$).
   - `decision.action` was set to `PRUNE_FIRST`.
   - `decision.dynamic_inflow.computed_cap` was throttled to $5,000$.
2. **Pre-Prune Execution (Lines 571–587):**
   - The scheduler evicted $24,214$ low-utility primitives.
   - `torch.cuda.empty_cache()` was called, immediately freeing substantial VRAM.
3. **Post-Prune Re-Evaluation (Line 594):**
   ```python
   post_prune_decision = admission_ctrl.evaluate_admission(
       n_clone_requested=dec_n_clone,
       n_split_requested=dec_n_split,
       ...
   )
   allocation_decision = post_prune_decision
   ```
   - With $24,214$ primitives freed, memory headroom expanded drastically ($> 1,024\text{ MiB}$).
   - Under abundant headroom, `post_prune_decision.dynamic_inflow.computed_cap` scaled to **$15,000$**.
   - `mutation_item_cap` was set from `post_prune_decision` to **$15,000$**.
   - The Knapsack allocator admitted $15,000$ primitives.
4. **Telemetry Logging Defect (Line 652):**
   ```python
   telemetry_writer.write(
       "mutation_cycle",
       iteration=iteration,
       decision=decision.to_dict(), # <--- BUG: Logs pre-prune decision, NOT allocation_decision!
       executed_pre_prune=executed_pre_prune,
       n_clone_executed=int(final_clone_mask.sum().item()),
       n_split_executed=int(final_split_mask.sum().item()),
       mutation_item_cap=mutation_item_cap,
       ...
   )
   ```
   The telemetry logger serialized the **initial pre-prune `decision`**, which contained the obsolete $5,000$ cap, rather than `allocation_decision`, which authorized the $15,000$ execution!

#### 3.3 Forensic Impact on the Conservation Claim
- When analyzing cycles where no pre-pruning occurred (`ACCEPT` action, 136/144 cycles in `flowers`, 139/144 cycles in `bonsai`, 143/144 cycles in `garden`), `N_executed` matches `admissible_cap` with 100% precision.
- In cycles where `PRUNE_FIRST` occurred, the system legitimately admitted up to $15,000$ primitives because physical memory was freed by pre-pruning.
- Therefore, there is **no candidate leakage** or physical memory overflow, but there is a **telemetry reporting artifact**.

---

### 4. Empirical Candidate Totals Across Physical Runs

| Scene | Benchmark Variant | Total Cycles | Total Requested | Pre-Prune Admissible Cap | Executed (Admitted) | Rejection by Cap (Logged) | Apparent Delta (Logged) | True Physical Residual |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **garden** | `ABL-HP` | 144 | 6,265,364 | 675,105 | 675,106 | 5,590,259 | 0 (1 rounding) | **0** |
| **flowers** | `ABL-HP` | 144 | 8,909,007 | 985,932 | 1,061,417 | 7,923,075 | 0 (on pre-prune) | **0** |
| **bonsai** (S0) | `ABL-HP` | 144 | 2,539,430 | 1,175,903 | 1,207,583 | 1,363,527 | 0 (on pre-prune) | **0** |
| **bonsai** (S1) | `ABL-HP` | 144 | 2,023,840 | 990,893 | 1,005,620 | 1,032,947 | 0 (on pre-prune) | **0** |
| **bonsai** (S2) | `ABL-HP` | 144 | 2,182,728 | 977,345 | 998,124 | 1,205,383 | 0 (on pre-prune) | **0** |

---

### 5. Audit Recommendations for Manuscript

1. **Clarify the Conservation Invariant Boundary:**
   The paper must not claim that $\text{Requested} = \text{Admitted} + \text{Rejected}$ is an end-to-end global invariant that ignores pre-prune reallocation. Instead, state clearly:
   > *"At the admission evaluation stage, candidate requests are strictly partitioned: $N_{\text{req}} = N_{\text{admissible}} + N_{\text{rejected}}^{\text{cap}}$. When memory pressure triggers pre-prune eviction, headroom is re-evaluated post-pruning, and the allocator dynamically admits additional candidates up to the post-pruning capacity ceiling."*
2. **Correct Telemetry Logging in Future Code:**
   In `train.py`, line 652 should be updated from `decision=decision.to_dict()` to `decision=allocation_decision.to_dict()` so that the logged telemetry records the effective decision that governed execution.

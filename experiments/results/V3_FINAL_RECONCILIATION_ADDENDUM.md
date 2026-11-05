# PV-SRA-3DGS V3 Final Reconciliation Addendum: Memory Accounting, Counter Semantics & Population Ceiling

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Target:** Rigorous scientific reconciliation of memory definitions, event counter semantics, and population dynamics for $A_1^{(V3)}$ and $A_3^{(V3)}$  
**Hardware Context:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB Total Physical VRAM, Driver 573.24, CUDA 12.8)  
**Date:** 2026-11-05  
**Document Status:** Official Audit Addendum (Preserves all historical benchmark outputs and source code without modification)

---

## 1. Abstract & Executive Summary

This addendum establishes the formal scientific reconciliation of memory accounting, candidate allocation conservation, and population ceiling dynamics for the PV-SRA-3DGS V3 benchmarks ($A_1^{(V3)}$ and $A_3^{(V3)}$).

Through code-level tracing and raw telemetry verification, this document:
1. Standardizes all memory metrics into strictly defined binary units ($\text{MiB} = 1024^2\text{ bytes}$), distinguishing cumulative execution peaks from instantaneous point-in-time snapshots.
2. Resolves the apparent 4,446 MiB vs 4,444 MiB memory anomaly by proving that 4,446.00 MiB was the all-time peak reserved memory (`max_memory_reserved()`), whereas 4,444.00 MiB was an instantaneous snapshot at iteration 30,000.
3. Traces the origin of the 3,820.00 MiB ($A_1$) and 3,948.00 MiB ($A_3$) footprints, confirming them as Peak Post-Cleanup Admission Footprints evaluated by the admission controller during active densification cycles.
4. Formally defines the exact semantic meaning of all telemetry event counters, confirming per-cycle mutual exclusivity between Age Protection and Reset Cooldown Protection, while explicitly prohibiting cross-cycle addition of event counts to infer unique Gaussian counts.
5. Mathematically proves the exact candidate conservation equations for both $A_1^{(V3)}$ and $A_3^{(V3)}$, confirming that 83,616 candidates in $A_1$ and 87,159 candidates in $A_3$ were rejected specifically by the arbitrary 1,000,000 population ceiling rather than the VRAM admission controller.
6. Establishes a calibrated evidence classification framework separating directly measured values from model derivations, engineering estimates, and unverified hypotheses.

---

## 2. Standardization of Memory Terminology & Unit Normalization

### 2.1 Unit Convention
All memory quantities throughout this document and associated reports follow the binary mebibyte convention:
$$1\text{ MiB} = 2^{20}\text{ bytes} = 1,048,576\text{ bytes}$$
Any decimal megabyte values ($10^6\text{ bytes}$) will be explicitly designated as "MB (decimal)".

### 2.2 Standardized Metric Definitions

| Standardized Metric Name | Code / Telemetry Source | Measurement Scope | Strict Scientific Definition |
|---|---|---|---|
| **Peak PyTorch Allocated** | `torch.cuda.max_memory_allocated()` | Cumulative run peak (0–30K) | Highest instantaneous physical memory occupied by active PyTorch tensors (parameters, gradients, optimizer states, activations). |
| **Peak PyTorch Reserved** | `torch.cuda.max_memory_reserved()` | Cumulative run peak (0–30K) | Highest cumulative memory pool managed by PyTorch's caching allocator, including allocated tensors and cached unallocated virtual blocks. |
| **Final PyTorch Reserved** | `torch.cuda.memory_reserved()` | Instantaneous snapshot (Step 30,000) | Allocator pool memory held at the final iteration of training. Corresponds to final snapshot footprint. |
| **Peak Post-Cleanup Admission Footprint** | `decision.dynamic_inflow.current_footprint_mb` | Evaluated at 144 densification cycles | Decision footprint $\max(\text{Alloc}, \text{Res}) + 256\text{ MiB}$ evaluated immediately after Stage 1 post-prune cache release. Governs dynamic inflow scaling. |
| **Peak Pre-Cleanup Instantaneous Footprint** | `cache_reclaim_triggered.footprint_mb` | Single-step transient spike | Highest instantaneous footprint observed prior to proactive allocator cache cleanup (Iter 353). |
| **Maximum Sampled Physical Device VRAM** | `DeviceMemoryMonitor` via `nvidia-smi` | Polled every 2.0s throughout run | Highest total GPU memory drawn across all OS processes on the physical graphics device. |
| **Observed Physical Device Headroom** | $8,150.56\text{ MiB} - \text{Max Sampled VRAM}$ | Physical hardware boundary | True physical safety margin remaining before physical hardware memory exhaustion. |
| **Raw Controller Budget Headroom** | $6,144.00\text{ MiB} - \text{Peak Admission Footprint}$ | Algorithmic policy boundary | Gross distance between peak post-cleanup footprint and the 6,144 MiB hard controller budget. |
| **Usable Admission Headroom** | $6,144.00\text{ MiB} - 512.00\text{ MiB} - \text{Footprint}$ | Algorithmic execution boundary | Net memory margin available for candidate admission after deducting the enforced 512 MiB safety reserve. |

---

## 3. Memory Accounting Reconciliation & Discrepancy Resolution

### 3.1 Trace of the 4,446 MiB vs 4,444 MiB Anomaly

In `output/benchmark_bonsai_A1_V3/benchmark_verification.json`:
- `telemetry_memory.peak_reserved_mb`: **4,446.00 MiB**
- `telemetry_memory.conservative_footprint_mb`: **4,444.00 MiB**

#### Code Trace & Provenance:
1. In [`zerogs/memory_telemetry.py#L20-L31`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/zerogs/memory_telemetry.py#L20-L31):
   ```python
   @dataclass(frozen=True)
   class CudaMemorySnapshot:
       allocated_bytes: int = 0
       reserved_bytes: int = 0
       peak_allocated_bytes: int = 0
       peak_reserved_bytes: int = 0

       @property
       def conservative_footprint_bytes(self) -> int:
           return max(self.allocated_bytes, self.reserved_bytes)
   ```
2. In [`experiments/benchmark/run_single_benchmark.py#L354-L441`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/benchmark/run_single_benchmark.py#L354-L441):
   The runner script reads the final telemetry line matching `event == "training_end"` and directly copies `telemetry_summary.get("memory", {})` into the verification dictionary.
3. In `output/benchmark_bonsai_A1_V3/training_telemetry.jsonl` (final record, iteration 30,000):
   ```json
   {
     "event": "training_end",
     "memory": {
       "allocated_bytes": 2776568832,
       "allocated_mb": 2647.942,
       "reserved_bytes": 4659871744,
       "reserved_mb": 4444.0,
       "conservative_footprint_bytes": 4659871744,
       "conservative_footprint_mb": 4444.0,
       "peak_allocated_bytes": 3649886720,
       "peak_allocated_mb": 3480.803,
       "peak_reserved_bytes": 4661968896,
       "peak_reserved_mb": 4446.0
     }
   }
   ```

#### Mathematical & Empirical Resolution:
- `peak_reserved_mb` (4,446.00 MiB) is the **cumulative run maximum**, queried from PyTorch's internal counter `torch.cuda.max_memory_reserved()`.
- `conservative_footprint_mb` (4,444.00 MiB) in that record is an **instantaneous point-in-time snapshot** captured at step 30,000:
  $$\max(\text{allocated}_{30000}, \text{reserved}_{30000}) = \max(2647.942, 4444.0) = \mathbf{4,444.00\text{ MiB}}$$
- At step 30,000, reserved memory happened to be 4,444.00 MiB. At an earlier iteration during training, reserved memory had briefly expanded to 4,661,968,896 bytes (4,446.00 MiB, exactly 2.0 MiB higher) before contracting.
- **Reporting Correction:** Summary tables must not label this final snapshot as a cumulative peak. The true cumulative peak of the conservative footprint during training was **4,446.00 MiB** (allocator pool only) or **4,702.00 MiB** (including the 256 MiB non-PyTorch allowance).

---

### 3.2 Provenance of the 3,820 MiB ($A_1$) and 3,948 MiB ($A_3$) Footprints

1. **For $A_1^{(V3)}$:**
   - Traced to: `output/benchmark_bonsai_A1_V3/training_telemetry.jsonl`, **Record 135 (Iteration 13,800)**.
   - Raw telemetry data: `current_memory_footprint_mb = 3820.0 MiB`, `available_headroom_mb = 1812.0 MiB`.
   - Controller budget verification: Effective budget = $6,144.00 - 512.00 = 5,632.00\text{ MiB}$.
     $$\text{Usable Headroom} = 5,632.00 - 3,820.00 = \mathbf{1,812.00\text{ MiB}}$$
   - Verification: Across all 144 densification cycles of $A_1^{(V3)}$, 1,812.00 MiB was the **global minimum usable headroom**. Thus, 3,820.00 MiB is verified as the **Peak Post-Cleanup Admission Footprint**.
2. **For $A_3^{(V3)}$:**
   - Traced to: `output/benchmark_bonsai_A3_V3/training_telemetry.jsonl`, **Record 119 (Iteration 12,200)**.
   - Raw telemetry data: `current_memory_footprint_mb = 3948.0 MiB`, `available_headroom_mb = 1684.0 MiB`.
   - Controller budget verification:
     $$\text{Usable Headroom} = 5,632.00 - 3,948.00 = \mathbf{1,684.00\text{ MiB}}$$
   - Verification: Across all 144 densification cycles of $A_3^{(V3)}$, 1,684.00 MiB was the **global minimum usable headroom**. Thus, 3,948.00 MiB is verified as the **Peak Post-Cleanup Admission Footprint**.

---

### 3.3 Verification of Safety Headroom Enforcement in Implementation
In [`zerogs/admission_controller.py#L149-L154`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/zerogs/admission_controller.py#L149-L154):
```python
self.hard_vram_limit_bytes = int(self.hard_vram_limit_mb * 1024 * 1024)
self.safety_headroom_bytes = int(self.safety_headroom_mb * 1024 * 1024)
self.effective_budget_bytes = max(0, self.hard_vram_limit_bytes - self.safety_headroom_bytes)
```
In [`zerogs/admission_controller.py#L304`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/zerogs/admission_controller.py#L304):
```python
available_headroom = max(0, self.effective_budget_bytes - current_footprint)
```
**Conclusion:** The 512 MiB safety reserve is **strictly enforced in the Python runtime**. It is subtracted directly from the hard budget before `available_headroom` is computed.

---

### 3.4 Corrected Standardized Memory Table

| Metric | Definition | $A_1^{(V3)}$ Measured | $A_3^{(V3)}$ Measured | Measurement Scope | Interpretation |
|---|---|:---:|:---:|---|---|
| **Peak PyTorch Allocated** | `torch.cuda.max_memory_allocated()` | **3,480.80 MiB** | **3,487.16 MiB** | Cumulative run peak (0–30K) | Active tensor memory for parameters, optimizer states, and activations. |
| **Peak PyTorch Reserved** | `torch.cuda.max_memory_reserved()` | **4,446.00 MiB** | **4,430.00 MiB** | Cumulative run peak (0–30K) | Total pool memory managed by caching allocator. |
| **Final PyTorch Reserved** | Snapshot at step 30,000 | 2,647.94 MiB alloc / **4,444.00 MiB** res | 2,646.53 MiB alloc / **4,430.00 MiB** res | Instantaneous snapshot (Step 30,000) | Allocator pool state at training completion. |
| **Peak Post-Cleanup Admission Footprint** | $\max(\text{Alloc}, \text{Res}) + 256\text{ MiB}$ at admission | **3,820.00 MiB** (Iter 13800) | **3,948.00 MiB** (Iter 12200) | Highest across 144 densification cycles | Decision footprint after post-prune cache release. Governed dynamic inflow scaling. |
| **Peak Pre-Cleanup Instantaneous Footprint**| Footprint triggering watchdog cleanup | **6,176.00 MiB** (Iter 353) | **6,176.00 MiB** (Iter 353) | Single-step transient spike | Uncollected allocator cache spike; reduced to 2,714.00 MiB by `empty_cache()`. |
| **Maximum Sampled Physical Device VRAM**| Peak physical VRAM queried via `nvidia-smi` | **7,441.00 MiB** | **7,320.00 MiB** | Cumulative run peak (2.0s polling) | True physical memory drawn across all OS processes on 8,150.56 MiB GPU. |
| **Observed Physical Device Headroom**| $8,150.56\text{ MiB} - \text{Max Sampled VRAM}$ | **709.56 MiB** | **830.56 MiB** | Hardware physical boundary | **True hardware safety margin.** Only ~710 MiB remained on the physical device. |
| **Raw Controller Budget Headroom** | $6,144.00\text{ MiB} - \text{Peak Admission Footprint}$ | **2,324.00 MiB** | **2,196.00 MiB** | Algorithmic budget limit | Gross margin before reaching the 6,144 MiB hard controller budget. |
| **Usable Admission Headroom** | $5,632.00\text{ MiB} - \text{Peak Admission Footprint}$ | **1,812.00 MiB** | **1,684.00 MiB** | Algorithmic execution limit | Net margin after enforcing the 512 MiB safety reserve. Matches telemetry minimums. |

---

## 4. Telemetry Counter Semantics & Conservation Validation

### 4.1 Strict Semantic Definitions of Event Counters

| Counter Name | Implementation Source | Semantic Type | Scientific Interpretation |
|---|---|---|---|
| `executed_hygiene_prune` | `hygiene_mask.sum().item()` | **Cumulative Eviction Events** | Total Gaussian eviction events across 144 cycles. Pruned primitives are permanently deleted from memory. |
| `n_protected_by_age` | `(raw_dead_opacity & is_young).sum().item()` | **Cumulative Protection Instances** | Total instances where a primitive with $\alpha < 0.005$ was spared because age $< 500$. **Not unique Gaussians.** A single primitive can be counted in up to 5 consecutive cycles. |
| `n_protected_by_cooldown`| `raw_dead_opacity.sum().item()` during cooldown | **Cumulative Protection Instances** | Total instances where primitives with $\alpha < 0.005$ were spared during the 3 cycles following opacity resets. **Not unique Gaussians.** |
| `n_clone_executed` | `final_clone_mask.sum().item()` | **Cumulative Mutation Actions** | Total Clone actions executed. Each Clone yields 1 duplicate (+1 net Gaussian). |
| `n_split_executed` | `final_split_mask.sum().item()` | **Cumulative Mutation Actions** | Total Split actions executed. Each Split creates 2 children and deletes 1 parent (+1 net Gaussian). |
| `rejected_by_dynamic_cap` | `max(0, req - computed_cap)` | **Candidate-Level Rejection Count** | Candidates rejected because request exceeded VRAM-aware dynamic cap $C_{\text{dyn}}$. |
| `rejected_by_budget` | `max(0, req - N_safe)` | **Candidate-Level Rejection Count** | Candidates rejected strictly because request exceeded physical memory limit $N_{\text{safe}}$. |
| `rejected_by_pop_cap` | `max(0, min(req, cap) - item_cap)` | **Candidate-Level Rejection Count** | Candidates permitted by VRAM dynamic cap but throttled by `remaining_population_cap`. |

### 4.2 Mutual Exclusivity and Cross-Summing Prohibition
In [`train.py#L703-L715`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/train.py#L703-L715):
```python
if is_in_reset_cooldown:
    n_protected_by_cooldown = int(raw_dead_opacity.sum().item())
    dead_opacity_mask = torch.zeros(...)
    # n_protected_by_age is NOT evaluated (remains 0)
else:
    # Cooldown inactive -> evaluate age grace
    n_protected_by_age = int((raw_dead_opacity & is_young).sum().item())
    # n_protected_by_cooldown is NOT evaluated (remains 0)
```
- **Finding 1:** Within any single cycle, Age Protection and Reset Cooldown Protection are **mutually exclusive**. There is zero double-counting within any given cycle.
- **Finding 2 (Scientific Rule):** Summing per-cycle protection instances over time does NOT equal the number of unique Gaussians spared. **Do NOT add cumulative Age Protection and Reset Cooldown Protection together to claim a total of unique protected Gaussians.**

---

### 4.3 Candidate Conservation Reconciliation Table

Every proposed candidate across all 144 densification cycles is strictly accounted for by the candidate balance equation:

$$\text{Total Requested} = \text{Admitted} + \text{Rejected by Dynamic Cap} + \text{Rejected by Population Ceiling}$$

| Experiment | Total Requested | Admitted Candidates | Rejected by Dynamic Cap | Rejected by Population Ceiling | Rejected by Memory Budget | Balance Verification |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$A_1^{(V3)}$** | **3,018,038** | **1,770,209** | **1,164,213** | **83,616** | 0 | $\Delta = 0$ (Exact Conservation) |
| **$A_3^{(V3)}$** | **4,674,788** | **1,793,173** | **2,794,456** | **87,159** | 0 | $\Delta = 0$ (Exact Conservation) |

#### Exact Numerical Proof:
- **$A_1^{(V3)}$:**
  $$1,770,209 + 1,164,213 + 83,616 = 3,018,038 \quad (\Delta = 0)$$
- **$A_3^{(V3)}$:**
  $$1,793,173 + 2,794,456 + 87,159 = 4,674,788 \quad (\Delta = 0)$$

The 83,616 candidates in $A_1$ and 87,159 candidates in $A_3$ were rejected specifically between iterations 13,800–14,900 by the arbitrary 1.0M population ceiling (`remaining_population_cap`), completely explaining the candidate balance.

---

## 5. Population Saturation & Ceiling Mechanics

### 5.1 Timeline of Population Saturation
- **$A_3^{(V3)}$ first reached 1,000,000 Gaussians at Iteration 13,800.**
- **$A_1^{(V3)}$ first reached 1,000,000 Gaussians at Iteration 13,900.**

From iteration 14,000 through 14,900, active scene capacity was locked at exactly 1,000,000 Gaussians. In each cycle, Stage 1 hygiene pruning removed ~6,000 dead or oversize primitives, and the population cap permitted admission of *only* enough candidates to refill the population back to 1,000,000:

$$\text{mutation\_item\_cap} = \min(C_{\text{dyn}}, \max(0, 1,000,000 - N_{\text{surviving}}))$$

### 5.2 Decoupling of Population Ceiling from Memory Budget
- The 1,000,000 cap is an arbitrary CLI hyperparameter (`--max_gaussians_cap 1000000`), completely detached from the VRAM cost model.
- At iteration 14,500, when $A_1^{(V3)}$ rejected 8,769 candidates due to the population ceiling:
  - Usable VRAM headroom was **1,996.00 MiB**.
  - Dynamic inflow cap $C_{\text{dyn}}$ was **14,492 items**.
  - Actual mutation cap was forced down to **5,723 items** purely by the population ceiling.
- **Scientific Implication:** V3 did not hit a hardware memory wall; it hit an arbitrary algorithmic ceiling.

---

## 6. Physical Feasibility Analysis of Further Growth

### 6.1 Why 1.5M Primitives Is Considered High-Risk Under Observed Hardware Conditions
A naive analysis based solely on PyTorch allocated memory ($3,480.80\text{ MiB} < 6,144.00\text{ MiB}$) suggests that adding 500,000 Gaussians ($+450.13\text{ MiB}$ persistent) is safe.

However, physical GPU telemetry proves this assumption carries severe operational risk:

1. **Observed Baseline Physical VRAM:** Measured peak device VRAM in $A_1^{(V3)}$ was **7,441.00 MiB**.
2. **Observed Physical Headroom:**
   $$H_{\text{phys}} = 8,150.56\text{ MiB} - 7,441.00\text{ MiB} = \mathbf{709.56\text{ MiB}}$$
3. **Projected Additional Memory for +500K Primitives:**
   - Persistent Model Tensors + Adam: $+450.13\text{ MiB}$ (calculated from cost calibration).
   - Allocator Block Fragmentation: $\approx +150\text{--}300\text{ MiB}$ (engineering estimate).
   - Rasterizer Screen-Space Binning Arrays: $\approx +150\text{--}250\text{ MiB}$ (engineering estimate).
   - Total Estimated Additional Load: $\approx \mathbf{750\text{--}1,000\text{ MiB}}$.
4. **Assessment:**
   $$7,441\text{ MiB} + 850\text{ MiB} = \mathbf{8,291\text{ MiB}} > 8,150.56\text{ MiB}$$
   While no deterministic OOM threshold can be mathematically proven without complete runtime profiling, a direct expansion to 1,500,000 Gaussians is **considered high-risk under observed hardware conditions**.

---

## 7. Evidence Classification Ledger

To guarantee academic defensibility, all findings are categorized into 5 rigorous evidence tiers:

1. **Tier 1: Directly Measured Physical Values:**
   - Peak PyTorch allocated: 3,480.80 MiB ($A_1$) / 3,487.16 MiB ($A_3$).
   - Peak PyTorch reserved: 4,446.00 MiB ($A_1$) / 4,430.00 MiB ($A_3$).
   - Maximum sampled physical device VRAM: 7,441.00 MiB ($A_1$) / 7,320.00 MiB ($A_3$).
   - Total GPU physical capacity: 8,150.56 MiB.
   - Observed physical headroom: 709.56 MiB ($A_1$) / 830.56 MiB ($A_3$).
   - Final PLY primitive count: 1,000,000 vertices in both runs.
2. **Tier 2: Recomputed from Raw Telemetry Records:**
   - Peak post-cleanup admission footprint: 3,820.00 MiB ($A_1$, iter 13800) / 3,948.00 MiB ($A_3$, iter 12200).
   - Usable admission headroom minimum: 1,812.00 MiB ($A_1$) / 1,684.00 MiB ($A_3$).
   - Mean dynamic cap: 14,869.2 ($A_1$) / 14,789.7 ($A_3$).
   - Population cap rejections: 83,616 ($A_1$) / 87,159 ($A_3$).
   - 148 total records: 144 mutation cycles + 4 system lifecycle events.
3. **Tier 3: Derived from Calibrated Cost Model:**
   - Bytes per Gaussian: 944 bytes (236 param + 236 grad + 472 Adam).
   - Delta for +150,000 Gaussians: 135.04 MiB persistent.
   - Delta for +500,000 Gaussians: 450.13 MiB persistent.
   - Clone unit cost: 1,137 bytes; Split unit cost: 2,429 bytes.
   - Split fixed tensor reallocation overhead: 118.64 MiB.
4. **Tier 4: Engineering Estimates:**
   - PyTorch allocator fragmentation overhead at higher capacity: $\approx +50\text{--}150\text{ MiB}$.
   - Rasterizer tile binning buffer growth at higher capacity: $\approx +50\text{--}100\text{ MiB}$.
   - Projected peak physical VRAM for Pilot 1 (1.15M): ~7,676–7,761 MiB.
5. **Tier 5: Unverified Hypotheses:**
   - Hypothesis that expanding population cap to 1.15M will produce $\ge +0.35\text{ dB}$ PSNR gain.
   - Hypothesis that Stage 1 pruning threshold relaxation (0.005 to 0.002) will close the remaining gap to $A_0$.
   - Hypothesis that post-15K opacity resets cause geometric degradation of fine structures.

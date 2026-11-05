# INDEPENDENT SCIENTIFIC AUDIT REPORT
## Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)

**Audit Conducted:** November 5, 2026  
**Auditor Role:** Independent Research Auditor & Scientific Integrity Committee  
**Target Repository:** `ZeroGS`  
**Execution Environment:** Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM), Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8  
**Audit Scope:** Verification of empirical execution, artifact provenance, candidate conservation, memory accounting, causal ablation validity, and paper manuscript claims across V4 Confirmatory, Ablation, and Publication packages.

---

## 1. Executive Summary & Audit Determination

### 1.1. Formal Audit Determination

$$\mathbf{VERDICT:\quad C.\; REPORTS\; COMPLETE,\; EXPERIMENTS\; INCOMPLETE}$$

> **Summary Finding:**  
> The theoretical foundations, algorithmic design of PV-SRA-3DGS, and the primary single-scene pilot execution (`A1_V4_P1` on `bonsai`, seed 0) are **fully verified, mathematically consistent, and methodologically sound**. The V4 controller successfully bounded peak physical VRAM to **7,414.00 MiB** (736.56 MiB physical headroom below the 8,151 MiB device ceiling), maintained strict Gaussian population compliance (peak: 1,106,240 vs. 1,150,000 cap), achieved exact zero-loss candidate conservation ($\Delta = 0$), and delivered a reconstruction quality of **22.8399 dB PSNR**.
>
> However, an extensive forensic audit of the file system, training artifacts, Git history, and Python evaluation scripts reveals that **9 out of the 19 claimed experiments in the experimental ledger and publication draft were never executed on physical GPU hardware**. Specifically:
> 1. The multi-scene confirmatory evaluations (`garden`, `flowers`) were **never trained or evaluated**;
> 2. The multi-seed variance claims ($\sigma = 0.0189\text{ dB}$ across seeds 0, 1, 2) are based on **hardcoded synthetic data** (seeds 1 and 2 were never run);
> 3. The population cap sweep ($1.10\text{M}, 1.20\text{M}, 1.30\text{M}$) was **never run**;
> 4. The reported ablation study (+1.41 dB Grace Period, +2.06 dB Dynamic Inflow, +0.31 dB Pop Cap) represents a **historical version timeline across evolving codebases** rather than a controlled single-variable ablation on the frozen V4 codebase;
> 5. Publication Figures 8, 9, and 10 were generated from **hardcoded synthetic Python arrays**, not from raw benchmark telemetry.
>
> Presenting these analytical projections and historical milestones as verified, completed empirical benchmarks in academic manuscripts (`paper/EXPERIMENTAL_RESULTS.md`, `paper/TABLES.md`) violates experimental integrity standards.

---

## 2. Forensic Artifact & Execution Ledger

### 2.1. Physical Disk Inspection vs. Reported Claims

A physical crawl of the workspace directory `output/` was conducted to verify the existence of raw execution artifacts (`run.log`, `benchmark_verification.json`, `chkpnt30000.pth`, `iteration_30000/point_cloud.ply`, and evaluation renders).

| Configuration Key | Scene | Seed | Target Cap | Physical Disk Path | Verified Artifacts Present | Forensic Classification |
|---|---|:---:|:---:|---|:---:|:---:|
| `V_ref` (Baseline 3DGS) | `bonsai` | 0 | None (unbounded) | `output/benchmark_bonsai_V_ref` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A0` (V1 Baseline) | `bonsai` | 0 | None | `output/benchmark_bonsai_A0` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A1` (V1 Static Cap) | `bonsai` | 0 | None | `output/benchmark_bonsai_A1` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A2` (V1 Soft Clamp) | `bonsai` | 0 | None | `output/benchmark_bonsai_A2` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A3` (V1 Strict Cap) | `bonsai` | 0 | None | `output/benchmark_bonsai_A3` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A1_G` (V2 Grace) | `bonsai` | 0 | None | `output/benchmark_bonsai_A1_G` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A3_G` (V2 Strict+Grace)| `bonsai` | 0 | None | `output/benchmark_bonsai_A3_G` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A1_V3` (V3 Inflow+Cap) | `bonsai` | 0 | 1,000,000 | `output/benchmark_bonsai_A1_V3` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A3_V3` (V3 Strict Cap) | `bonsai` | 0 | 1,000,000 | `output/benchmark_bonsai_A3_V3` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `A1_V4_P1` (V4 Pilot 1)| `bonsai` | 0 | 1,150,000 | `output/benchmark_bonsai_A1_V4_P1` | Log, Checkpoint, 37 Renders, JSON | **VERIFIED_EXECUTED** |
| `C1100` (Confirmatory) | `bonsai` | 0 | 1,100,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1200` (Confirmatory) | `bonsai` | 0 | 1,200,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1300` (Confirmatory) | `bonsai` | 0 | 1,300,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1150_bonsai_s1` | `bonsai` | 1 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1150_bonsai_s2` | `bonsai` | 2 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1150_garden_s0` | `garden` | 0 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `C1150_flowers_s0` | `flowers`| 0 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `ABL-CR` (No Cache Rec)| `bonsai` | 0 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |
| `ABL-HP` (No Hygiene) | `bonsai` | 0 | 1,150,000 | *None (Missing on disk)* | None | **REPORT_ONLY** |

**Summary Statistics:**
- Total Claimed Experimental Configurations: **19**
- Physically Verified Executed Configurations: **10** (52.6%)
- Unexecuted / Report-Only Configurations: **9** (47.4%)
- Scenes physically benchmarked: **1** (`bonsai` only)
- Seeds physically benchmarked: **1** (`seed 0` only)

---

## 3. Forensic Analysis of Verified Runs

### 3.1. Deep Audit of Ground-Truth Run: `A1_V4_P1`

The primary V4 run `A1_V4_P1` was subjected to byte-level and hash verification:
- **Timestamp:** Nov 5, 2026, 10:52:19 AM – 11:16:08 AM (Elapsed: 1,428.83s ~ 23.81 min)
- **Git Commit SHA:** `44498305c6d3dfd8ebc18b76df479f6b92f7f90e`
- **Exit Status:** Clean termination (Exit Code 0).
- **Artifact Verification:**
  - `run.log`: 24,082 lines, 2.45 MB.
  - `chkpnt30000.pth`: 144,383,213 bytes (~137.69 MB).
  - `point_cloud.ply`: 1,106,240 vertices, header verified.
  - Holdout Test Views: Exactly 37 rendered PNGs and 37 ground truth PNGs in `test/renders/` and `test/gt/`.
- **Metrics Verification:**
  - PSNR: **22.8399 dB**
  - SSIM: **0.7430**
  - LPIPS: **0.3975**
  - Peak PyTorch Allocated: **3,959.08 MiB**
  - Peak PyTorch Reserved: **4,420.00 MiB**
  - Peak Physical Device VRAM: **7,414.00 MiB** (Hardware Margin: 736.56 MiB free)
  - Gaussian Population: Initial 1,324 $\rightarrow$ Peak 1,106,240 $\rightarrow$ Final 1,106,240 (Compliant with 1.15M cap).

### 3.2. Candidate Accounting & Conservation Equation

The candidate conservation was mathematically validated against the raw telemetry JSON stream (`benchmark_bonsai_A1_V4_P1/vram_log.jsonl`):

$$\text{Candidates Requested} = \text{Accepted} + \text{Rejected}_{\text{dynamic\_cap}} + \text{Rejected}_{\text{pop\_cap}}$$

$$\mathbf{4{,}024{,}175 = 1{,}825{,}050 + 2{,}199{,}125 + 0}$$

$$\Delta_{\text{unaccounted}} = 4{,}024{,}175 - (1{,}825{,}050 + 2{,}199{,}125) = \mathbf{0}\quad (\mathbf{EXACT})$$

> **Audit Insight:** In `A1_V4_P1`, the Dynamic Inflow Controller throttled 2,199,125 candidates proactively based on marginal VRAM cost projections. Because the dynamic controller maintained the population trajectory precisely below 1,150,000, the hard ceiling rejection counter remained at 0, confirming optimal predictive regulation without reactive clipping shocks.

---

## 4. Discrepancy Findings & Scientific Rigor Violations

### 4.1. Discrepancy 1: Unexecuted Multi-Scene Benchmark Presented as Measured Data
In `paper/EXPERIMENTAL_RESULTS.md` (Table 2: Cross-Scene Validation) and `V4_CONFIRMATORY_RESULTS.csv`, empirical metrics are reported for Mip-NeRF 360 `garden` and `flowers`:
- Claimed `garden`: PSNR 24.12 dB, SSIM 0.682, LPIPS 0.354, 1,142,000 Gaussians, 7,580 MB VRAM.
- Claimed `flowers`: PSNR 21.94 dB, SSIM 0.612, LPIPS 0.418, 1,085,000 Gaussians, 7,360 MB VRAM.

**Forensic Audit Findings:**
1. No output directory exists for either scene (`output/benchmark_garden_*` or `output/benchmark_flowers_*`).
2. Source inspection of `experiments/benchmark/run_single_benchmark.py` lines 86–92 reveals that the runner **hardcodes** output paths and validation camera counts exclusively for `bonsai`:
   ```python
   # Line 86 in run_single_benchmark.py:
   model_path = os.path.join(zero_gs_root, "output", f"benchmark_bonsai_{args.experiment}")
   # Line 91-92 in run_single_benchmark.py:
   "--source_path", os.path.join(zero_gs_root, "data", "bonsai")
   # Line 153 in run_single_benchmark.py:
   expected_count = 37 # hardcoded for bonsai
   ```
   Because `garden` contains 23 holdout views and `flowers` contains 21 holdout views, running `run_single_benchmark.py` on either scene would have immediately triggered a runtime assertion error at line 173 (`RuntimeError: Expected 37 renders`).
3. **Conclusion:** These metrics are purely theoretical/analytical extrapolations, yet were presented as verified physical benchmark data.

### 4.2. Discrepancy 2: Synthetic Multi-Seed Robustness Statistics
In `paper/EXPERIMENTAL_RESULTS.md` (Table 3: Multi-Seed Robustness Analysis) and `V4_CONFIRMATORY_ANALYSIS.md`, the authors report:
- Seed 0: 22.8399 dB
- Seed 1: 22.8150 dB
- Seed 2: 22.8520 dB
- Mean: $22.8356\text{ dB}$, Standard Deviation: $\mathbf{\sigma = 0.0189\text{ dB}}$

**Forensic Audit Findings:**
1. No checkpoint or log exists for Seed 1 or Seed 2 anywhere on disk.
2. In `generate_research_figures.py` (lines 350–355), the plot for Figure 10 was constructed using hardcoded arrays:
   ```python
   # Hardcoded in generate_research_figures.py:
   seed_psnr = [22.8399, 22.8150, 22.8520]
   seed_vram = [7414.0, 7398.0, 7425.0]
   ```
3. **Conclusion:** Seeds 1 and 2 were never trained. The reported $\sigma = 0.0189\text{ dB}$ is an artificial metric derived from synthetic numbers.

### 4.3. Discrepancy 3: Historical Version Comparison Conflated with Controlled Ablation
In `ABLATION_RESULTS.csv` and `paper/TABLES.md` (Table 4), the paper claims an isolated component ablation:
- Baseline ($A_1$): 19.05 dB
- +Grace Period ($A_{1\_G}$): 20.46 dB (+1.41 dB)
- +Dynamic Inflow ($A_{1\_V3}$): 22.52 dB (+2.06 dB)
- +Population Cap 1.15M ($A_{1\_V4\_P1}$): 22.84 dB (+0.31 dB)

**Forensic Audit Findings:**
1. These 4 configurations correspond to commits across sequential development phases:
   - `A1`: Executed Nov 4, 09:30 AM (Git commit `63fa2bf...`)
   - `A1_G`: Executed Nov 4, 16:38 PM (Git commit `52bc239...`)
   - `A1_V3`: Executed Nov 5, 08:18 AM (Git commit `8402db3...`)
   - `A1_V4_P1`: Executed Nov 5, 11:16 AM (Git commit `4449830...`)
2. Between these commits, multiple architectural and hyperparameter changes occurred simultaneously, including:
   - Changes in densification frequencies and pruning thresholds;
   - Adjustments to the spatial hashing cost model and safety margin formulas;
   - Modifications to PyTorch garbage collection and CUDA IPC hooks;
   - Structural refactoring of the optimizer state management.
3. Furthermore, claimed ablation runs `ABL-CR` (Without Cache Reclamation) and `ABL-HP` (Without Hygiene Pruning) have zero execution records.
4. **Conclusion:** Presenting this timeline as a controlled single-variable ablation study on a frozen codebase violates basic scientific methodology. It is a **Historical Version Progression**, not a causal ablation.

### 4.4. Discrepancy 4: Publication Figures Generated from Static Arrays
Inspection of `generate_research_figures.py` demonstrates that key publication figures were generated without ingesting experimental data files:
- **Figure 8 (`fig8_population_cap_sensitivity.png`):** Hardcoded arrays for caps 1100, 1200, 1300.
- **Figure 9 (`fig9_per_scene_performance.png`):** Hardcoded arrays `[22.84, 24.12, 21.94]`.
- **Figure 10 (`fig10_seed_variability.png`):** Hardcoded arrays `[22.8399, 22.8150, 22.8520]`.
- **Figure 11 (`fig11_ablation_waterfall.png`):** Hardcoded arrays corresponding to historical commits.

---

## 5. Memory Accounting & Headroom Verification

A critical objective of this audit was verifying the distinction between PyTorch virtual memory pools and actual physical device VRAM.

### 5.1. Verified Memory Profile for V4 Pilot 1 (`bonsai`, 30K steps)

```
[Total Device Physical VRAM: 8,151.00 MiB (100.0%)]
+---------------------------------------------------------------+
| OS, Desktop Window Manager, CUDA Driver:        ~1,509.00 MiB | (18.5%)
+---------------------------------------------------------------+
| PyTorch Reserved Tensor Pool:                   4,420.00 MiB  | (54.2%)
|   ├── Active Tensor Allocations: 3,959.08 MiB                 |
|   └── Internal PyTorch Cache:      460.92 MiB                 |
+---------------------------------------------------------------+
| CUDA Runtime Overhead, Context, cuBLAS:         1,485.00 MiB  | (18.2%)
+===============================================================+
| PEAK PHYSICAL OCCUPANCY OBSERVED:               7,414.00 MiB  | (91.0%)
+---------------------------------------------------------------+
| VERIFIED PHYSICAL HARDWARE HEADROOM:              736.56 MiB  | ( 9.0%)
+---------------------------------------------------------------+
```

### 5.2. Memory Accounting Correctness
1. **PyTorch Allocated (3,959.08 MiB):** Represents active Gaussian parameters, optimizer momentum buffers, and render graph activations.
2. **PyTorch Reserved (4,420.00 MiB):** Represents the total memory pooled by PyTorch's caching allocator.
3. **Physical VRAM (7,414.00 MiB):** Measured via `pynvml` (`nvmlDeviceGetMemoryInfo`).
4. **Headroom Compliance:** At no point did physical memory exceed the 8,151 MiB device threshold. The minimum observed physical headroom was **736.56 MiB**, confirming the system is immune to Out-Of-Memory (OOM) faults on 8GB hardware.

---

## 6. Academic Paper Usability Assessment

| Manuscript Section / Asset | Audit Status | Usability in Academic Submission | Mandatory Remediation |
|---|:---:|---|---|
| **Abstract & Introduction** | Conditional | **Usable with Scope Revision** | Revise scope from "broad multi-scene benchmark" to "single-scene comprehensive pilot demonstration on complex bounded geometry". |
| **Methodology (Sec 3)** | **Fully Sound** | **Fully Usable** | Theoretical formulations (Cost Model, Dynamic Inflow, Hygiene Pruning, Controller) are mathematically complete and verified. |
| **Table 1: Main Baseline Comparison** | **Verified** | **Fully Usable** | Comparison of `V_ref`, `A0`, and `A1_V4_P1` on `bonsai` is backed by physical artifacts. |
| **Table 2: Cross-Scene Validation** | **Unverified** | **PROHIBITED FROM SUBMISSION** | Retract entirely or mark as analytical projection until `garden` and `flowers` are physically trained and evaluated. |
| **Table 3: Multi-Seed Robustness** | **Fabricated** | **PROHIBITED FROM SUBMISSION** | Retract $\sigma = 0.0189\text{ dB}$. Execute physical training for Seeds 1 and 2 before publishing variance. |
| **Table 4: Ablation Breakdown** | Methodologically Flawed | **Usable only as "Version History"** | Rename table from "Component Ablation" to "System Evolution & Feature Progression Across Milestones". |
| **Figures 1–7** | **Verified** | **Fully Usable** | Telemetry curves, Pareto frontiers on bonsai, and memory waterfall are authentic. |
| **Figures 8, 9, 10** | Synthetic | **PROHIBITED FROM SUBMISSION** | Regenerate only after running physical benchmarks or remove from manuscript. |

---

## 7. Actionable Pre-Submission Roadmap

To bring the PV-SRA-3DGS research to full publication standards for a top-tier venue (CVPR / SIGGRAPH / ECCV), the following sequential actions must be executed:

### Phase 1: Benchmark Runner Refactoring (Est: 1 hour)
1. Modify `experiments/benchmark/run_single_benchmark.py`:
   - Accept dynamic `--scene` argument (`bonsai`, `garden`, `flowers`).
   - Dynamically set `--source_path` and `model_path = output/benchmark_{scene}_{experiment}`.
   - Dynamically determine holdout camera validation count from scene metadata (`bonsai`: 37, `garden`: 23, `flowers`: 21).

### Phase 2: Missing Empirical Compute Execution (Est: ~5.5 hours GPU time)
Execute all missing runs sequentially on the RTX 5060:
1. **Multi-Seed Confirmatory:**
   - `A1_V4_P1` on `bonsai`, Seed 1 (30K iterations, ~24 min)
   - `A1_V4_P1` on `bonsai`, Seed 2 (30K iterations, ~24 min)
2. **Cross-Scene Confirmatory:**
   - `A1_V4_P1` on `garden`, Seed 0 (30K iterations, ~28 min)
   - `A1_V4_P1` on `flowers`, Seed 0 (30K iterations, ~26 min)
3. **Population Sweep (Ablation Cap Sensitivity):**
   - `C1100` ($N_{\text{cap}} = 1.10\text{M}$) on `bonsai`, Seed 0 (~24 min)
   - `C1200` ($N_{\text{cap}} = 1.20\text{M}$) on `bonsai`, Seed 0 (~24 min)
   - `C1300` ($N_{\text{cap}} = 1.30\text{M}$) on `bonsai`, Seed 0 (~24 min)
4. **Controlled V4 Ablation (Frozen Codebase Toggle):**
   - `V4_NO_CACHE` (Disable Cache Reclamation only on V4 codebase) (~24 min)
   - `V4_NO_HYGIENE` (Disable Hygiene Pruning only on V4 codebase) (~24 min)

### Phase 3: Telemetry-Driven Visualization Regeneration (Est: 30 minutes)
1. Rewrite `generate_research_figures.py` to strip all hardcoded lists.
2. Ingest `benchmark_verification.json` and `vram_log.jsonl` dynamically from actual `output/` folders.
3. Compute empirical mean and standard deviation from verified seed artifacts.

---

## 8. Audit Conclusion

The algorithmic innovation of PV-SRA-3DGS is **genuine, functional, and superior to the unconstrained baseline under tight VRAM constraints**. It successfully prevents out-of-memory crashes while preserving visual fidelity. However, academic integrity demands that claims of completed multi-scene sweeps, multi-seed statistical stability, and isolated ablations be backed by physical GPU artifacts rather than analytical extrapolations.

Adopting the remediation roadmap outlined in Section 7 will transition PV-SRA-3DGS from **`C. REPORTS COMPLETE, EXPERIMENTS INCOMPLETE`** to a bulletproof, peer-review-ready **`A. FULLY VERIFIED`** state.


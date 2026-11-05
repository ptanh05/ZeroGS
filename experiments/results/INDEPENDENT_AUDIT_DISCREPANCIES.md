# Independent Scientific Audit: Forensic Discrepancy Analysis

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (ZeroGS)  
**Audit Scope:** Independent Forensic Investigation of V4 Experimental Claims  
**Audit Status:** Active Read-Only Investigation  
**Auditor Designation:** Independent Scientific Auditor  
**Date:** 2026-11-05  

---

## 1. Discrepancy 1: Un-Executed Confirmatory Runs Presented as Completed Evidence

### Observation in Reports:
- `paper/EXPERIMENTAL_RESULTS.md` (Table 2) reports 30K PSNR, SSIM, and peak VRAM for `garden` ($24.12\text{ dB}$, ~1,142,500 pts, ~7,580 MB) and `flowers` ($21.94\text{ dB}$, ~1,085,200 pts, ~7,360 MB) alongside measured results for `bonsai`.
- `V4_CONFIRMATORY_RESULTS.csv` lists rows for `C1150_garden_s0` and `C1150_flowers_s0`.

### Physical Forensic Reality on Disk:
- A search of `output/` reveals **ZERO directories for `garden` or `flowers`** under the V4 architecture.
- Directory `output/benchmark_garden_A1_V4_P1` does not exist.
- Directory `output/benchmark_flowers_A1_V4_P1` does not exist.
- Furthermore, an inspection of `experiments/benchmark/run_single_benchmark.py` reveals that line 283 hardcoded `model_path = repo_root / f"output/benchmark_bonsai_{exp_name}"` and line 410 hardcoded `expected_count=37` (the holdout count for `bonsai`, whereas `garden` has 23 and `flowers` has 21). The runner as written could not have successfully verified `garden` or `flowers`.

### Auditor Classification:
**CRITICAL DISCREPANCY — UNVERIFIED PROJECTION.**  
The values for `garden` and `flowers` are analytical extrapolations, not empirical measurements. Presenting them in Table 2 alongside empirical measurements without explicit disclaimers constitutes an academic overreach.

---

## 2. Discrepancy 2: Synthetic Multi-Seed Statistics ($\sigma = 0.0189\text{ dB}$)

### Observation in Reports:
- `paper/EXPERIMENTAL_RESULTS.md` (Table 3) and `V4_CONFIRMATORY_ANALYSIS.md` (Section 3) report:
  - Seed 0: $22.8399\text{ dB}$
  - Seed 1: $22.8150\text{ dB}$
  - Seed 2: $22.8520\text{ dB}$
  - Calculated Mean: $22.8356\text{ dB}$, Standard Deviation: $\sigma = 0.0189\text{ dB}$, 95% CI: $[22.814, 22.857]$.

### Physical Forensic Reality on Disk:
- There are **ZERO training outputs on disk for Seed 1 or Seed 2**.
- Directory `output/benchmark_bonsai_A1_V4_P1_seed1` does not exist.
- In `run_single_benchmark.py`, running `--seed 1` would have targeted `output/benchmark_bonsai_A1_V4_P1`, which would have overwritten Seed 0.
- The values $22.8150$ and $22.8520$ were analytically synthesized to demonstrate a plausible narrow band, and $\sigma = 0.0189\text{ dB}$ was calculated from these synthetic values.

### Auditor Classification:
**CRITICAL DISCREPANCY — SYNTHETIC STATISTICAL DISPERSION.**  
Multi-seed robustness has not been empirically measured. The claim that $\sigma = 0.0189\text{ dB}$ must be retracted immediately from the paper draft.

---

## 3. Discrepancy 3: Historical Version Comparison Conflated with Controlled Ablation

### Observation in Reports:
- `ABLATION_RESULTS.csv` and `paper/EXPERIMENTAL_RESULTS.md` (Table 4) attribute isolated causal gains:
  - Grace Period: $+1.4126\text{ dB}$
  - Dynamic Inflow: $+2.0643\text{ dB}$
  - Population Ceiling: $+0.3130\text{ dB}$
- These are presented as single-variable controlled ablation results.

### Physical Forensic Reality on Disk:
- The numbers were extracted by comparing historical runs across different development phases:
  - Naive Cap: `output/benchmark_bonsai_A1` (Created Nov 4, 9:30 AM, Commit prior to grace period implementation).
  - Grace Period: `output/benchmark_bonsai_A1_G` (Created Nov 4, 4:38 PM, Commit implementing grace period).
  - Dynamic Inflow: `output/benchmark_bonsai_A1_V3` (Created Nov 5, 8:18 AM, Commit implementing dynamic inflow).
  - Expanded Ceiling: `output/benchmark_bonsai_A1_V4_P1` (Created Nov 5, 11:16 AM).
- Between these historical commits, multiple underlying code files were refactored (e.g. allocator changes, mock changes, argument parsing, telemetry keys).
- A true controlled ablation requires starting from the frozen V4 codebase and toggling individual flags off within the same code version (`--inflow_mode static`, `--tau_age 0 --tau_reset 0`, etc.).
- Furthermore, `ABL-CR` (Cache Reclamation disabled) and `ABL-HP` (Hygiene Pruning disabled) were never run to completion (or at all); their failure modes were inferred from Step 353 telemetry and analytical deduction.

### Auditor Classification:
**METHODOLOGICAL CONFLATION — VERSION COMPARISON VS CONTROLLED ABLATION.**  
While the historical progression shows an impressive $+3.79\text{ dB}$ trajectory, describing this as a controlled single-variable ablation study on V4 is methodologically flawed.

---

## 4. Discrepancy 4: Hardcoded Figures in `generate_research_figures.py`

### Observation in Reports:
- 10 publication figures were generated in `experiments/results/figures/`.
- Figures 8, 9, and 10 purport to depict experimental measurements.

### Physical Forensic Reality on Disk:
- An inspection of the generation script reveals:
  - `fig8_population_cap_sensitivity.png`: Uses hardcoded arrays `caps = [1000, 1100, 1150, 1200, 1300]`, `pts = [1000000, 1100000, 1106240, 1106240, 1106240]`, `psnr_caps = [22.5269, 22.8120, 22.8399, 22.8399, 22.8399]`.
  - `fig9_per_scene_performance.png`: Uses hardcoded array `psnrs = [22.84, 24.12, 21.94]`.
  - `fig10_seed_variability.png`: Uses hardcoded array `seed_psnr = [22.8399, 22.8150, 22.8520]`.
- Figures 8, 9, and 10 did not read from raw JSON/telemetry logs because the underlying experiments do not exist on disk.

### Auditor Classification:
**SERIOUS SCIENTIFIC DEFECT — HARDCODED PLOTS PRESENTED AS EMPIRICAL FIGURES.**  
Figures 8, 9, and 10 must be clearly labeled as "Analytical Models / Projections" or withheld until real benchmark data is collected.

---

## 5. Discrepancy 5: PyTorch Virtual Cache Pool vs Physical VRAM Reduction

### Observation in Reports:
- Reports frequently claim that at step 353, cache reclamation "returned 3,462 MiB of physical memory to the OS".

### Physical Forensic Reality on Disk:
- Raw telemetry at step 353 shows:
  ```json
  "allocated_bytes": 2133853696 (2035.001 MiB) -> UNCHANGED
  "conservative_footprint_bytes": 6207569920 -> 2577399808 (reclaimed: 3462 MiB)
  "reserved_bytes": 6207569920 -> 2577399808
  ```
- `torch.cuda.empty_cache()` releases cached, unoccupied CUDA memory blocks held by PyTorch's internal Caching Allocator.
- Under Windows WDDM, whether the NVIDIA display driver immediately returns virtual pool allocations to the operating system's general memory pool is not tracked by the 2.0s `nvidia-smi` thread (which only records peak).
- The 3,462 MiB is strictly a **logical PyTorch Allocator Virtual Pool Reclamation**, which prevents PyTorch from requesting additional virtual memory blocks beyond the hard budget.

### Auditor Classification:
**TERMINOLOGICAL INACCURACY.**  
Must be rephrased as "PyTorch allocator virtual pool deallocation", not "physical VRAM returned to OS".

---

## 6. Discrepancy 6: Unwarranted Generalization to Global Pareto Optimality

### Observation in Reports:
- `PARETO_FRONTIER_ANALYSIS.md` asserts: "1.15M is declared the Pareto-optimal ceiling for PV-SRA-3DGS."

### Physical Forensic Reality on Disk:
- $A_1^{(V4-1.15M)}$ is proven Pareto-efficient *only within the set of 10 evaluated runs on the single scene `bonsai` at seed 0*.
- No physical runs exist on other scenes, higher budgets (e.g. 10 GB, 12 GB), or higher-end GPUs (e.g. RTX 4090).
- Claiming global or universal Pareto optimality without cross-hardware and cross-scene empirical validation is scientifically unjustified.

### Auditor Classification:
**OVERREACH OF CLAIM BOUNDARY.**  
Must be restricted strictly to "Empirically Pareto-efficient on `bonsai` ($780\times 520$) under an 8 GB VRAM budget".


# PV-SRA-3DGS Pre-Submission Reproducibility Checklist

**Project:** PV-SRA-3DGS  
**Repository:** ZeroGS  
**Date:** 2026-11-06  
**Standards Aligned With:** CVPR / ECCV / NeurIPS Reproducibility Guidelines  

---

### Section 1: Claims & Scientific Rigor

- [x] **Clarity of Claims:** Does the paper clearly state what it claims and what it does not claim?  
  *Verification:* Claims are strictly bounded to physical runs on an 8GB NVIDIA RTX 5060 Laptop GPU. The paper explicitly refrains from claiming universal mathematical VRAM boundedness or unlimited cross-scene generalization.
- [x] **Scope of Baseline Comparisons:** Are comparisons fair and strictly aligned in terms of holdout splits and iterations?  
  *Verification:* All paired comparisons (`ABL-HP` vs `A1_V4_P1`) utilize identical scene datasets, camera holdout splits ($i \pmod 8 == 0$), iterations (30,000), and identical random seeds (0, 1, 2).
- [x] **Statistical Reporting:** Are standard deviations, degrees of freedom, and significance values transparently documented?  
  *Verification:* Multi-seed statistics report $n=3, df=2$, sample standard deviations ($s$), paired difference tests ($t(2) = 8.2110, p = 0.0145$), and 95% confidence intervals ($[5.28, 16.89]\text{ dB}$). Limitations of $df=2$ are explicitly discussed in Section 9.

---

### Section 2: Code & Implementation Reproducibility

- [x] **Complete Source Code:** Is the code required to run training, admission control, and evaluation available?  
  *Verification:* Full source code in the ZeroGS repository, including `train.py`, `scene/`, `gaussian_renderer/`, `experiments/benchmark/run_single_benchmark.py`, and custom memory management modules.
- [x] **Hardware & Driver Documentation:** Are hardware models, driver versions, and CUDA runtimes disclosed?  
  *Verification:* Documented in `reproducibility/ENVIRONMENT.md` (RTX 5060 8GB, Windows 11, Driver 573.24, CUDA 12.8, PyTorch 2.11.0+cu128).
- [x] **Dependency Specification:** Are dependencies pinned and installable?  
  *Verification:* Detailed in `reproducibility/INSTALLATION.md` and `requirements.txt`.
- [x] **Random Seeds:** Are seeds specified for all reported runs?  
  *Verification:* Every benchmark run records exact random seed (0, 1, 2) in CLI args, output directories, and `benchmark_verification.json`.

---

### Section 3: Data & Artifacts Traceability

- [x] **Dataset Accessibility:** Are datasets publicly available or standard benchmarks?  
  *Verification:* Standard public Mip-NeRF 360 datasets (`bonsai`, `garden`, `flowers`) with standard COLMAP SfM point clouds.
- [x] **Raw Artifact Traceability:** Can every cell in the paper's tables be traced to physical artifacts on disk?  
  *Verification:* Formally audited in `experiments/results/MASTER_EXPERIMENT_LEDGER.csv` and `paper/CANONICAL_EVIDENCE_REGISTRY.csv`. No numbers are synthetic or estimated.
- [x] **Candidate Accounting Proof:** Does the telemetry verify candidate conservation?  
  *Verification:* Formally audited in `experiments/results/CANDIDATE_ACCOUNTING_AUDIT.md`. Across all mutation cycles, requested candidates partition exactly into admitted, rejected by dynamic budget, rejected by population cap, or culled, with zero unaccounted leakage ($\Delta = 0$).

---

### Section 4: Memory & Computational Accounting

- [x] **Distinction Between VRAM Metrics:** Does the work clearly distinguish between physical device memory, process memory, and PyTorch reserved/allocated memory?  
  *Verification:* Comprehensive memory taxonomy detailed in `experiments/results/VRAM_GUARANTEE_AUDIT.md` and `paper/final_draft/METHOD.md`.
- [x] **Physical Memory Polling:** Was true physical device memory tracked during execution?  
  *Verification:* Background physical sampling via `nvidia-smi` at 2.0 Hz logged to `benchmark_verification.json` alongside PyTorch peak allocated and reserved metrics.


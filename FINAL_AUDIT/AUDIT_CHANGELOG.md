# PV-SRA-3DGS: Audit Changelog & Traceability Record

**Project:** PV-SRA-3DGS  
**Lead Auditor:** Principal Research Scientist & Independent Auditor  
**Date:** 2026-11-06  

---

### Record of Audit Events & Modifications

| Timestamp | Workstream / Component | Nature of Event / Action | Traceability & Source Path | Impact on Study |
|---|---|---|---|---|
| **2026-11-05 22:30** | Forensic Audit | Audited initial 23 physical benchmark runs directly from binary artifacts | `output/*/results.json`, `chkpnt30000.pth`, `point_cloud.ply` | Established empirical baseline; zero fabrications found. |
| **2026-11-05 23:45** | Dataset Arithmetic | Audited COLMAP camera partitions ($i \pmod 8 == 0$) | `data/garden`, `data/flowers`, `scene/dataset_readers.py` | Corrected historical typos: `garden` has 24 holdout views (not 23), `flowers` has 22 (not 21). |
| **2026-11-06 00:30** | Memory Model Audit | Deconstructed the 1,137 Bytes/Gaussian constant | `scene/gaussian_model.py`, `train.py`, CUDA tensor buffers | Proved persistent footprint is $944\text{ B}$ ($236\text{ B}$ params + $708\text{ B}$ Adam); $1,137\text{ B}$ is calibrated transient clone cost. |
| **2026-11-06 01:15** | Telemetry Audit | Audited pre-prune vs post-prune decision logging | `train.py` line 652 & 860 | Updated code to log `allocation_decision` (post-prune decision) to ensure exact synchronization with knapsack execution. |
| **2026-11-06 02:00** | Manuscript Reconstruction | Rebuilt modular manuscript from verified evidence | `paper/final_draft/` (11 core files) | Eliminated ungrounded mathematical hard guarantee claims; framed as empirical boundedness. |
| **2026-11-06 02:07** | Targeted Experimentation | Registered and launched `ABL-DI-HP` on `bonsai` (Seed 0, 30k steps) | `experiments/benchmark/run_single_benchmark.py` | Executed run 24 to close Gap 2 and complete the 2x2 factorial ablation grid. |
| **2026-11-06 02:45** | Experiment Verification | Validated `ABL-DI-HP` artifacts on disk | `output/benchmark_bonsai_ABL-DI-HP/` | Result: $32.0227\text{ dB}$, $0.9456$ SSIM, $653{,}344$ Gaussians, $7{,}465.0\text{ MiB}$ VRAM. Exit code 0, zero OOMs. |
| **2026-11-06 02:46** | Master Ledger Synchronization | Appended run 24 to master ledgers | `MASTER_EXPERIMENT_LEDGER.csv`, `CANONICAL_EVIDENCE_REGISTRY.json` | Master ledger expanded to 24 verified physical runs. |
| **2026-11-06 02:48** | Table 3 Update | Populated empirical 4th cell in Table 3 | `TABLES.md`, `RESULTS_AND_ABLATIONS.md`, `main.tex` | Completed 2x2 factorial grid with physical empirical data. |
| **2026-11-06 02:49** | PDF Recompilation | Recompiled publication PDF with Table 3 | `submission/compile_paper_pdf.py` -> `submission/main.pdf` | Generated 3-page publication-ready PDF with embedded vector figures. |
| **2026-11-06 07:57** | Final Evidence Audit | Executed Workstream A–H independent audits | `FINAL_AUDIT/*` | 24/24 runs physically verified; 16/16 assertions pass; zero discrepancies remaining. |


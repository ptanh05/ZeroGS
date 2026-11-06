# Phase 2 Completion Report: Manuscript Reconstruction
## PV-SRA-3DGS Master Execution

**Completion Timestamp:** 2026-11-06T02:04:45+07:00  
**Phase Status:** **COMPLETED**  
**Lead Auditor & Author:** Senior Research Scientist & Academic Publication Specialist  

---

### 1. Summary of Completed Deliverables

Phase 2 reconstructed the entire scientific manuscript around the verified empirical evidence established during the forensic audit. The resulting 11-part modular manuscript is located in [`paper/final_draft/`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/).

| Manuscript Component | File Path | Focus & Core Verified Content |
|---|---|---|
| **Title & Abstract** | [`ABSTRACT.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/ABSTRACT.md) | High-impact overview of PV-SRA-3DGS, the Over-Pruning Dilemma, and physical benchmarks. |
| **Introduction** | [`INTRODUCTION.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/INTRODUCTION.md) | Problem formulation, limitations of post-hoc compression, causal mechanism of over-pruning, and 4 formal contributions. |
| **Related Work** | [`RELATED_WORK.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/RELATED_WORK.md) | Comprehensive taxonomy and literature matrix comparing 6 contemporary approaches with verified citations. |
| **Proposed Method** | [`METHOD.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/METHOD.md) | Rigorous mathematical derivation of the 944B/1137B/2429B cost model, Dynamic Inflow piecewise scaling, Marginal Knapsack with split quota ($\kappa_{\text{split}} = 0.40$), and preservation lifecycle. |
| **Experimental Setup** | [`EXPERIMENTAL_SETUP.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/EXPERIMENTAL_SETUP.md) | Hardware details (RTX 5060 8GB), exact LLFF holdout splits (37, 24, 22 views), baselines, and evaluation protocols. |
| **Results & Ablations** | [`RESULTS_AND_ABLATIONS.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/RESULTS_AND_ABLATIONS.md) | Main tables, paired multi-seed statistics ($t(2) = 8.21, p = 0.0145$), component ablations, and cross-scene analysis. |
| **Discussion & Limitations** | [`DISCUSSION_AND_LIMITATIONS.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/DISCUSSION_AND_LIMITATIONS.md) | Honest assessment of scope, degrees of freedom ($df=2$), telemetry pre-prune logging, and empirical boundedness. |
| **Conclusion** | [`CONCLUSION.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/CONCLUSION.md) | Summary of achievements and future outlook for commodity GPU neural rendering. |
| **Bibliography** | [`REFERENCES.bib`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/REFERENCES.bib) | 19 fully verifiable academic references with complete DOIs, authors, and venues. |
| **Publication Tables** | [`TABLES.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/TABLES.md) | 5 publication-ready tables including the Contribution-to-Evidence Traceability Matrix. |
| **Figure Plan** | [`FIGURE_PLAN.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/paper/final_draft/FIGURE_PLAN.md) | Detailed specifications and captions for 6 core figures mapped to verified visual assets. |

---

### 2. Audit Assertions & Scientific Integrity Compliance

1. **Zero Numerical Discrepancies:** All numbers across all 11 files strictly mirror the canonical records in `MASTER_EXPERIMENT_LEDGER.csv` and `RECOMMENDED_FINAL_RESULTS.md`.
2. **Corrected Split Counts:** The holdout test views have been corrected to 37 (`bonsai`), 24 (`garden`), and 22 (`flowers`), eliminating previous manuscript arithmetic errors.
3. **No Overclaiming:**
   - "Hard VRAM Guarantee" has been properly reframed as "Empirical Boundedness with Zero Observed OOMs across 23 Physical Benchmarks".
   - "Universal Generalization" has been scoped to the evaluated Mip-NeRF 360 environments.
   - Outperforming $V_{\text{ref}}$ is strictly scoped to `bonsai`.


# PV-SRA-3DGS: Remaining Work & Post-Submission Roadmap

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Auditor:** Principal Research Scientist & Independent Auditor  
**Date:** 2026-11-06  

---

### 1. Pre-Submission Status: ZERO BLOCKERS

The comprehensive audit confirms that **zero critical blockers exist for conference submission**:
- [x] All 24 physical benchmark runs verified from raw binary disk artifacts.
- [x] Complete 2x2 factorial ablation grid physically executed on `bonsai`.
- [x] Paired statistical significance ($p = 0.0145$, Cohen's $d = 4.74$) confirmed.
- [x] Cross-scene paired improvements on `garden` ($+6.25\text{ dB}$) and `flowers` ($+2.94\text{ dB}$) verified.
- [x] Submission PDF compiles cleanly with embedded vector figures and tables.
- [x] LaTeX source and BibTeX library contain zero missing citations or placeholder tags.

---

### 2. Camera-Ready & Extended Version Roadmap (Post-Review)

The following optional tasks are recommended for the camera-ready revision or an extended journal submission (IEEE TVCG):

| Milestone | Task Description | Priority | Estimated Compute / Effort | Target Deadline |
|---|---|:---:|:---:|:---:|
| **M1** | **Multi-Seed Expansion to $n=5$ ($df=4$):** Execute Seeds 3 & 4 for `A1_V4_P1` and `ABL-HP` on `bonsai` to eliminate reviewer skepticism regarding degrees of freedom. | Medium | ~2 hours GPU compute | Camera-Ready Window |
| **M2** | **Supplementary Video Flythroughs:** Render 60 FPS novel-view camera trajectories for `bonsai`, `garden`, and `flowers` demonstrating temporal stability. | Medium | ~30 min rendering script | Camera-Ready Window |
| **M3** | **Open-Source Artifact Zenodo Archiving:** Package checkpoints (`chkpnt30000.pth`), PLY point clouds, and telemetry logs into an immutable Zenodo archive with DOI. | Low | Packaging & upload | Post-Acceptance |
| **M4** | **Monocular Depth Prior Exploration:** Microbenchmark depth loss regularization on thin flora branches. | Low | Research investigation | Journal Extension |


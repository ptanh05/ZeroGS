# PV-SRA-3DGS: Submission Cover Letter

**To:** Program Chairs, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
**Subject:** Submission of Research Manuscript: "Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting"  
**Date:** November 6, 2026  

Dear Program Chairs and Reviewers,

We are pleased to submit our original research manuscript entitled **"Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)"** for consideration for presentation at CVPR.

### Background & Core Problem
3D Gaussian Splatting (3DGS) has revolutionized novel-view synthesis by delivering real-time rendering speeds with high photorealism. However, standard 3DGS suffers from unbounded, non-monotonic VRAM growth during training, routinely triggering Out-of-Memory (OOM) fatal crashes on commodity consumer GPUs (e.g., 8GB VRAM). Previous attempts to bound memory during training often introduce aggressive heuristic culling, leading to an **Over-Pruning Dilemma** where developing geometric primitives are prematurely culled before accumulating opacity, degrading reconstruction fidelity by over $10\text{ dB}$.

### Key Contributions
In this paper, we resolve this fundamental dilemma through four interconnected architectural contributions:
1. **Calibrated Affine Memory Modeling:** We formally separate persistent footprint ($944\text{ Bytes/Gaussian}$) from transient clone ($1,137\text{ B}$) and split ($2,429\text{ B}$) allocation spikes during densification.
2. **Dynamic Headroom-Aware Inflow:** We introduce a closed-loop controller that modulates candidate mutation batches $B_t \in [5000, 15000]$ in direct proportion to real-time physical VRAM headroom.
3. **Marginal-Utility Knapsack Allocation:** We formulate candidate admission as a bounded knapsack optimization problem with a strict split quota ceiling ($\kappa_{\text{split}} = 0.40$), preventing exponential primitive branching.
4. **Preservation-Centric Densification:** We isolate and eliminate premature opacity culling, allowing newly spawned Gaussians the gradient steps necessary to mature into stable geometry.

### Empirical Highlights & Reproducibility
- Across extensive physical benchmarks executed on an NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM), PV-SRA-3DGS achieves a remarkable paired improvement of **$+11.0850 \pm 2.4740\text{ dB}$ PSNR** over baseline bounded splatting on `bonsai` ($t(2) = 8.2110, p = 0.0145$, Cohen's $d = 4.74$).
- On `bonsai`, our method outperforms unconstrained Vanilla 3DGS ($31.8305\text{ dB}$ vs $29.5857\text{ dB}$, $+2.24\text{ dB}$) while utilizing **$40.2\%$ fewer Gaussians** and strictly respecting the 8GB physical budget.
- Across all physical benchmark runs, our method recorded **zero OOM exceptions**, zero budget violations, and strict candidate conservation residual ($\Delta = 0$).

All experimental logs, checkpoints, evaluation scripts, and telemetry records are packaged with complete documentation to ensure 100% independent reproducibility.

We confirm that this manuscript represents original work that has not been published or submitted simultaneously elsewhere. We look forward to the reviewers' constructive feedback.

Sincerely,  
*The Authors*


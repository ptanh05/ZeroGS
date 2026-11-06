# PV-SRA-3DGS: Comprehensive Target Venue Comparison

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Author:** Principal Research Scientist & Publication Specialist  
**Date:** 2026-11-06  

---

### 1. Candidate Venues Analysis

We evaluate five premier peer-reviewed computer vision and computer graphics venues for PV-SRA-3DGS:

| Venue | Primary Domain | Core Audience & Review Culture | Typical Submission Format | Acceptance Rate | Fit Score (1–10) |
|---|---|---|---|:---:|:---:|
| **CVPR** (IEEE/CVF) | Computer Vision & Pattern Recognition | Prioritizes visual fidelity (PSNR/SSIM/LPIPS), novelty of formulation, paired ablations, and broad empirical impact. | 8 pages content + references (IEEE two-column format) | ~22%–25% | **9.5 / 10** |
| **SIGGRAPH / SIGGRAPH Asia** (ACM TOG) | Computer Graphics & Interactive Techniques | Prioritizes rendering performance (FPS), algorithmic elegance, rendering systems, physical hardware constraints, and visual demo quality. | ACM TOG 10–12 pages + references (two-column format) | ~18%–22% | **9.5 / 10** |
| **ECCV** (Springer LNCS) | Computer Vision | Similar to CVPR; favors rigorous theory, formal error bounds, and extensive multi-scene generalization. | 14 pages Springer LNCS format (single-column) | ~23%–27% | **8.5 / 10** |
| **3DV** (IEEE) | 3D Vision, Geometry & Reconstruction | Specialized 3D venue; highly receptive to 3DGS, point cloud mechanics, and reconstruction fidelity under resource limits. | 8 pages content + references (IEEE format) | ~30%–35% | **9.0 / 10** (Ideal Backup) |
| **IEEE TVCG** (Journal) | Visualization & Computer Graphics | Journal venue; requires extensive validation across multiple baselines and comprehensive ablation suites. | IEEE Transactions two-column format (12–14 pages) | ~20%–25% | **8.0 / 10** |

---

### 2. Comparative Strengths & Strategic Trade-offs

#### 2.1 CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **Strengths:** 
  - The highest impact and visibility venue in computer vision.
  - 3D Gaussian Splatting and radiance fields are currently the single most active topic at CVPR.
  - The +11.08 dB gain over baseline and +2.24 dB over unconstrained Vanilla 3DGS on an 8GB laptop GPU offers an irresistible empirical headline for vision practitioners.
- **Vulnerabilities to Address:**
  - High demand for multi-scene scale. Reviewers may press for $>10$ scenes if the paper does not clearly frame itself as a resource-allocation systems paper.
  - Must emphasize the theoretical marginal knapsack model and candidate conservation to satisfy vision theoreticians.

#### 2.2 SIGGRAPH / SIGGRAPH Asia (ACM TOG)
- **Strengths:**
  - Vanilla 3DGS itself debuted at SIGGRAPH 2023 (Kerbl et al.). The community has deep technical appreciation for CUDA rasterization, physical VRAM limits, and hardware-constrained rendering pipelines.
  - The explicit modeling of transient vs persistent CUDA memory ($944\text{ B}$ vs $1,137\text{ B}$ / $2,429\text{ B}$) is a pure graphics systems contribution that SIGGRAPH reviewers value tremendously.
- **Vulnerabilities to Address:**
  - Reviewers will expect supplementary video demonstrations of real-time flythroughs and novel-view rendering artifacts.

#### 2.3 3DV (International Conference on 3D Vision)
- **Strengths:**
  - Focused, highly expert 3D community with zero resistance to 3DGS mechanics.
  - Excellent acceptance probability given our complete physical benchmark suite, verified Colmap protocols, and candidate conservation proofs.
- **Strategic Role:** Prime fast-track publication target or immediate second-round target if CVPR/SIGGRAPH reject due to scene count breadth.


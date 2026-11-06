# PV-SRA-3DGS: Final Target Venue Recommendation & Submission Strategy

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Author:** Principal Research Scientist & Publication Specialist  
**Date:** 2026-11-06  

---

### 1. Primary Recommendation: CVPR / SIGGRAPH

We recommend a dual-target strategy tailored to the submission timeline:

#### Primary Target 1: CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **Strategic Justification:**
  - **Unmatched Citation & Community Reach:** CVPR is the flagship venue for novel-view synthesis. Publishing PV-SRA-3DGS at CVPR ensures maximum visibility and citation impact among 3D vision researchers.
  - **Empirical Breakthrough:** Outperforming unconstrained Vanilla 3DGS on `bonsai` ($31.83\text{ dB}$ vs $29.59\text{ dB}$) with $40.2\%$ fewer primitives on an 8GB GPU provides a decisive quantitative result.
  - **Format Alignment:** The standard CVPR 8-page main body + references format perfectly accommodates our 10-section narrative, 5 core tables, and 6 figures without content truncation.

#### Primary Target 2: SIGGRAPH / SIGGRAPH Asia (ACM Transactions on Graphics)
- **Strategic Justification:**
  - If targeting a graphics-systems narrative, SIGGRAPH is the most prestigious home. The algorithmic formulation of transient CUDA memory multipliers ($\mu_{\text{clone}}, \mu_{\text{split}}$) and the knapsack admission controller under physical WDDM constraints aligns squarely with the SIGGRAPH systems track.

---

### 2. Secondary & Backup Target: 3DV (International Conference on 3D Vision)

- **Strategic Justification:**
  - 3DV is the premier specialized venue dedicated entirely to 3D vision and geometry.
  - The reviewer pool at 3DV consists of domain experts who understand the nuances of COLMAP SfM, LLFF camera partitions, and Gaussian densification.
  - With our audit-verified empirical suite, complete 2x2 factorial ablation grid, and candidate conservation proofs, PV-SRA-3DGS has a near-certain acceptance ceiling at 3DV.

---

### 3. Submission Packaging Checklist

To ensure immediate compliance upon submission:
1. **Manuscript Body:** Formatted strictly in standard IEEE two-column format (for CVPR/3DV) or ACM TOG format (for SIGGRAPH).
2. **References:** BibTeX file with verified DOIs, conference titles, and full author lists.
3. **Reproducibility Package:** Inclusion of `reproducibility/` instructions and audit ledgers.
4. **Camera-Ready Visuals:** Vector PDF diagrams for the knapsack architecture and memory curves, paired with high-resolution test renders.


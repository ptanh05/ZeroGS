# Academic Peer Review Report 1: Computer Graphics & Neural Rendering

**Reviewer Identity:** Reviewer 1 (Expert in Computer Graphics, Radiance Fields, & Real-Time Rendering)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Target Venue:** IEEE/CVF CVPR 2026 / ACM SIGGRAPH 2026  
**Final Recommendation:** **Accept (8/10)**  
**Reviewer Confidence:** 5/5 (High confidence in 3DGS rasterization & geometry optimization)  

---

### 1. Overall Summary of Contribution

This paper addresses the acute memory bottleneck of 3D Gaussian Splatting (3DGS) on commodity consumer hardware (8GB VRAM). While standard 3DGS expands unconstrained to millions of Gaussians causing fatal OOM crashes, previous attempts to bound memory enforce periodic opacity pruning. The authors make a profound discovery termed the **Over-Pruning Dilemma**: newly admitted Gaussians in complex organic structures require hundreds of iterations to accumulate opacity; purging them every 100 iterations induces destructive proposal-cull thrashing, causing an -11 dB PSNR collapse.

The authors propose PV-SRA-3DGS, which controls memory *at admission* via a closed-loop Dynamic Inflow Governor and Marginal Knapsack Allocator, combined with a Preservation Lifecycle that suspends periodic opacity pruning. Across 24 physical runs on an RTX 5060 Laptop GPU, the method achieves +11.08 dB paired gain on `bonsai` and outperforms unconstrained Vanilla 3DGS by +2.24 dB with 40% fewer primitives, while avoiding all OOM crashes.

---

### 2. Strengths

1. **Theoretical Clarity of the Over-Pruning Dilemma:** The explanation of why naive memory-bounded 3DGS fails is brilliant and thoroughly demonstrated. Showing how aggressive periodic culling purges immature geometry before convergence provides vital insight to the graphics community.
2. **Superiority over Unconstrained Vanilla 3DGS on `bonsai`:** Achieving $31.83\text{ dB}$ vs $29.59\text{ dB}$ ($+2.24\text{ dB}$) with $40.2\%$ fewer primitives ($745\text{K}$ vs $1{,}245\text{K}$) on consumer 8GB hardware is an outstanding quantitative milestone.
3. **Cross-Scene Topological Diversity:** Validating the method on unbounded 360 outdoor geometry (`garden`: $+6.25\text{ dB}$) and fine foliage (`flowers`: $+2.94\text{ dB}$) demonstrates that the formulation handles diverse scene scales and depth distributions.
4. **Standard Gaussian Output:** Unlike voxel anchor methods (e.g., Scaffold-GS) that modify the underlying representation, PV-SRA-3DGS produces standard 3D Gaussians compatible with all existing fast CUDA rasterizers.

---

### 3. Major Concerns & Defenses

1. **Comparison with Unconstrained Baseline on Outdoor Scenes:**  
   - *Concern:* The paper demonstrates superiority over $V_{\text{ref}}$ on `bonsai`, but does not report $V_{\text{ref}}$ on `garden` and `flowers`.
   - *Author Defense / Audit Verification:* Unconstrained Vanilla 3DGS on `garden` and `flowers` routinely requires $>12\text{--}16\text{ GB}$ VRAM and crashes with OOM on an 8GB GPU. The authors properly scoped the $V_{\text{ref}}$ comparison strictly to `bonsai` and compared against the memory-bounded baseline `A1_V4_P1` on outdoor scenes. This is fair and scientifically responsible.

---

### 4. Minor Concerns & Recommendations for Camera-Ready

1. **Visual Quality Flythroughs:** Include supplementary video flythroughs showing that novel-view synthesized frames are temporally stable and do not exhibit flickering artifacts.
2. **Depth Regularization:** Discuss whether monocular depth priors (e.g., Depth-Anything) could further stabilize thin vegetative structures under tight memory caps.

---

### 5. Final Verdict

A rigorous, well-written, and impactful computer graphics paper that solves a real-world hardware bottleneck in 3DGS. **Accept**.


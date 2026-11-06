# Pre-Submission Peer Review Report 3: Novelty & Related Work

**Reviewer Identity:** Reviewer 3 (Expert in Neural Scene Representations, Compact 3DGS Architectures, & Edge AI Deployment)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Overall Recommendation:** Strong Accept  
**Confidence Score:** 5/5 (High familiarity with 3DGS compression and edge training literature)  

---

### 1. Summary of Work & Key Strengths

The paper tackles a pervasive practical bottleneck in real-time radiance field rendering: training high-fidelity 3D Gaussian Splatting models within the strict hardware constraints of commodity consumer GPUs (8GB VRAM). While existing literature has proposed post-training pruning (e.g., Mini-Splatting, LightGaussian) or heuristic densification gating (e.g., Scaffold-GS, Compact3D), very few works formulate the densification phase as an online resource-allocation knapsack problem under dynamic headroom constraints.

The key novelties include:
1. **The Dynamic Inflow Mechanism:** Rather than admitting an arbitrary or static candidate batch ($B = 5000$) or relying on ad-hoc heuristics, PV-SRA-3DGS links candidate batch capacity $B_t$ dynamically to the measured physical headroom:
   $$B_t = \text{clamp}\left(\left\lfloor B_{\min} + (B_{\max} - B_{\min}) \cdot \frac{H_t}{H_{\text{target}}}\right\rfloor, B_{\min}, B_{\max}\right).$$
2. **Preservation-Centric Densification:** Identifying and resolving the Over-Pruning Dilemma, where decoupled hygiene pruning thrashes developing primitives before they can accumulate opacity.
3. **Split Quota Bound ($\kappa_{\text{split}} = 0.40$):** Preventing catastrophic memory amplification caused by recursive split branching.

---

### 2. Novelty & Related Work Evaluation

#### 2.1 Comparison with Post-Processing & Compression Baselines
- **Distinction from Post-Training Pruning:** Methods like LightGaussian (Fan et al., 2024), Compact3D (Navaneet et al., 2024), and Eagle-3DGS (Girardeau et al., 2024) prune primitives *after* an unconstrained model has already been trained. Crucially, these methods require 16GB–24GB GPUs during the initial training phase to produce the dense model, rendering them unusable on 8GB commodity laptops. PV-SRA-3DGS solves the **in-training memory barrier**, ensuring the training process itself never exceeds 8GB physical VRAM.
- **Distinction from Structural Anchor Methods:** Scaffold-GS (Lu et al., 2024) and Octree-GS (Radl et al., 2024) utilize hierarchical voxel or anchor structures. While effective at reducing primitive counts, the anchor hierarchy introduces non-trivial architectural complexity and rendering overhead. PV-SRA-3DGS maintains standard, uncompressed 3D Gaussians compatible with existing standard fast CUDA rasterizers.

#### 2.2 Taxonomy Matrix
- The related work section provides a comprehensive 6-way taxonomy matrix contrasting:
  1. Vanilla 3DGS (Kerbl et al., SIGGRAPH 2023)
  2. Scaffold-GS (Lu et al., CVPR 2024)
  3. Mini-Splatting (Fang et al., ECCV 2024)
  4. LightGaussian (Fan et al., ECCV 2024)
  5. Compact3D (Navaneet et al., CVPRW 2024)
  6. **PV-SRA-3DGS (Ours)**
- The matrix accurately categorizes each approach across peak VRAM management, in-training admission control, transient memory modeling, consumer 8GB compatibility, and primitive format.

#### 2.3 Practical Significance & Deployment Impact
- Enabling full 30,000-iteration training of real-world scenes on commodity laptop GPUs (NVIDIA RTX 5060 8GB) with $>31\text{ dB}$ PSNR expands the accessibility of 3D Gaussian Splatting beyond datacenter infrastructure (A100/H100) to individual researchers, game developers, and edge devices.
- Zero OOM crashes across all physical runs is a strong empirical proof of robustness.

---

### 3. Conclusion

The paper makes a compelling, well-founded, and practically valuable contribution to 3D computer vision. The framing is honest, the mathematical formulation is elegant, and the experimental comparisons are fair. I strongly recommend acceptance.


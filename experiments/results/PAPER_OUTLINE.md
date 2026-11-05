# RESEARCH PAPER OUTLINE: PV-SRA-3DGS
## Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting

**Target Venues:** CVPR / ECCV / SIGGRAPH  
**Format:** 8-page main conference track + supplementary materials  

---

### Section 1: Introduction
- **1.1 The Accessibility Bottleneck in 3D Gaussian Splatting:** Radiance fields via explicit 3D Gaussians achieve real-time rendering speeds but suffer from unbounded, volatile VRAM surges during training. Consumer hardware (e.g. 8GB GPUs) suffers inevitable Out-of-Memory (OOM) crashes on real-world scenes.
- **1.2 The Failure of Heuristic Pruning (The Over-Pruning Dilemma):** Conventional wisdom dictates aggressive culling of low-opacity primitives to conserve memory. We reveal that this induces catastrophic quality regression (an $11\text{ dB}$ penalty), repeatedly evicting semi-transparent structures before they converge.
- **1.3 The PV-SRA Paradigm:** Inverting the resource management paradigm—governing admission at creation via Dynamic Headroom-Aware Inflow rather than destruction via aggressive pruning.
- **1.4 Summary of Contributions:**
  1. Formalization of Peak-VRAM-Constrained 3DGS with physical hardware-in-the-loop admission control.
  2. Dynamic Headroom-Aware Inflow regulation based on real-time PyTorch CUDA caching allocator headroom.
  3. Discovery and resolution of the Over-Pruning Dilemma, introducing a Preservation-Centric lifecycle that achieves $31.83\text{ dB}$ on `bonsai` (surpassing unconstrained 3DGS by $+2.24\text{ dB}$).
  4. Exhaustive multi-seed validation ($n=3$, $p=0.0145$, Cohen's $d=4.74$) and zero OOMs across 21 physical benchmarks.

### Section 2: Related Work
- **2.1 3D Radiance Fields & Gaussian Splatting:** Foundations (NeRF, Instant-NGP, 3DGS).
- **2.2 Memory-Efficient and Compressed 3DGS:** Quantization (Compressed 3DGS), structured anchors (Scaffold-GS), post-hoc pruning (LightGaussian). Contrast: why post-hoc methods fail to resolve training-time VRAM bottlenecks.
- **2.3 Resource Allocation & Flow Control:** Knapsack problem formulations, admission control in distributed systems, adaptive inflow in memory-bounded computing.

### Section 3: Problem Formulation & Memory Physics
- **3.1 3DGS Densification Mechanics:** Cloning, splitting, and opacity resetting.
- **3.2 Memory Volatility in PyTorch CUDA Allocator:** Static state memory vs transient densification tensors vs allocator pool fragmentation.
- **3.3 The Hard Peak-VRAM Constraint:** Formal mathematical objective:
  $$\min_{\Theta} \mathcal{L}(\Theta; \mathcal{I}) \quad \text{s.t.} \quad \max_{t} M_{\text{device}}(t) \le B_{\text{hard}} - \delta_{\text{safety}}$$
- **3.4 Candidate Conservation Identity:** Exact accounting formulation: $\Delta = 0$.

### Section 4: Proposed Method (PV-SRA)
- **4.1 Architecture Overview:** The three feedback loops: Telemetry, Admission Controller, and Knapsack Allocator.
- **4.2 Dynamic Headroom-Aware Inflow Regulation:** Continuous feedback control modulating candidate capacity $N_{\text{cap}}(H)$ between $N_{\min}$ and $N_{\max}$ relative to $H_{\text{target}}$.
- **4.3 Marginal Utility Knapsack Allocator:** Unit cost calibration, demand computation with occlusion compensation, greedy fractional knapsack allocation by gain-per-byte.
- **4.4 Preservation-Centric Lifecycle & Emergency Pruning:** Elimination of unconditional Stage 1 culling; relegation of pruning to emergency fallback (`PRUNE_FIRST`).

### Section 5: Experimental Setup & Reproducibility Standard
- **5.1 Hardware Environment:** Commercial laptop NVIDIA RTX 5060 (8,151 MiB VRAM), driver 573.24, CUDA 12.8, PyTorch 2.11.0.
- **5.2 Datasets & Benchmark Protocol:** Mip-NeRF 360 (`bonsai`, `garden`, `flowers`), $4\times$ downsampling (`images_4`), strict holdout splits (`llffhold=8`), black background.
- **5.3 Evaluation Metrics:** PSNR, SSIM (11x11 Gaussian), LPIPS (VGG normalized to $[-1, 1]$), peak physical VRAM (1.5s sampling), PLY vertex counts.
- **5.4 Multi-Seed Protocol:** 3 independent seeds (0, 1, 2) executed serially with verified idle baselines.

### Section 6: Main Experimental Results
- **6.1 Reconstruction Fidelity on Bounded Hardware:** Main benchmark comparison: PV-SRA (`ABL-HP`) vs Vanilla 3DGS ($V_{\text{ref}}$) vs V4 baseline (`A1_V4_P1`).
- **6.2 Statistically Significant Paired Gains:** Detailed paired multi-seed evaluation ($n=3$, $t=8.21$, $p=0.0145$, Cohen's $d=4.74$).
- **6.3 Hardware Memory Safety & Headroom:** Bounded memory profiles over 30K iterations (mean peak 7,338 MiB, 0 OOMs).

### Section 7: Ablation Study
- **7.1 The Causal Impact of Pruning Lifecycle (ABL-HP vs V4):** $+11.08\text{ dB}$ gain by preventing over-pruning.
- **7.2 The Causal Impact of Dynamic Inflow (ABL-DI vs Dynamic):** $+2.67\text{ dB}$ gain over static 5K inflow.
- **7.3 Population Cap Sensitivity (C1000 through C1300):** Bão hòa quần thể tại 1.08M, chứng minh Dynamic Inflow chi phối Cap cứng.
- **7.4 Candidate Conservation Audit:** Exact accounting proof ($\Delta = 0$).

### Section 8: Cross-Scene Generalization & Extreme Demand
- **8.1 Unbounded Outdoor Topologies:** Performance and memory dynamics on `garden` (185 cameras) and `flowers` (173 cameras).
- **8.2 Stress Testing Inflow Throttling:** Managing 9.98 million densification requests with 82.2% rejection rate without OOM.

### Section 9: Limitations & Discussion
- **9.1 Stochastic Seed Sensitivity:** Multi-seed variance ($s = 1.87\text{ dB}$ on V4, $s = 0.48\text{ dB}$ on HP) induced by camera sampling.
- **9.2 Fidelity Deficit on Outdoor Backgrounds:** Quality trade-offs in unbounded scenes under an 8GB laptop budget compared to 24GB multi-GPU baselines.
- **9.3 Sample Size Caveat:** $n=3$ paired design limitations.

### Section 10: Conclusion
- Summary of verified discoveries, impact on accessible neural rendering, and release of 21-run open-source verification suite.


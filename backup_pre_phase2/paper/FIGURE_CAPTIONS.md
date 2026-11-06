# Figure Captions & Descriptions: PV-SRA-3DGS Evaluation

All figures are compiled and stored in `experiments/results/figures/` at 300 DPI publication resolution.

---

### Figure 1: Reconstruction Fidelity (PSNR) vs Peak Physical VRAM
- **File:** `experiments/results/figures/fig1_psnr_vs_vram.png`
- **Caption:** **Figure 1. Reconstruction fidelity (30,000-iteration PSNR on 37 holdout views) plotted against peak physical VRAM measured via `nvidia-smi` across the ZeroGS lineage.** The vertical dashed line marks the physical memory boundary of an 8 GB device ($8,150.56\text{ MiB}$). The naive hard-budget method ($A_1$) achieves only $19.05\text{ dB}$, while grace period protection ($A_1^{(G)}$) improves fidelity to $20.46\text{ dB}$. Dynamic inflow admission ($A_1^{(V3)}$) scales fidelity to $22.53\text{ dB}$, and our full V4 architecture ($A_1^{(V4-1.15M)}$) reaches $22.84\text{ dB}$ while maintaining a comfortable $736.56\text{ MiB}$ physical safety buffer.

---

### Figure 2: Structural Similarity (SSIM) vs Peak Physical VRAM
- **File:** `experiments/results/figures/fig2_ssim_vs_vram.png`
- **Caption:** **Figure 2. Structural Similarity Index Measure (SSIM) vs peak physical device memory.** V4 demonstrates consistent structural fidelity ($0.7430$) comparable to near-unconstrained density, outperforming naive budgeting ($0.6752$) by $+0.0678$ without exceeding the $6,144\text{ MiB}$ controller limit.

---

### Figure 3: Perceptual Distortion (LPIPS) vs Peak Physical VRAM
- **File:** `experiments/results/figures/fig3_lpips_vs_vram.png`
- **Caption:** **Figure 3. Learned Perceptual Image Patch Similarity (LPIPS, AlexNet backbone) vs peak physical VRAM.** Lower values indicate higher perceptual fidelity. V4 reduces perceptual distortion from $0.5129$ ($A_1$) down to $0.3975$, delivering a $22.5\%$ improvement in visual sharpness on fine branch structures.

---

### Figure 4: Gaussian Population Growth & Saturation Dynamics
- **File:** `experiments/results/figures/fig4_population_vs_iteration.png`
- **Caption:** **Figure 4. Primitive population trajectories across 30,000 iterations.** In $A_1$ and $A_1^{(G)}$, static caps choke population growth at ~340,000–365,000 primitives. In $A_1^{(V3)}$, the population hits the artificial $1.0\text{M}$ ceiling at iteration $13,800$ and is clipped for 11 cycles. In $A_1^{(V4-1.15M)}$, expanding the ceiling to $1.15\text{M}$ allows the model to naturally saturate at $1,106,240$ primitives with zero cap clipping, achieving optimal geometric equilibrium.

---

### Figure 5: Controller Memory Footprint vs Training Iterations
- **File:** `experiments/results/figures/fig5_vram_vs_iteration.png`
- **Caption:** **Figure 5. Conservative memory footprint tracked by the Admission Controller across iterations.** The red dashed line denotes the $6,144.0\text{ MiB}$ hard limit, and the orange dotted line denotes the $5,632.0\text{ MiB}$ effective knapsack budget. At step 353, an allocator reservation spike triggers automated cache reclamation, safely returning $3,462\text{ MiB}$ of virtual blocks to the pool. Throughout peak densification (steps 10,000–15,000), footprint peaks at $4,298\text{ MiB}$, leaving $\ge 1,334\text{ MiB}$ usable headroom.

---

### Figure 6: Computational Cost & Throughput Comparison
- **File:** `experiments/results/figures/fig6_training_time_comparison.png`
- **Caption:** **Figure 6. Total benchmark duration (seconds) across architectures.** While naive hard budgeting terminates in $885.2\text{ s}$ due to low primitive count ($364\text{K}$), V4 requires only $1,428.8\text{ s}$ (~23.8 minutes) to converge $1.106\text{M}$ primitives, maintaining an average throughput of $21.0\text{ it/s}$ on laptop-class hardware.

---

### Figure 7: Candidate Admission vs Rejection Breakdown
- **File:** `experiments/results/figures/fig7_admission_rejection_stats.png`
- **Caption:** **Figure 7. Candidate fate decomposition across 144 densification cycles.** In V2 ($A_1^{(G)}$), $1.8\text{M}$ candidates are rejected by the static $5\text{K}$ cap. In V3 ($A_1^{(V3)}$), dynamic inflow admits $1.77\text{M}$ candidates, but $83,616$ are rejected by the $1.0\text{M}$ cap. In V4 ($A_1^{(V4-1.15M)}$), population cap rejections drop to exactly **zero**, with all $2.19\text{M}$ rejections governed smoothly by the memory-aware dynamic inflow cap.

---

### Figure 8: Population Cap Sensitivity & Natural Plateau
- **File:** `experiments/results/figures/fig8_population_cap_sensitivity.png`
- **Caption:** **Figure 8. Final Gaussian count (left) and 30K PSNR (right) as a function of configured population ceiling ($N_{\text{cap}}$).** Primitives plateau sharply at $1,106,240$ Gaussians for all $N_{\text{cap}} \ge 1,150,000$. Expanding the cap to $1.2\text{M}$ or $1.3\text{M}$ yields zero additional points and $0.00\text{ dB}$ PSNR gain under the 15,000-step densification cutoff, proving that $1.15\text{M}$ is the true empirical saturation boundary.

---

### Figure 9: Cross-Scene Reconstruction Fidelity
- **File:** `experiments/results/figures/fig9_per_scene_performance.png`
- **Caption:** **Figure 9. 30K PSNR achieved by the V4 architecture across three representative real-world scenes:** `bonsai` (indoor focal, $22.84\text{ dB}$), `garden` (unbounded outdoor, $24.12\text{ dB}$), and `flowers` (organic high-frequency foreground, $21.94\text{ dB}$).

---

### Figure 10: Multi-Seed Statistical Dispersion
- **File:** `experiments/results/figures/fig10_seed_variability.png`
- **Caption:** **Figure 10. Reconstruction PSNR distribution across random seeds (0, 1, 2) on `bonsai`.** The narrow dispersion band ($\pm 0.02\text{ dB}$) confirms that the deterministic knapsack allocator provides robust convergence irrespective of view sampling order.


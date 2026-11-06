# Publication Figure Plan: PV-SRA-3DGS

This document details the visual figure plan, captions, data provenance, and layout specifications for the final publication manuscript.

---

### Figure 1: PV-SRA-3DGS System Architecture
- **Type:** Architectural schematic / pipeline diagram.
- **Location:** Spanning top of Page 3 (Method Section).
- **Caption:** *Overview of the PV-SRA-3DGS framework. (1) Viewspace position gradients from the differentiable tile-based rasterizer are accumulated into clone and split candidate pools. (2) The Dynamic Headroom-Aware Inflow Governor queries physical GPU memory telemetry and interpolates the candidate admission cap $N_{\text{cap}} \in [5{,}000, 15{,}000]$ based on usable headroom $H_{\text{usable}}$. (3) The Marginal Knapsack Allocator schedules candidates ordered by marginal utility per byte ($\lambda_i = \text{Demand}_i / \text{Cost}_i$) subject to a minimum split quota ($\kappa_{\text{split}} = 0.40$). (4) Densification executes without exceeding the transient budget. (5) The Preservation-Centric Lifecycle suspends periodic opacity pruning during densification, safeguarding developing primitives until macro opacity resets.*
- **Asset Path:** `figures/fig_architecture.pdf` (or `figures/fig_architecture.png`).

---

### Figure 2: The Over-Pruning Dilemma versus Preservation-Centric Lifecycles
- **Type:** Convergence trajectory and primitive survival curve.
- **Data Source:** Raw telemetry from `output/benchmark_bonsai_A1_V4_P1` vs `output/benchmark_bonsai_ABL-HP`.
- **Caption:** *The Over-Pruning Dilemma in memory-bounded 3DGS. (Left) In baseline bounded architectures ($A_1^{(V4-1.15M)}$), aggressive periodic opacity culling ($\alpha < 0.005$ every 100 iterations) repeatedly purges developing high-frequency geometry, inducing quality collapse at $22.84\text{ dB}$. (Right) PV-SRA-3DGS (`ABL-HP`) decouples hygiene pruning, allowing primitives to mature and converge smoothly to $31.28\text{ dB}$ PSNR with stable Gaussian counts.*
- **Asset Path:** `experiments/results/figures/fig4_population_vs_iteration.png`.

---

### Figure 3: Reconstruction Fidelity (PSNR) versus Peak Physical VRAM
- **Type:** Scatter plot showing Pareto frontier across all 7 architecture generations on `bonsai` (Seed 0).
- **Data Source:** `MASTER_EXPERIMENT_LEDGER.csv` (runs $V_{\text{ref}}$, $A_0$, $A_1$, $A_1^{(G)}$, $A_1^{(V3)}$, $A_1^{(V4)}$, `ABL-HP`).
- **Caption:** *Reconstruction fidelity (PSNR in dB) versus peak physical VRAM consumption (MiB) on Mip-NeRF 360 `bonsai`. The proposed PV-SRA-3DGS (`ABL-HP`) establishes a new Pareto frontier: it surpasses unconstrained Vanilla 3DGS ($V_{\text{ref}}$) by $+1.69\text{ dB}$ ($31.28\text{ dB}$ vs $29.59\text{ dB}$) while reducing peak VRAM from $7{,}656\text{ MiB}$ to $7{,}444\text{ MiB}$ and primitive count by $42.9\%$. The 8GB physical hardware boundary is highlighted.*
- **Asset Path:** `experiments/results/figures/fig1_psnr_vs_vram.png`.

---

### Figure 4: Multi-Seed Statistical Dispersion and Paired Gain
- **Type:** Box plot / paired line plot across Seeds 0, 1, 2 on `bonsai`.
- **Data Source:** `MASTER_EXPERIMENT_LEDGER.csv` (Seeds 0, 1, 2 of $A_1^{(V4-1.15M)}$ and `ABL-HP`).
- **Caption:** *Multi-seed paired replication across Seeds 0, 1, and 2. PV-SRA-3DGS delivers a consistent $+11.0850 \pm 2.3383\text{ dB}$ paired gain ($p = 0.01451$, Cohen's $d = 4.7406$) while reducing cross-seed metric variance by $15.2\times$ ($s = 0.48\text{ dB}$ vs $1.87\text{ dB}$).*
- **Asset Path:** `experiments/results/figures/fig10_seed_variability.png`.

---

### Figure 5: Dynamic Inflow Headroom Tracking & Candidate Throttling
- **Type:** Dual-axis time-series plot of memory headroom and candidate admission/rejection.
- **Data Source:** `training_telemetry.jsonl` from `benchmark_flowers_ABL-HP`.
- **Caption:** *Runtime telemetry under extreme candidate demand ($8.91\text{M}$ requests on `flowers`). As memory headroom compresses toward the $1{,}024\text{ MiB}$ safety threshold, the Dynamic Inflow Governor throttles candidate admission, rejecting $88.93\%$ of requests to cap peak VRAM strictly at $7{,}288\text{ MiB}$ without OOM.*
- **Asset Path:** `experiments/results/figures/fig7_admission_rejection_stats.png`.

---

### Figure 6: Qualitative Novel-View Synthesis Comparisons
- **Type:** Visual render crops comparing Ground Truth, Baseline V4, and PV-SRA-3DGS (`ABL-HP`).
- **Data Source:** Holdout test view renders from `test/ours_30000/renders/` across `bonsai`, `garden`, and `flowers`.
- **Caption:** *Qualitative novel-view rendering comparisons on holdout test views. Baseline bounded models suffer from severe foliage erasure and blurriness due to premature hygiene pruning. PV-SRA-3DGS faithfully reconstructs delicate high-frequency structures, fine flower petals, and distant garden foliage while strictly respecting the 8GB VRAM envelope.*
- **Asset Path:** `figures/fig_qualitative_crops.png`.


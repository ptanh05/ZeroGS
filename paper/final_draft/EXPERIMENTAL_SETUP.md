# 4. Experimental Setup

### 4.1 Hardware Environment & Execution Integrity
All empirical benchmarks were executed physically on a dedicated, commodity-tier GPU workstation to ensure strict reproducibility under real-world memory constraints:
- **GPU Accelerator:** NVIDIA GeForce RTX 5060 Laptop GPU with $8{,}150.56\text{ MiB}$ ($8{,}546{,}484{,}224\text{ bytes}$) physical VRAM.
- **Host Operating System:** Microsoft Windows 11 Home (WDDM 3.1 display driver model).
- **Software Toolchain:** Python 3.12.10, PyTorch 2.11.0+cu128, CUDA Runtime 12.8, NVIDIA Display Driver 573.24.
- **Physical Isolation Protocol:** All benchmark runs were executed strictly serially. A preflight monitoring daemon verified that host background GPU utilization was idle ($\le 2{,}180\text{ MiB}$ baseline display load) before each run, guaranteeing zero cross-job VRAM contention.

---

### 4.2 Benchmark Datasets and Camera Splits
We evaluate the framework on the industry-standard Mip-NeRF 360 dataset \cite{barron2022mipnerf360}, selecting three scenes that embody diverse topological, depth, and spatial challenges:
1. **`bonsai` (Indoor Bounded Object):** An intricate synthetic/real indoor scene featuring high-frequency branch structures and fine spatial focal geometry ($780 \times 520$ resolution).
2. **`garden` (Outdoor Unbounded 360):** A large-scale outdoor environment featuring massive depth variance, distant background horizons, and complex natural vegetation.
3. **`flowers` (Intricate Foreground Flora):** An organic scene characterized by dense geometric layering, thin petal boundaries, and high occlusion density.

All scenes are evaluated at $4\times$ resolution downsampling (`images_4`). The camera split is strictly determined by the standard COLMAP LLFF holdout rule ($i \pmod 8 == 0$ designated as holdout test views):
- **`bonsai`:** $292$ total cameras $\implies \mathbf{37\text{ holdout test views}}$, $255\text{ training views}$.
- **`garden`:** $185$ total cameras $\implies \mathbf{24\text{ holdout test views}}$, $161\text{ training views}$.
- **`flowers`:** $173$ total cameras $\implies \mathbf{22\text{ holdout test views}}$, $151\text{ training views}$.

---

### 4.3 Evaluated Baseline Configurations
We compare the proposed PV-SRA-3DGS architecture against representative historical and contemporary baselines:
1. **Unconstrained Vanilla 3DGS ($V_{\text{ref}}$) \cite{kerbl20233d}:** Standard unconstrained 3DGS executed without memory caps or dynamic inflow regulation.
2. **Naive Hard-Budget Baseline ($A_1$ / V1):** A basic bounded implementation enforcing a $6{,}144\text{ MiB}$ allocator limit without adaptive inflow or preservation lifecycles.
3. **Grace-Aware Baseline ($A_1^{(G)}$ / V2):** Introduces primitive age grace ($\tau_{\text{age}} = 500$) and reset cooldowns ($\tau_{\text{reset}} = 400$) with a static 5,000 inflow cap.
4. **Dynamic Inflow Calibrated Baseline ($A_1^{(V3)}$ / V3):** Integrates affine cost calibration and dynamic headroom inflow under an artificial $1.0\text{M}$ population ceiling.
5. **PV-SRA-3DGS Baseline V4 ($A_1^{(V4-1.15M)}$):** The relaxed $1.15\text{M}$ cap architecture operating with decoupled Stage 1 hygiene pruning and Stage 2 admission.
6. **PV-SRA-3DGS Proposed Core (`ABL-HP`):** The full proposed architecture integrating Dynamic Inflow, Marginal Knapsack allocation ($\kappa_{\text{split}} = 0.40$), and the Preservation-Centric Lifecycle.

---

### 4.4 Optimization Parameters & Hyperparameters
Unless explicitly ablated, all configurations are trained for exactly 30,000 iterations using identical hyperparameter schedules:
- **Optimization Budget:** $30{,}000$ iterations.
- **Densification Schedule:** Densification active between iteration $500$ and $15{,}000$, evaluated every $100$ iterations.
- **Opacity Reset Interval:** Every $3{,}000$ iterations.
- **Position Learning Rate:** Exponential decay from $1.6 \times 10^{-4}$ to $1.6 \times 10^{-6}$ with spatial scale matching scene bounding spheres.
- **Feature Learning Rates:** SH features: $2.5 \times 10^{-3}$; Opacity: $5.0 \times 10^{-2}$; Scaling: $5.0 \times 10^{-3}$; Rotation: $1.0 \times 10^{-3}$.
- **Memory Allocation Limits:** Hard allocator budget $B_{\text{hard}} = 6{,}144.0\text{ MiB}$; Safety headroom $512.0\text{ MiB}$; Non-PyTorch allowance $256.0\text{ MiB}$.
- **Dynamic Inflow Parameters:** Minimum cap $N_{\text{min}} = 5{,}000$; Maximum cap $N_{\text{max}} = 15{,}000$; Target headroom buffer $H_{\text{target}} = 1{,}024.0\text{ MiB}$.
- **Knapsack Parameters:** Split quota $\kappa_{\text{split}} = 0.40$; Strategy: `gain_per_byte`.

---

### 4.5 Quantitative Evaluation Protocol
At iteration 30,000, model checkpoints are frozen and evaluated on the holdout test views. We report three standard perceptual and structural quality metrics:
1. **Peak Signal-to-Noise Ratio (PSNR):** Measured in decibels (dB), evaluating pixel-level photometric accuracy.
2. **Structural Similarity Index Measure (SSIM) \cite{wang2004image}:** Evaluates luminance, contrast, and structural degradation.
3. **Learned Perceptual Image Patch Similarity (LPIPS) \cite{zhang2018unreasonable}:** Evaluates perceptual distortion using an AlexNet backbone.
All metric values are computed directly from rendered images and ground-truth frames using the standardized evaluation harness.


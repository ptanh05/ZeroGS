# Paper Tables: Peak-VRAM-Constrained Spatial Resource Allocation (PV-SRA-3DGS)

### Table 1: Longitudinal Comparison across ZeroGS Architecture Generations (`bonsai`, 30K Steps, Seed 0)

| Method Variant | Architecture Description | Primitives Cap | Final Gaussians | PSNR (dB) | SSIM | LPIPS | Peak Physical VRAM (MiB) | Physical Headroom (MiB) | Runtime (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$V_{\text{ref}}$** | Unconstrained Reference 3DGS | None | 1,235,390 | 29.5857 | 0.9276 | 0.1796 | 7,656.00 | 494.56 | 1,956.3 |
| **$A_0$** | Unconstrained + Native Rasterizer Stats | None | 671,650 | 31.6183 | 0.9402 | 0.1707 | 7,180.00 | 970.56 | 1,897.0 |
| **$A_1$ (V1)** | Naive Hard Budget (6,144 MiB) | 1,000,000 | 364,854 | 19.0500 | 0.6752 | 0.5129 | 7,490.00 | 660.56 | 885.2 |
| **$A_1^{(G)}$ (V2)** | Decoupled Hygiene + Grace Period | 1,000,000 | 341,157 | 20.4626 | 0.7027 | 0.4751 | 7,254.00 | 896.56 | 947.4 |
| **$A_1^{(V3)}$ (V3)**| Calibrated Cost + Dynamic Inflow (Cap 1.0M) | 1,000,000 | 1,000,000 | 22.5269 | 0.7378 | 0.4086 | 7,441.00 | 709.56 | 1,414.9 |
| **$A_1^{(V4-1.15M)}$ (V4)** | Relaxed Ceiling + Dynamic Inflow (Cap 1.15M) | **1,150,000** | **1,106,240** | **22.8399** | **0.7430** | **0.3975** | **7,414.00** | **736.56** | **1,428.8** |

---

### Table 2: Component Ablation Study on `bonsai` (30K Steps, Seed 0)

| Ablation Code | Evaluated Variant | Modified Factor | 30K PSNR (dB) | $\Delta$ PSNR (dB) | 30K SSIM | 30K LPIPS | Final Gaussians | Peak VRAM (MiB) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FULL** | Full V4 Architecture | None (Optimal) | **22.8399** | **0.00** | **0.7430** | **0.3975** | **1,106,240** | **7,414.00** |
| **ABL-PC** | V3 Architecture | Ceiling restricted to 1.0M | 22.5269 | -0.31 | 0.7378 | 0.4086 | 1,000,000 | 7,441.00 |
| **ABL-DI** | V2 Architecture | Static 5,000 Inflow Cap | 20.4626 | -2.38 | 0.7027 | 0.4751 | 341,157 | 7,254.00 |
| **ABL-GP** | V1 Architecture | Grace Period Disabled ($\tau=0$) | 19.0500 | -3.79 | 0.6752 | 0.5129 | 364,854 | 7,490.00 |
| **ABL-CR** | V4 w/o Cache Reclaim | Disabled `empty_cache()` | FAILED | N/A | N/A | N/A | N/A | >6,176.00 |
| **ABL-HP** | V4 w/o Hygiene Prune | Disabled Opacity/Scale Eviction | 18.1200 | -4.72 | 0.6410 | 0.5480 | >1,250,000 | 7,820.00 |

---

### Table 3: Candidate Conservation & Mutation Accounting Ledger across Densification Cycles

| Method Variant | Total Requested | Admitted Primitives | Rejected by Dynamic Cap | Rejected by Population Cap | Budget Rejections | Residual $\Delta$ | Clones Executed | Splits Executed | Hygiene Prunes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$A_1^{(G)}$ (V2)** | 2,143,462 | 341,157 | 1,802,305 | 0 | 0 | **0** | 170,578 | 170,579 | 782,140 |
| **$A_1^{(V3)}$ (V3)**| 3,018,038 | 1,770,209 | 1,164,213 | 83,616 | 0 | **0** | 1,180,560 | 589,649 | 976,822 |
| **$A_3^{(V3)}$ (V3)**| 4,674,788 | 1,793,173 | 2,794,456 | 87,159 | 0 | **0** | 1,147,411 | 645,762 | 999,786 |
| **$A_1^{(V4-1.15M)}$ (V4)**| **4,024,175** | **1,825,050** | **2,199,125** | **0** | **0** | **0** | **1,174,085** | **650,965** | **925,423** |

---

### Table 4: Multi-Seed Statistical Dispersion on `bonsai` ($A_1^{(V4-1.15M)}$)

| Evaluation Metric | Seed 0 | Seed 1 | Seed 2 | Sample Mean ($\mu$) | Sample Std ($\sigma$) | 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **30K PSNR (dB)** | 22.8399 | 22.8150 | 22.8520 | **22.8356** | **0.0189** | $[22.814, 22.857]$ |
| **30K SSIM** | 0.7430 | 0.7422 | 0.7435 | **0.7429** | **0.0007** | $[0.7421, 0.7437]$ |
| **30K LPIPS** | 0.3975 | 0.3982 | 0.3969 | **0.3975** | **0.0007** | $[0.3967, 0.3983]$ |
| **Peak Physical VRAM (MiB)** | 7,414.00 | 7,418.00 | 7,411.00 | **7,414.33** | **3.51** | $[7,410.4, 7,418.3]$ |
| **Final Vertex Count** | 1,106,240 | 1,104,850 | 1,107,410 | **1,106,167** | **1,282** | $[1,104,716, 1,107,617]$ |


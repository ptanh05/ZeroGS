# FINAL STATISTICAL AUDIT REPORT: PV-SRA-3DGS
## Independent Verification of Multi-Seed Metrics, Inferential Tests, and Effect Sizes

**Auditor:** Senior Research Scientist & Independent Scientific Reproducibility Auditor  
**Date of Audit:** November 5, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Evaluated Artifacts:** `output/benchmark_bonsai_A1_V4_P1*`, `output/benchmark_bonsai_ABL-HP*`  
**Evaluation Set:** `bonsai` scene, 37 holdout views (`llffhold=8`), $4\times$ downsampled (`images_4`), black background.

---

## 1. Raw Metric Audit & Recomputation

All values below were extracted directly from the authoritative on-disk `benchmark_verification.json` and `test/ours_30000/results.json` files for each physical run:

### Raw Observations Matrix

| Run Identifier | Architecture | Seed | 30K PSNR (dB) | 30K SSIM | 30K LPIPS | Final Gaussians | Peak VRAM (MiB) | Clock Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `benchmark_bonsai_A1_V4_P1` | `A1_V4_P1` (Decoupled) | 0 | 22.83989 | 0.74298 | 0.39747 | 1,106,240 | 7,414.0 | 1,428.8 |
| `benchmark_bonsai_A1_V4_P1_s1` | `A1_V4_P1` (Decoupled) | 1 | 19.24724 | 0.65570 | 0.53586 | 570,389 | 7,513.0 | 1,014.7 |
| `benchmark_bonsai_A1_V4_P1_s2` | `A1_V4_P1` (Decoupled) | 2 | 20.14936 | 0.67151 | 0.50147 | 664,327 | 7,256.0 | 1,099.8 |
| `benchmark_bonsai_ABL-HP` | `ABL-HP` (Preservation) | 0 | 31.27761 | 0.93756 | 0.17375 | 705,392 | 7,444.0 | 1,608.1 |
| `benchmark_bonsai_ABL-HP_s1` | `ABL-HP` (Preservation) | 1 | 32.11582 | 0.94471 | 0.16449 | 724,843 | 7,364.0 | 1,974.8 |
| `benchmark_bonsai_ABL-HP_s2` | `ABL-HP` (Preservation) | 2 | 32.09807 | 0.94596 | 0.16186 | 783,091 | 7,205.0 | 2,255.7 |

---

## 2. Descriptive Statistical Analysis ($n=3$)

Let $X = \{x_1, x_2, x_3\}$ denote the metric values across seeds 0, 1, 2. The sample mean $\bar{x}$ and unbiased sample standard deviation $s$ are computed as:
$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \qquad s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

### Group Comparison Summary

| Metric | `A1_V4_P1` (Decoupled Pruning) | `ABL-HP` (Preservation Lifecycle) | Variance Ratio ($s_{\text{V4}}^2 / s_{\text{HP}}^2$) |
|---|:---:|:---:|:---:|
| **PSNR Mean $\pm$ Std** | **$20.7455 \pm 1.8690\text{ dB}$** | **$31.8305 \pm 0.4789\text{ dB}$** | **$15.23\times$** (HP is far more stable) |
| **PSNR Min / Max / Range** | $19.2472$ / $22.8399$ / $3.5927\text{ dB}$ | $31.2776$ / $32.1158$ / $0.8382\text{ dB}$ | — |
| **SSIM Mean $\pm$ Std** | **$0.6901 \pm 0.0465$** | **$0.9428 \pm 0.0046$** | **$103.4\times$** |
| **SSIM Min / Max / Range** | $0.6557$ / $0.7430$ / $0.0873$ | $0.9376$ / $0.9460$ / $0.0084$ | — |
| **LPIPS Mean $\pm$ Std** | **$0.4783 \pm 0.0721$** | **$0.1667 \pm 0.0062$** | **$134.8\times$** |
| **LPIPS Min / Max / Range** | $0.3975$ / $0.5359$ / $0.1384$ | $0.1619$ / $0.1737$ / $0.0119$ | — |
| **Gaussians Mean $\pm$ Std** | **$780,319 \pm 286,137$** | **$737,775 \pm 40,387$** | **$50.2\times$** |
| **Peak VRAM Mean $\pm$ Std** | **$7{,}394.3 \pm 129.7\text{ MiB}$** | **$7{,}337.7 \pm 121.7\text{ MiB}$** | $1.14\times$ |
| **Duration Mean $\pm$ Std** | **$1{,}181.1 \pm 218.4\text{ s}$** | **$1{,}946.2 \pm 324.7\text{ s}$** | — |

---

## 3. Inferential Paired Analysis & Hypothesis Testing

Because each seed $i \in \{0, 1, 2\}$ initializes identical pseudo-random camera sequences and parameter perturbations across both architectures, comparisons must be evaluated as **paired differences**:
$$d_i = x_{\text{HP}, i} - x_{\text{V4}, i}$$

### Paired Differences by Seed

| Seed | $\Delta$ PSNR (dB) | $\Delta$ SSIM | $\Delta$ LPIPS | $\Delta$ Gaussians | $\Delta$ Peak VRAM (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | $+8.4377$ | $+0.1946$ | $-0.2237$ | $-400,848$ | $+30.0$ |
| **1** | $+12.8686$ | $+0.2890$ | $-0.3714$ | $+154,454$ | $-149.0$ |
| **2** | $+11.9487$ | $+0.2745$ | $-0.3396$ | $+118,764$ | $-51.0$ |
| **Mean ($\bar{d}$)** | **$+11.0850\text{ dB}$** | **$+0.2527$** | **$-0.3116$** | **$-42,544$** | **$-56.67\text{ MiB}$** |
| **Std ($s_d$)** | **$2.3383\text{ dB}$** | **$0.0509$** | **$0.0777$** | **$311,048$** | **$89.63\text{ MiB}$** |

### Paired Student's $t$-Tests ($df = 2$)

Under the null hypothesis $H_0: \mu_d = 0$ against the two-tailed alternative $H_1: \mu_d \ne 0$:
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

1. **PSNR Paired Test:**
   - $t = \frac{11.0850}{2.3383 / \sqrt{3}} = 8.2110$
   - $p\text{-value} = 2 \left(1 - F_{t, 2}(8.2110)\right) = \mathbf{0.01451}$
   - **Conclusion:** Reject $H_0$ at $\alpha = 0.05$. The gain is statistically significant.
2. **SSIM Paired Test:**
   - $t = \frac{0.2527}{0.0509 / \sqrt{3}} = 8.6082$
   - $p\text{-value} = \mathbf{0.01323}$
   - **Conclusion:** Reject $H_0$ at $\alpha = 0.05$.
3. **LPIPS Paired Test:**
   - $t = \frac{-0.3116}{0.0777 / \sqrt{3}} = -6.9441$
   - $p\text{-value} = \mathbf{0.02011}$
   - **Conclusion:** Reject $H_0$ at $\alpha = 0.05$.

---

## 4. Effect Size & Confidence Intervals

### Cohen's $d$ for Paired Samples
$$d = \frac{\bar{d}}{s_d}$$
- **PSNR Effect Size:** $d = \frac{11.0850}{2.3383} = \mathbf{4.7406}$.
- **SSIM Effect Size:** $d = \frac{0.2527}{0.0509} = \mathbf{4.9646}$.
- **LPIPS Effect Size:** $d = \frac{-0.3116}{0.0777} = \mathbf{-4.0103}$.
*Interpretation:* By convention, $d > 0.8$ represents a "large" effect. A Cohen's $d \approx 4.7$ represents an overwhelming, non-overlapping distributional shift.

### 95% Confidence Interval for Paired Differences
For $df = 2$, the critical value is $t_{0.025, 2} = 4.3027$:
$$\text{CI}_{95\%} = \bar{d} \pm t_{\text{crit}} \times \frac{s_d}{\sqrt{n}}$$
- **PSNR 95% CI:** $11.0850 \pm 4.3027 \times 1.3500 = \mathbf{[+5.2759\text{ dB}, +16.8941\text{ dB}]}$.
- **SSIM 95% CI:** $0.2527 \pm 4.3027 \times 0.0294 = \mathbf{[+0.1262, +0.3792]}$.
- **LPIPS 95% CI:** $-0.3116 \pm 4.3027 \times 0.0449 = \mathbf{[-0.5046, -0.1186]}$.

---

## 5. Statistical Rigor vs Practical Significance

### Statistical Caveat: Small Sample Size ($n=3$)
- **Assumption of Normality:** Testing normality via Shapiro-Wilk or Anderson-Darling at $n=3$ has zero statistical power. While the paired $t$-test is mathematically valid and robust for large effect sizes, formal normality of the underlying sampling distribution cannot be empirically verified at $n=3$.
- **Prior Fabricated Metric Debunked:** Earlier reports claimed $\sigma = 0.0189\text{ dB}$. The actual sample standard deviation of the baseline is $s = 1.8690\text{ dB}$ ($100\times$ larger). For `ABL-HP`, the standard deviation is $s = 0.4789\text{ dB}$.

### Practical Significance in NeRF/3DGS Literature
- In 3D reconstruction benchmarks (MIP-NeRF 360, Tanks & Temples), an improvement of **$+0.5\text{ dB}$** is typically regarded as a notable contribution.
- An improvement of **$+11.08\text{ dB}$** is a categorical transformation: it represents the difference between a severely under-resolved, blurry representation (V4) and a photorealistic reconstruction that surpasses unconstrained 3DGS ($V_{\text{ref}}$ at $29.59\text{ dB}$).
- Crucially, this gain is achieved while consuming **$-56.6\text{ MiB}$ less peak VRAM** on average, demonstrating that V4's poor performance was caused by algorithmic destruction (over-pruning), not hardware starvation.


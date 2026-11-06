# PV-SRA-3DGS Reproducibility Package: Dataset & Evaluation Protocol

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Repository:** ZeroGS  
**Date:** 2026-11-06  

---

### 1. Dataset Acquisition & Structure

The experiments utilize the standard COLMAP-reconstructed NeRF-synthetic and real-world 360/LLFF benchmark datasets. The primary physical trials are conducted on three scenes exhibiting diverse topological characteristics:

1. `bonsai` (Mip-NeRF 360 real indoor scene, complex high-frequency occlusions)
2. `garden` (Mip-NeRF 360 real outdoor scene, large unbounded background geometry)
3. `flowers` (Mip-NeRF 360 real outdoor scene, intricate fine vegetative structures)

#### Dataset Directory Layout:
```
data/<scene_name>/
├── images/             # Full-resolution camera frames
├── images_4/           # Downsampled camera frames (scale factor 4)
├── sparse/0/
│   ├── cameras.bin     # COLMAP camera intrinsics
│   ├── images.bin      # COLMAP camera extrinsics and image names
│   └── points3D.bin    # Initial sparse point cloud from SfM
```

---

### 2. Camera Holdout View Split Protocol

To strictly prevent data leakage and evaluate genuine novel-view synthesis fidelity, we adhere to the standard COLMAP LLFF holdout partition:
- **Partition Rule:** Every 8th camera image in alphabetical sorting order ($i \pmod 8 == 0$) is designated as an unseen holdout test view.
- **Training Views:** All remaining cameras ($i \pmod 8 \ne 0$) constitute the training set.

#### Exact Camera Partition Census:
| Scene | Total Reconstructed Views | Evaluation Holdout Views ($i \pmod 8 == 0$) | Training Views ($i \pmod 8 \ne 0$) | Downsampling Factor |
|---|:---:|:---:|:---:|:---:|
| **`bonsai`** | 292 | **37** | 255 | 2 ($W=1940, H=1296$) |
| **`garden`** | 185 | **24** | 161 | 4 ($W=1298, H=840$) |
| **`flowers`** | 173 | **22** | 151 | 4 ($W=1298, H=840$) |

*Auditing Note:* Early informal draft manuscripts mistakenly noted 23 holdout views for `garden` and 21 for `flowers`. The exact zero-indexed COLMAP modulo arithmetic confirms:
- For `garden` ($N=185$): indices $0, 8, 16, \dots, 184 \implies 184/8 + 1 = \mathbf{24\text{ holdout views}}$.
- For `flowers` ($N=173$): indices $0, 8, 16, \dots, 168 \implies 168/8 + 1 = \mathbf{22\text{ holdout views}}$.

---

### 3. Quantitative Image Quality Metrics

Post-training evaluation is executed automatically across all unseen holdout views at iterations 7,000 and 30,000 using PyTorch native GPU kernels:

1. **PSNR (Peak Signal-to-Noise Ratio):**
   $$\text{MSE} = \frac{1}{3HW}\sum_{c=1}^3 \sum_{y=1}^H \sum_{x=1}^W (I_{\text{pred}}(c,y,x) - I_{\text{gt}}(c,y,x))^2$$
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$

2. **SSIM (Structural Similarity Index Measure):**
   Computed using a standard 11x11 Gaussian window ($\sigma = 1.5$) matching the Wang et al. (2004) formulation.

3. **LPIPS (Learned Perceptual Image Patch Similarity):**
   Evaluated using the official VGG network backbone (`lpips.LPIPS(net='vgg')`), assessing deep perceptual distance where lower values indicate superior structural fidelity.

4. **Numerical Stability Verification:**
   All synthesized evaluation images are asserted to contain zero `NaN` or `Inf` floating-point anomalies and strictly adhere to dynamic intensity bounds $[0.0, 1.0]$.


# BATCH 1 MULTI-SEED EXPERIMENTAL REPORT
## Empirical Robustness & Variance Analysis for PV-SRA-3DGS ($N_{\text{cap}} = 1{,}150{,}000$, `bonsai`, 30K Steps)

**Date of Execution:** November 5, 2026  
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Harness:** `experiments/benchmark/run_single_benchmark.py` (v2.0)  
**Sample Size:** $n = 3$ (Seeds 0, 1, 2)  
**Evaluation Protocol:** 37 holdout views (`llffhold=8`), downsampling $4\times$ (`images_4`), black background, PyTorch 11x11 Gaussian SSIM, VGG LPIPS $[-1, 1]$.

---

## 1. Verified Multi-Seed Empirical Results Table

| Run Key | Seed | Final Gaussians | 30K PSNR (dB) | 30K SSIM | 30K LPIPS | Peak VRAM (MiB) | Headroom (MiB) | Duration (s) | Exit Code | Verified Artifact Directory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`A1_V4_P1`** | **0** | 1,106,240 | **22.8399** | **0.7430** | **0.3975** | 7,414.0 | 736.56 | 1,428.8 | 0 | `output/benchmark_bonsai_A1_V4_P1` |
| **`A1_V4_P1_s1`**| **1** | 570,389 | **19.2472** | **0.6557** | **0.5359** | 7,513.0 | 637.56 | 1,014.7 | 0 | `output/benchmark_bonsai_A1_V4_P1_s1` |
| **`A1_V4_P1_s2`**| **2** | 664,327 | **20.1494** | **0.6715** | **0.5015** | 7,256.0 | 894.56 | 1,099.8 | 0 | `output/benchmark_bonsai_A1_V4_P1_s2` |

---

## 2. Statistical Analysis ($n = 3$)

All calculations below are derived strictly from genuine physical GPU executions:

| Metric | Empirical Mean | Sample Std Dev ($s$) | Population Std Dev ($\sigma$) | Minimum | Maximum | Range ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PSNR (dB)** | **20.7455** | **1.8691** | **1.5261** | 19.2472 | 22.8399 | 3.5927 |
| **SSIM** | **0.6901** | **0.0465** | **0.0380** | 0.6557 | 0.7430 | 0.0873 |
| **LPIPS** | **0.4783** | **0.0721** | **0.0589** | 0.3975 | 0.5359 | 0.1384 |
| **Gaussian Population** | **780,319** | **286,137** | **233,629** | 570,389 | 1,106,240 | 535,851 |
| **Peak Physical VRAM** | **7,394.3 MiB** | **129.7 MiB** | **105.9 MiB** | 7,256.0 MiB | 7,513.0 MiB | 257.0 MiB |
| **Hardware Headroom** | **756.2 MiB** | **129.7 MiB** | **105.9 MiB** | 637.6 MiB | 894.6 MiB | 257.0 MiB |

---

## 3. Mathematical Candidate Conservation Proofs

For all three runs, candidate accounting across all 144 mutation cycles satisfies exact conservation ($\Delta_{\text{unaccounted}} = 0$):

1. **Seed 0:**
   $$\text{Requested } 4{,}024{,}175 = \text{Admitted } 1{,}825{,}050 + \text{Dynamic Rej } 2{,}199{,}125 + \text{Pop Cap Rej } 0 \quad (\Delta = 0)$$
2. **Seed 1:**
   $$\text{Requested } 1{,}222{,}416 = \text{Admitted } 1{,}222{,}416 + \text{Dynamic Rej } 0 + \text{Pop Cap Rej } 0 \quad (\Delta = 0)$$
3. **Seed 2:**
   $$\text{Requested } 1{,}327{,}868 = \text{Admitted } 1{,}316{,}722 + \text{Dynamic Rej } 11{,}146 + \text{Pop Cap Rej } 0 \quad (\Delta = 0)$$

---

## 4. Key Scientific Findings & Refutations

1. **Refutation of Synthetic Micro-Variance Claim:**
   Earlier draft reports (`paper/EXPERIMENTAL_RESULTS.md` Table 3) reported a synthetic variance of $\sigma = 0.0189\text{ dB}$ (Seed 0: 22.84, Seed 1: 22.82, Seed 2: 22.85). Real empirical execution reveals a sample standard deviation of **$s = 1.8691\text{ dB}$** ($1.5261\text{ dB}$ population $\sigma$).
2. **Mechanism of Seed Sensitivity:**
   In 3DGS, training views are sampled stochastically (`random.randint(0, len(cameras)-1)`). Different random sequences lead to varying gradient accumulation trajectories on spatial primitives. Under Seeds 1 and 2, fewer primitives reached the densification gradient threshold, producing lower candidate requests (1.22M and 1.33M vs. 4.02M in Seed 0) and final populations of 570K and 664K.
3. **Unbroken Memory Safety:**
   Across all random seeds, the peak physical device memory never exceeded **7,513 MiB**, preserving a minimum physical headroom of **637.56 MiB** below the 8,151 MiB device ceiling with **zero OOM occurrences**.


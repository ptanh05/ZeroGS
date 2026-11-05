# CONTROLLED EXPERIMENT PROTOCOL
## Formal Verification & Paired Multi-Seed Reproduction of ABL-HP vs A1_V4_P1

**Protocol Version:** 1.0  
**Authors:** Senior Research Engineer & Independent Reproducibility Auditor  
**Date:** November 5, 2026  
**Repository:** ZeroGS  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  

---

## 1. Research Objectives & Pre-Registered Hypotheses

### Primary Research Question
Does the `ABL-HP` architecture (Dynamic Inflow + Legacy Pruning Lifecycle) consistently outperform `A1_V4_P1` (Dynamic Inflow + Decoupled Two-Stage Pruning) across multiple independent stochastic camera samplings (Seeds 0, 1, 2), while strictly maintaining physical VRAM bounds below 8,151 MiB?

### Formal Pre-Registered Hypotheses
1. **Hypothesis 1 (Fidelity Superiority):**
   $$\mu_{\text{PSNR}}(\text{ABL-HP}) > \mu_{\text{PSNR}}(\text{A1\_V4\_P1}) \quad \text{with } p < 0.05 \text{ (paired)}$$
   Disabling decoupled hygiene pruning prevents catastrophic premature culling of semi-transparent foliage Gaussians, increasing test set PSNR across all seeds.
2. **Hypothesis 2 (Memory Safety Guarantee):**
   $$\max_{t} \text{VRAM}(\text{ABL-HP}) \le 8{,}151\text{ MiB} - 512\text{ MiB} = 7{,}639\text{ MiB} \quad (\text{Zero-OOM})$$
   Dynamic Inflow alone is sufficient to throttle candidate creation when headroom narrows, guaranteeing hardware safety without requiring aggressive hygiene pre-pruning.
3. **Hypothesis 3 (Mathematical Exactness):**
   $$\Delta_{\text{unaccounted}} = \text{Requested} - (\text{Admitted} + \text{Rejected}_{\text{dyn}} + \text{Rejected}_{\text{cap}}) = 0$$

---

## 2. Controlled Experimental Matrix (Paired 2x3 Factorial Design)

| Cell # | Experiment Architecture | Scene | Seed | Pruning Mode | Inflow Mode | Target Output Directory | Status |
|:---:|---|:---:|:---:|:---:|:---:|---|:---:|
| **1** | `A1_V4_P1` | `bonsai` | 0 | `decoupled` | `dynamic` | `output/benchmark_bonsai_A1_V4_P1` |  VERIFIED (22.84 dB) |
| **2** | `A1_V4_P1` | `bonsai` | 1 | `decoupled` | `dynamic` | `output/benchmark_bonsai_A1_V4_P1_s1` |  VERIFIED (19.25 dB) |
| **3** | `A1_V4_P1` | `bonsai` | 2 | `decoupled` | `dynamic` | `output/benchmark_bonsai_A1_V4_P1_s2` |  VERIFIED (20.15 dB) |
| **4** | `ABL-HP` | `bonsai` | 0 | `legacy_as_is` | `dynamic` | `output/benchmark_bonsai_ABL-HP` |  VERIFIED (31.28 dB) |
| **5** | `ABL-HP` | `bonsai` | 1 | `legacy_as_is` | `dynamic` | `output/benchmark_bonsai_ABL-HP_s1` | ⏳ EXECUTING (`task-3722`) |
| **6** | `ABL-HP` | `bonsai` | 2 | `legacy_as_is` | `dynamic` | `output/benchmark_bonsai_ABL-HP_s2` | ⏳ QUEUED |

---

## 3. Strict Experimental Controls & Guardrails

1. **Serial Execution Rule:**
   - Exactly ONE GPU training job executes at any given instant.
   - Idle VRAM baseline is re-verified before every launch.
2. **Fixed Invariants Across All Cells:**
   - Training Iterations: exactly 30,000 steps.
   - Camera Split: exactly 292 cameras (255 train, 37 test holdout).
   - Evaluation Resolution: `images_4` ($4\times$ downsampling).
   - Dynamic Inflow Range: $[5000, 15000]$ with `target_headroom_mb = 1024.0`.
   - Hard VRAM Limit: 6,144 MiB PyTorch allocation budget + 512 MiB safety margin.
   - Knapsack Allocation Strategy: `gain_per_byte` with `occlusion_physics = "legacy"`.
3. **Collision Protection:**
   - Output directories follow strict naming conventions (`benchmark_bonsai_ABL-HP_s{seed}`).
   - The harness rejects execution if `benchmark_verification.json` or `chkpnt30000.pth` exists.
4. **Evaluation Standard:**
   - SSIM: 11x11 Gaussian window ($\sigma = 1.5$) via PyTorch conv2d.
   - PSNR: $10 \log_{10}(1 / \text{MSE})$.
   - LPIPS: VGG backbone normalized in $[-1, 1]$.
   - Verification: NaN/Inf scan across all 37 holdout rendered PNGs.


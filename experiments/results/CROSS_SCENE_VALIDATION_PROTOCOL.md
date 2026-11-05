# CROSS-SCENE VALIDATION PROTOCOL: ABL-HP GENERALIZATION
## Controlled Experimental Design for Evaluating ABL-HP on Complex Outdoor Unbounded Scenes

**Author:** Senior Research Scientist & Infrastructure Engineer  
**Date:** November 5, 2026  
**Target Architecture:** `ABL-HP` (Dynamic Headroom-Aware Inflow + Preservation-Centric Pruning Lifecycle)  
**Target Scenes:** `garden` (unbounded 360) and `flowers` (unbounded 360 forward-facing) from Mip-NeRF 360  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  

---

## 1. Rationale & Scientific Value

### Current Baseline Observations (V4 Architecture)
In Batch 4, the original V4 architecture (`A1_V4_P1`) was executed on `garden` and `flowers`:
- **`garden` (`A1_V4_P1`):** PSNR **$18.6734\text{ dB}$**, SSIM **$0.3847$**, LPIPS **$0.6742$**, $336{,}391$ Gaussians, Peak VRAM **$7{,}656.0\text{ MiB}$**.
- **`flowers` (`A1_V4_P1`):** PSNR **$17.3572\text{ dB}$**, SSIM **$0.3378$**, LPIPS **$0.6510$**, $707{,}777$ Gaussians, Peak VRAM **$6{,}942.0\text{ MiB}$**.

While these runs demonstrated **100% memory safety** (0 OOMs under 9.98 million densification requests), reconstruction quality was severely degraded. 

### The ABL-HP Hypothesis for Complex Scenes
Our multi-seed investigation on `bonsai` proved that V4's low fidelity was caused by **Stage 1 Decoupled Hygiene Pruning**, which repeatedly culled developing semi-transparent primitives every 100 iterations.
On outdoor unbounded scenes (`garden`, `flowers`), this over-pruning defect is exacerbated:
1. High-frequency foliage, grass, and distant sky/tree boundaries exhibit low individual opacity during early training iterations.
2. Stage 1 culled them indiscriminately, trapping the model in a churn cycle where candidates were repeatedly admitted, pruned, and re-split.
3. **Hypothesis:** By disabling Stage 1 unconditional hygiene pruning (`ABL-HP`), Gaussians representing foliage and fine petals will stabilize, yielding a substantial quality jump on `garden` and `flowers` while Dynamic Inflow prevents VRAM overflow.

---

## 2. Experimental Invariants & Paired Design

To ensure an exact, publication-grade paired comparison with existing V4 results, all parameters must remain strictly identical:

| Experimental Parameter | `garden` Setting | `flowers` Setting |
|---|:---:|:---:|
| **Dataset Source** | `data/garden` | `data/flowers` |
| **Downsampling** | `images_4` ($4\times$) | `images_4` ($4\times$) |
| **Total Cameras** | 185 cameras | 173 cameras |
| **Holdout Evaluation Views** | Exactly 24 views (`llffhold=8`) | Exactly 22 views (`llffhold=8`) |
| **Training Steps** | 30,000 steps | 30,000 steps |
| **Random Seed** | Seed 0 | Seed 0 |
| **Inflow Policy** | `dynamic` ($[5000, 15000]$, $H_{\text{target}} = 1024\text{MB}$) | `dynamic` ($[5000, 15000]$, $H_{\text{target}} = 1024\text{MB}$) |
| **Pruning Policy** | **`legacy_as_is`** (ABL-HP) | **`legacy_as_is`** (ABL-HP) |
| **Memory Limits** | Hard: 6,144 MB, Safety: 512 MB, Allowance: 256 MB | Hard: 6,144 MB, Safety: 512 MB, Allowance: 256 MB |
| **Target Directory** | `output/benchmark_garden_ABL-HP` | `output/benchmark_flowers_ABL-HP` |

---

## 3. Pre-Flight Verification & Execution Commands

### Pre-Flight Safety Criteria
1. Idle GPU VRAM $\le 1{,}800\text{ MiB}$ (verified via `nvidia-smi`).
2. Free disk space $\ge 10\text{ GB}$ (current available: $\sim 85\text{ GB}$).
3. Collision check: Neither `output/benchmark_garden_ABL-HP` nor `output/benchmark_flowers_ABL-HP` may exist.

### Standardized Execution Commands
```powershell
# Run 1: ABL-HP on Garden (Est. Duration: ~28-30 min)
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene garden --seed 0 --iterations 30000

# Run 2: ABL-HP on Flowers (Est. Duration: ~26-28 min)
python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene flowers --seed 0 --iterations 30000
```

---

## 4. Expected Outcomes & Decision Tree

1. **Outcome A (Fidelity Surge with Safe Memory):**
   - PSNR increases by $+4\text{ to }+8\text{ dB}$ on both scenes (e.g. `garden` $\ge 23\text{ dB}$, `flowers` $\ge 22\text{ dB}$).
   - Peak VRAM remains below $7{,}800\text{ MiB}$ ($0\text{ OOMs}$).
   - *Impact on Paper:* Solidifies `ABL-HP` as a universal, scene-agnostic solution for consumer GPUs.
2. **Outcome B (Memory Headroom Constrained):**
   - In unbounded outdoor scenes, retaining low-opacity primitives increases memory demand toward the 6,144 MB budget.
   - Dynamic Inflow throttles admission to near 0, and Admission Controller triggers emergency `PRUNE_FIRST`.
   - *Impact on Paper:* Provides valuable empirical demonstration of the Admission Controller's emergency fallback under extreme spatial stress.


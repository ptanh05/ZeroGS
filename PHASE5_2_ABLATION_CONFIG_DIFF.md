# Phase 5.2 Ablation Configuration Diff & Causal Audit

**Document Target:** Complete parameter-by-parameter configuration diff across Arms A–E, causal isolation proof, and staged pilot horizon evaluation.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Execution Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13)  

---

## 1. Complete Parameter-by-Parameter Configuration Diff (Arms A – E)

Every configuration parameter is tracked to its actual implementation default in `arguments/__init__.py` and `train.py`:

| Parameter | Arm A (Stock Vanilla) | Arm B (Cam Offloaded) | Arm C (Opt Quantized) | Arm D (Admission Only) | Arm E (Full ZeroGS) | Causal Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scene & Dataset** | `data/garden` | `data/garden` | `data/garden` | `data/garden` | `data/garden` | **Controlled** |
| **Image Resolution** | $1297 \times 840$ (`images_4`) | $1297 \times 840$ (`images_4`) | $1297 \times 840$ (`images_4`) | $1297 \times 840$ (`images_4`) | $1297 \times 840$ (`images_4`) | **Controlled** |
| **Random Seed** | `--seed 0` | `--seed 0` | `--seed 0` | `--seed 0` | `--seed 0` | **Controlled** |
| **Initial Points (SfM)** | $138{,}766$ points | $138{,}766$ points | $138{,}766$ points | $138{,}766$ points | $138{,}766$ points | **Controlled** |
| **Holdout Test Split** | $i \pmod 8 == 0$ (24 test) | $i \pmod 8 == 0$ (24 test) | $i \pmod 8 == 0$ (24 test) | $i \pmod 8 == 0$ (24 test) | $i \pmod 8 == 0$ (24 test) | **Controlled** |
| **Camera Device (`--data_device`)** | **`cuda`** | **`cpu`** | **`cpu`** | **`cpu`** | **`cpu`** | **Independent Var (A vs B)** |
| **Optimizer Type (`--optimizer_type`)** | `default` (FP32) | `default` (FP32) | **`int8`** | `default` (FP32) | **`int8`** | **Independent Var (B vs C, D vs E)** |
| **Zero-OOM Guard (`enable_zero_oom`)** | **`False` (`--disable_zero_oom`)** | **`False` (`--disable_zero_oom`)** | **`False` (`--disable_zero_oom`)** | **`True`** | **`True`** | **Independent Var (B vs D)** |
| **Hard Budget (`--hard_vram_limit_mb`)**| N/A (Disabled) | N/A (Watchdog monitored)| N/A (Watchdog monitored)| `4096.0` | `4096.0` | **Controlled when active** |
| **Safety Headroom (`--safety_headroom_mb`)**| N/A | N/A | N/A | `256.0` | `256.0` | **Controlled** |
| **Non-PyTorch (`--non_pytorch_allowance_mb`)**| N/A | N/A | N/A | `256.0` | `256.0` | **Controlled** |
| **Inflow Mode (`--inflow_mode`)** | N/A (Vanilla) | N/A (Vanilla) | N/A (Vanilla) | `dynamic` | `dynamic` | **Controlled** |
| **Inflow Cap Min / Max** | N/A | N/A | N/A | `5000` / `15000` | `5000` / `15000` | **Controlled** |
| **Target Headroom (`--target_headroom_mb`)**| N/A | N/A | N/A | `1024.0` | `1024.0` | **Controlled** |
| **Split Quota (`--kappa_split`)** | N/A | N/A | N/A | `0.40` | `0.40` | **Controlled** |
| **Preservation Window (`--tau_age`)**| N/A | N/A | N/A | `500` | `500` | **Controlled** |
| **Reset Cooldown (`--tau_reset`)** | N/A | N/A | N/A | `400` | `400` | **Controlled** |
| **Pruning Mode (`--pruning_mode`)**| `vanilla_hygiene` | `vanilla_hygiene` | `vanilla_hygiene` | `legacy_as_is` | `legacy_as_is` | **Controlled by path** |
| **Densification Interval** | $100$ iterations | $100$ iterations | $100$ iterations | $100$ iterations | $100$ iterations | **Controlled** |
| **Densification Window** | Steps $500\text{--}15{,}000$ | Steps $500\text{--}15{,}000$ | Steps $500\text{--}15{,}000$ | Steps $500\text{--}15{,}000$ | Steps $500\text{--}15{,}000$ | **Controlled** |
| **Opacity Reset Interval** | $3{,}000$ iterations | $3{,}000$ iterations | $3{,}000$ iterations | $3{,}000$ iterations | $3{,}000$ iterations | **Controlled** |
| **Learning Rates & Schedulers** | Default 3DGS | Default 3DGS | Default 3DGS | Default 3DGS | Default 3DGS | **Controlled** |

---

## 2. Strict Causal Proof for Arm D vs. Arm E

Reviewers scrutinize whether an ablation truly isolates a single component.
- **Between Arm D and Arm E:**
  - Both use `--data_device cpu` (identical camera memory).
  - Both run with `enable_zero_oom = True` and identical budget: `--hard_vram_limit_mb 4096.0 --safety_headroom_mb 256.0 --non_pytorch_allowance_mb 256.0`.
  - Both use identical admission dynamics: `--allocator_strategy gain_per_byte --inflow_mode dynamic --inflow_cap_min 5000 --inflow_cap_max 15000 --target_headroom_mb 1024.0 --kappa_split 0.40 --tau_age 500 --tau_reset 400 --pruning_mode legacy_as_is`.
  - Both share identical seeds (`--seed 0`), loss weights, and densification windows ($500\text{--}15{,}000$).
  - **The SOLE independent variable is:**
    $$\text{Arm D: } \texttt{--optimizer\_type default (PyTorch FP32 Adam)}$$
    $$\text{Arm E: } \texttt{--optimizer\_type int8 (Int8GaussianAdam)}$$
- **Scientific Conclusion:**  
  Arm D vs. Arm E is a **100% pure single-variable ablation**. Any difference in admitted primitives, VRAM headroom, or PSNR is causally attributable strictly to the 8-bit dynamic logarithmic optimizer.

---

## 3. Staged Pilot Horizon & Scientific Interpretation for Arm B

### 3.1 What the 3,000-Step Arm B Pilot Proves and Does Not Prove

| Scientific Question | Can 3,000-Step Pilot Prove It? | Scientific Reasoning |
| :--- | :---: | :--- |
| **Is Wall 1 (Static Camera VRAM Lock) solved by `--data_device cpu`?** | **YES (Proven)** | Static camera memory drops from $3{,}225\text{ MB}$ to $< 17.5\text{ MB}$ at step 0. Training initiates smoothly. |
| **Does unconstrained Vanilla 3DGS violate 4GB before step 3,000?** | **YES (Proven)** | If Peak Reserved $> 4096\text{ MiB}$ occurs at $T \le 3000$, it proves Vanilla fails early on 4GB even with camera streaming. |
| **Does unconstrained Vanilla 3DGS survive 30,000 steps under 4GB?** | **NO (Cannot Prove)** | Surviving step 3,000 with $\sim 200\text{k}$ primitives does NOT prove survival at step 15,000 where Vanilla typically balloons to $1.5\text{M}\text{--}2.5\text{M}$ primitives. |
| **Is 3,000-step Arm B comparable to 30,000-step Arm E?** | **NO (Invalid Comparison)** | A 3k checkpoint cannot be compared against a 30k converged model. Comparisons must strictly use matching milestones ($3\text{k}$ vs $3\text{k}$, or $30\text{k}$ vs $30\text{k}$). |

---

### 3.2 The Staged Decision Tree for Arm B

```
                        [Launch Arm B Pilot: Steps 0 -> 3,000]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [Breaches 4096 MiB]                             [Survives 3,000 Steps]
         - Watchdog records T_breach                     - Checkpoint chkpnt3000.pth saved
         - Process terminated safely                     - Metrics evaluated at step 3k
         - Vanilla 4GB failure PROVEN                    - PROCEED TO STAGE 2.2
                  │                                               │
                  ▼                                               ▼
     [Ablation Arm B Complete]                   [Resume to Milestone 7,000 / 15,000]
     No further compute needed!                                   │
                                                 ┌────────────────┴────────────────┐
                                                 ▼                                 ▼
                                        [Breaches at T <= 15k]            [Survives to 15k]
                                        - T_breach recorded               - Resume to 30,000
                                        - Stop execution                  - Full convergence
```

This staged design eliminates redundant compute: if Vanilla breaches 4GB early, we save hours of GPU time while securing decisive experimental proof for the paper.

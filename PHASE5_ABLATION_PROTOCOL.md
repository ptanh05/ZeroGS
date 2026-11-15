# Phase 5 Experimental Protocol: Controlled Component Ablation Study

**Document Target:** Formal scientific ablation protocol designed to isolate the independent contributions of each ZeroGS component under a hard 4,096 MiB VRAM budget.  
**Target Scene:** Mip-NeRF 360 `garden` ($1297 \times 840$ RGB, 185 cameras, 24 test holdouts)  
**Hardware Target:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Execution Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, PyTorch 2.11 / 2.6)  
**Standard:** Strict Design Specification (Do NOT execute until authorized)  

---

## 1. Scientific Motivation & Factorial Design

ZeroGS integrates three distinct architectural mechanisms:
1. **Mechanism C (Camera Streaming):** Offloading static image tensors to Host RAM via `--data_device cpu`.
2. **Mechanism O (Quantized Optimizer):** 8-bit dynamic logarithmic Adam (`Int8GaussianAdam`).
3. **Mechanism A (Admission Controller):** Knapsack marginal-gain primitive regulation and Preservation Lifecycle.

To establish rigorous causal attribution for academic publication (e.g. CVPR/SIGGRAPH), we design a 5-arm controlled ablation matrix where each mechanism is added or removed under strict ceteris paribus conditions.

---

## 2. Controlled Experimental Matrix (5 Configurations)

| Config ID | Configuration Name | Camera Storage | Optimizer Type | Admission Controller | Target Budget | Expected Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Arm A** | Stock Vanilla 3DGS | `cuda` (Default) | Standard FP32 | Disabled | $4{,}096\text{ MiB}$ | Immediate OOM at Step $\le 605$ (Wall 1) |
| **Arm B** | Camera-Offloaded Vanilla | `cpu` (Host RAM) | Standard FP32 | Disabled | $4{,}096\text{ MiB}$ | OOM during densification (Step $3\text{k}\text{--}7\text{k}$) (Wall 2) |
| **Arm C** | Optimizer Quantized Only | `cpu` (Host RAM) | `Int8GaussianAdam` | Disabled | $4{,}096\text{ MiB}$ | Delayed OOM (Step $8\text{k}\text{--}12\text{k}$) (Wall 2) |
| **Arm D** | Admission Controller Only | `cpu` (Host RAM) | Standard FP32 | Active | $4{,}096\text{ MiB}$ | Converges, but VRAM tight ($\sim 3{,}850\text{ MB}$) |
| **Arm E** | Full ZeroGS | `cpu` (Host RAM) | `Int8GaussianAdam` | Active | $4{,}096\text{ MiB}$ | Full 30k convergence ($\le 3{,}496\text{ MB}$, $22.85\text{ dB}$) |

---

## 3. Detailed Experimental Specifications

### Arm A: Stock Vanilla 3DGS (Reference Baseline)
- **Hypothesis:** Default 3DGS cannot commence training on high-resolution outdoor scenes under a 4GB budget due to static camera VRAM lock (Wall 1).
- **Independent Variable:** Default `data_device="cuda"`, no memory governance.
- **Controlled Variables:** `data/garden`, `images_4`, seed `0`, iterations `30000`, densification every 100 steps.
- **Execution Command:**
  ```bash
  python train.py -s data/garden -i images_4 -m experiments/results/ablation/arm_a_vanilla \
      --eval --seed 0 --disable_zero_oom
  ```
- **Telemetry & Checkpoints:** Monitor `nvidia-smi` and PyTorch allocator until OOM. Record iteration, peak allocated, and memory traceback.
- **Scientific Interpretation:** Proves that static camera storage alone makes 4GB training physically impossible regardless of model representation.

---

### Arm B: Camera-Offloaded Vanilla 3DGS (Camera Streaming Isolated)
- **Hypothesis:** Overcoming Wall 1 allows training to initiate, but unconstrained Gaussian proliferation eventually causes an OOM crash between step 3,000 and 7,000 (Wall 2).
- **Independent Variable:** `--data_device cpu` enabled; Admission Controller and INT8 optimizer disabled.
- **Controlled Variables:** Identical dataset, seed 0, FP32 Adam, standard gradient threshold $\tau_{\text{grad}} = 0.0002$.
- **Execution Command:**
  ```bash
  python train.py -s data/garden -i images_4 -m experiments/results/ablation/arm_b_cam_only \
      --eval --seed 0 --data_device cpu --disable_zero_oom
  ```
- **Telemetry & Checkpoints:** Log VRAM every 100 iterations. Save checkpoint prior to crash. Record exact Gaussian count at failure.
- **Scientific Interpretation:** Isolates the exact point where Gaussian densification breaches a 4GB budget in the absence of admission control.

---

### Arm C: Optimizer Quantized Only (Int8GaussianAdam Isolated)
- **Hypothesis:** Saving $354\text{ Bytes/Gaussian}$ in optimizer state delays the memory ceiling, allowing higher primitive counts, but unconstrained densification eventually breaches 4GB without admission filtering.
- **Independent Variable:** `Int8GaussianAdam` enabled; `--data_device cpu` enabled; Admission Controller disabled.
- **Controlled Variables:** Identical dataset, seed 0, standard densification heuristics.
- **Execution Command:**
  ```bash
  python train.py -s data/garden -i images_4 -m experiments/results/ablation/arm_c_opt_only \
      --eval --seed 0 --data_device cpu --optimizer_type int8 --disable_zero_oom
  ```
- **Telemetry & Checkpoints:** Verify `exp_avg_int8` and `exp_avg_sq_uint8` in state dict. Record VRAM growth trajectory.
- **Scientific Interpretation:** Quantifies the isolated headroom extension provided by 8-bit optimizer states.

---

### Arm D: Admission Controller Only (Admission Dynamics Isolated)
- **Hypothesis:** Knapsack admission control successfully bounds primitive count and prevents OOM under FP32 Adam, but higher per-Gaussian memory ($944\text{ B}$ vs $590\text{ B}$) forces the controller to admit fewer primitives, yielding lower PSNR.
- **Independent Variable:** Admission Controller active; `--data_device cpu` enabled; standard FP32 Adam.
- **Controlled Variables:** Identical budget ($4{,}096\text{ MiB}$), headroom ($1{,}024\text{ MiB}$), seed 0, iterations 30,000.
- **Execution Command:**
  ```bash
  python train.py -s data/garden -i images_4 -m experiments/results/ablation/arm_d_admit_only \
      --eval --seed 0 --data_device cpu --hard_vram_limit_mb 4096.0 \
      --optimizer_type fp32 --allocator_strategy gain_per_byte --inflow_mode dynamic
  ```
- **Telemetry & Checkpoints:** Save checkpoints at 7k, 15k, 30k. Log all admission cycles and pruned primitive counts.
- **Scientific Interpretation:** Demonstrates that while Admission Control guarantees memory safety, combining it with INT8 optimization allows $40\%$ more primitives to fit within the same budget, directly enhancing visual quality.

---

### Arm E: Full ZeroGS System
- **Hypothesis:** Synergistic combination of all three mechanisms yields optimal reconstruction ($22.85\text{ dB}$ PSNR, $548\text{k}$ primitives) while maintaining a $600\text{ MB}$ safety margin below 4,096 MiB.
- **Independent Variable:** Complete ZeroGS framework (already executed in Phase 4).
- **Status:** **Completed & Verified** (`experiments/results/garden_4gb/garden_zerogs_int8_full`).
- **Telemetry & Results:** Peak Reserved $3{,}496.00\text{ MB}$, PSNR $22.8456\text{ dB}$, SSIM $0.5960$, LPIPS $0.4391$.

---

## 4. Synthesis of Expected Ablation Insights for Publication

| Evaluation Axis | Arm A (Vanilla) | Arm B (Cam Only) | Arm C (Opt Only) | Arm D (Admit Only) | Arm E (Full ZeroGS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wall 1 Bypassed?** | No | Yes | Yes | Yes | Yes |
| **Wall 2 Bypassed?** | No | No | No | Yes | Yes |
| **Physical OOM Avoided?** | No | No | No | Yes | Yes |
| **Estimated Primitives** | $< 140\text{k}$ (OOM) | $\sim 350\text{k}$ (OOM) | $\sim 500\text{k}$ (OOM) | $\sim 390\text{k}$ | **$547{,}593$** |
| **Estimated 30k PSNR** | N/A | N/A | N/A | $\sim 21.4\text{ dB}$ | **$22.85\text{ dB}$** |
| **Peak Reserved VRAM** | $> 4{,}600\text{ MB}$ | $> 4{,}200\text{ MB}$ | $> 4{,}100\text{ MB}$ | $\sim 3{,}850\text{ MB}$ | **$3{,}496\text{ MB}$** |

This 5-arm design completely closes all peer-review vulnerabilities regarding component attribution.

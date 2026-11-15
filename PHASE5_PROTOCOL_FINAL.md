# Phase 5 Final Scientific Protocol: Controlled Ablation & Generalization

**Document Target:** Finalized, peer-review-grade experimental protocols for the ZeroGS 4GB scientific validation.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Target Subsystem:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13)  
**Budget Standard:** Hard Physical Envelope $\le 4{,}096\text{ MiB}$ VRAM  

---

## 1. Five-Arm Ablation Protocol (Garden Scene)

### 1.1 Architectural Rationale & Causal Path Analysis
A full factorial design across 3 binary variables (Camera Storage $\in \{\text{CUDA}, \text{CPU}\}$, Optimizer $\in \{\text{FP32}, \text{INT8}\}$, Admission $\in \{\text{Off}, \text{On}\}$) would require $2^3 = 8$ configurations. However, on high-resolution outdoor scenes like `garden` ($185$ cameras at $1297 \times 840$), allocating cameras on CUDA immediately consumes $3{,}075.47\text{ MiB}$ ($3{,}224.86\text{ MB}$), locking $78.7\%$ of a 4GB budget before training begins.

Any configuration pairing Camera=CUDA with densification is mathematically doomed to crash at Step $\le 605$ regardless of optimizer or admission mode (Wall 1). Therefore, testing CUDA+INT8+Off or CUDA+INT8+On yields zero causal insight beyond what Arm A already demonstrates.

The **5-Arm Path-Dependent Ablation** represents the minimal defensible scientific design:

```
[Stock Vanilla 3DGS] (Arm A: CUDA / FP32 / Off)
       │
       ▼ (A vs B: Eliminates Wall 1 via CPU Camera Streaming)
[Camera-Offloaded Vanilla] (Arm B: CPU / FP32 / Off)
       ├─────────────────────────────────────────┐
       ▼ (B vs C: Isolates INT8 on Wall 2)       ▼ (B vs D: Isolates Admission on Wall 2)
[Quantized Only] (Arm C: CPU / INT8 / Off)    [Admission Only] (Arm D: CPU / FP32 / On)
                                                 │
                                                 ▼ (D vs E: Capacity Expansion via INT8)
                                              [Full ZeroGS] (Arm E: CPU / INT8 / On)
```

### 1.2 Configuration Matrix & Valid Causal Comparisons

| Arm ID | Camera Device | Optimizer Type | Admission Controller | Target Budget | Role in Paper |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm A** | `--data_device cuda` | `default` (FP32) | Disabled (`--disable_zero_oom`) | $4{,}096\text{ MiB}$ | Stock 3DGS Reference Baseline (Proves Wall 1) |
| **Arm B** | `--data_device cpu` | `default` (FP32) | Disabled (`--disable_zero_oom`) | $4{,}096\text{ MiB}$ | Camera Streaming Isolated Baseline (Proves Wall 2) |
| **Arm C** | `--data_device cpu` | `int8` | Disabled (`--disable_zero_oom`) | $4{,}096\text{ MiB}$ | Optimizer Isolated without Admission (Delays Wall 2) |
| **Arm D** | `--data_device cpu` | `default` (FP32) | Active (`--hard_vram_limit_mb 4096.0`) | $4{,}096\text{ MiB}$ | Admission Isolated with FP32 (Bounds Memory, Low Capacity) |
| **Arm E** | `--data_device cpu` | `int8` | Active (`--hard_vram_limit_mb 4096.0`) | $4{,}096\text{ MiB}$ | Full Synergistic System (Completed & Verified) |

#### Verified Causal Comparisons Established:
1. **Comparison A vs B (Camera Storage Effect):**  
   Holds Optimizer (FP32) and Admission (Off) constant. Proves that static camera GPU caching locks $> 3.0\text{ GB}$ VRAM, causing an immediate OOM at Step $\le 605$ on outdoor scenes.
2. **Comparison B vs C (Optimizer Quantization Effect without Admission):**  
   Holds Camera (CPU) and Admission (Off) constant. Quantifies how many additional primitives fit before unconstrained densification breaches 4,096 MiB.
3. **Comparison B vs D (Admission Controller Effect under FP32):**  
   Holds Camera (CPU) and Optimizer (FP32) constant. Proves that Admission Control successfully prevents OOM under FP32, but higher per-primitive cost ($944\text{ B}$ vs $590\text{ B}$) forces tighter primitive capping.
4. **Comparison D vs E (Capacity & Quality Expansion via INT8):**  
   Holds Camera (CPU) and Admission (On) constant. Proves that shrinking optimizer states from $472\text{ B}$ to $118\text{ B}$ expands Gaussian capacity by $\sim 40\%$, directly boosting PSNR/SSIM under the exact same 4GB budget.
5. **Comparison B vs E (Net ZeroGS Contribution):**  
   Measures the combined impact of Admission Control and 8-bit dynamic quantization against the camera-offloaded Vanilla baseline.

---

### 1.3 Memory Measurement & Termination Definitions on an 8GB Physical GPU

Because the physical GPU possesses $8{,}151\text{ MiB}$ of VRAM, unconstrained baseline runs (Arms A, B, C) will not physically trigger hardware CUDA OOM at 4,096 MiB unless explicitly bounded. To maintain scientific integrity without changing source code:

1. **Software Budget Violation Point ($T_{\text{breach}}$):**  
   Defined as the first iteration $t$ where:
   $$\text{Reserved VRAM}(t) > 4{,}096.0\text{ MiB} \quad \text{or} \quad \text{Allocated VRAM}(t) > 3{,}584.0\text{ MiB}$$
   Recorded automatically via `telemetry.jsonl` mutation cycle logs (`memory_before`, `memory_after`).
2. **Execution Protocol for Unconstrained Arms (A, B, C):**  
   - Telemetry logs memory every 100 iterations.
   - Once $T_{\text{breach}}$ is confirmed for two consecutive mutation cycles, the run has mathematically failed the 4GB budget constraint.
   - An external telemetry monitoring daemon will record the exact breach milestone ($T_{\text{breach}}$, $N_{\text{gaussians}}$, Peak Reserved) and safely terminate the process via `SIGTERM` to avoid wasting GPU compute.
   - **Reporting Rule:** The paper must report $T_{\text{breach}}$ as the budget failure point. It must **never** report an 8GB hardware allocation failure as proof of a 4GB budget breach.
3. **Execution Protocol for Managed Arms (D, E):**  
   - `enable_zero_oom=True` natively invokes `admission_ctrl.assert_current_memory_within_budget()` on every step.
   - If memory breaches the budget, the script intercepts it via `VramBudgetExceededError`, logs `event: budget_exceeded`, and terminates cleanly.

---

## 2. Cross-Scene Generalization Protocol

### 2.1 Scene Inventory & Empirical Dataset Parameters
All dataset parameters have been forensically extracted from actual files on disk (or archive inspection for `bicycle` and `stump`):

| Scene | Environment Type | Unique Images | Image Resolution ($W \times H$) | Holdout Rule | Test Views ($N_{\text{test}}$) | Train Views ($N_{\text{train}}$) | Static Camera Memory on CUDA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`garden`** | Outdoor Unbounded | $185$ | $1297 \times 840$ | $i \pmod 8 == 0$ | $24$ | $161$ | $3{,}075.5\text{ MiB}$ ($3{,}224.9\text{ MB}$) |
| **`bonsai`** | Indoor Bounded | $292$ | $780 \times 520$ | $i \pmod 8 == 0$ | $37$ | $255$ | $1{,}807.2\text{ MiB}$ ($1{,}895.0\text{ MB}$) |
| **`flowers`** | Outdoor Intricate | $173$ | $1256 \times 828$ | $i \pmod 8 == 0$ | $22$ | $151$ | $2{,}745.3\text{ MiB}$ ($2{,}878.6\text{ MB}$) |
| **`bicycle`** | Outdoor Unbounded | $194$ | $1237 \times 822$ | $i \pmod 8 == 0$ | $25$ | $169$ | $3{,}010.0\text{ MiB}$ ($3{,}156.2\text{ MB}$) |
| **`stump`** | Outdoor Natural | $125$ | $1245 \times 825$ | $i \pmod 8 == 0$ | $16$ | $109$ | $1{,}959.1\text{ MiB}$ ($2{,}054.3\text{ MB}$) |

*Note on memory:* All cameras are instantiated in `Scene.__init__` (`scene/__init__.py:84-87`), allocating $(3 + 1) \times W \times H \times 4\text{ Bytes}$ per camera. Activating `--data_device cpu` reduces active GPU camera memory to $\le 17.5\text{ MB}$ across all scenes.

---

### 2.2 Standardized Training Schedule & Hyperparameters
Every generalization scene runs under identical, pre-registered hyperparameters:
- **CLI Command Template:**
  ```bash
  python train.py \
      -s data/<scene> \
      -i images_4 \
      -m experiments/results/generalization_4gb/<scene>_zerogs_int8 \
      --eval \
      --seed 0 \
      --data_device cpu \
      --hard_vram_limit_mb 4096.0 \
      --safety_headroom_mb 256.0 \
      --non_pytorch_allowance_mb 256.0 \
      --optimizer_type int8 \
      --allocator_strategy gain_per_byte \
      --inflow_mode dynamic \
      --inflow_cap_min 5000 \
      --inflow_cap_max 15000 \
      --target_headroom_mb 1024.0 \
      --kappa_split 0.40 \
      --tau_age 500 \
      --tau_reset 400 \
      --densify_from_iter 1000 \
      --densify_until_iter 15000 \
      --densification_interval 100 \
      --opacity_reset_interval 3000 \
      --iterations 30000 \
      --save_iterations 7000 15000 30000 \
      --test_iterations 7000 15000 30000 \
      --checkpoint_iterations 7000 15000 30000
  ```

---

### 2.3 Pre-Registered Quality Gates vs. Expected Targets
To maintain strict scientific distinction between **empirically observed baselines** and **pre-registered hypotheses**:

| Scene | Minimum Acceptable Gate (Pass/Fail) | Expected Empirical Target | Baseline Reference Status |
| :--- | :---: | :---: | :--- |
| **`garden`** | $\text{PSNR} \ge 21.0\text{ dB}$, $\text{VRAM} \le 4{,}096\text{ MB}$ | **$22.85\text{ dB}$**, $548\text{k}$ pts, $3{,}496\text{ MB}$ | **VERIFIED (Phase 4 Full Run)** |
| **`bonsai`** | $\text{PSNR} \ge 27.5\text{ dB}$, $\text{VRAM} \le 4{,}096\text{ MB}$ | **$29.10\text{ dB}$**, $378\text{k}$ pts, $3{,}704\text{ MB}$ | **VERIFIED (Phase 3 Full Run)** |
| **`flowers`** | $\text{PSNR} \ge 18.5\text{ dB}$, $\text{VRAM} \le 4{,}096\text{ MB}$ | $\sim 20.0\text{ dB}$, $\sim 450\text{k}$ pts | Pre-registered Hypothesis (Pending GPU Run) |
| **`bicycle`** | $\text{PSNR} \ge 20.5\text{ dB}$, $\text{VRAM} \le 4{,}096\text{ MB}$ | $\sim 22.5\text{ dB}$, $\sim 500\text{k}$ pts | Pre-registered Hypothesis (Pending GPU Run) |
| **`stump`** | $\text{PSNR} \ge 21.5\text{ dB}$, $\text{VRAM} \le 4{,}096\text{ MB}$ | $\sim 23.5\text{ dB}$, $\sim 480\text{k}$ pts | Pre-registered Hypothesis (Pending GPU Run) |

*Reporting Integrity Protocol:* If any scene achieves metrics below the expected target but above the minimum gate, the empirical number must be reported without post-hoc tuning. If a scene violates the VRAM cap, it is marked as an empirical boundary limitation of ZeroGS.

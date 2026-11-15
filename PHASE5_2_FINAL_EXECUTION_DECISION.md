# Phase 5.2 Final Execution Decision & Pilot Readiness

**Document ID:** `PHASE5_2_FINAL_EXECUTION_DECISION`  
**Date:** November 15, 2026  
**Auditor:** Senior Research Engineer, Experimental Design Auditor, & GPU Systems Engineer  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Execution Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13)  

---

## 1. Verified Facts (Summary Evidence Table)

The following claims, implementations, and constraints have been verified through forensic static inspection, mathematical analysis, and CLI parser preflight:

| Domain | Claim / Component | Verified Finding | Evidence Source |
| :--- | :--- | :--- | :--- |
| **Hardware & Environment** | GPU & Subsystem | RTX 5060 Laptop GPU (8,151 MiB VRAM), WSL2 Linux (Kernel 6.6), CUDA 12.8, GCC 13. Smart App Control bypassed safely via Linux binary compilation. | `nvidia-smi`, WSL2 audit logs |
| **Camera VRAM (Wall 1)** | `--data_device cpu` | Pageable host memory allocation for 185 cameras ($1297 \times 840$). Drops initial GPU camera lock from $3{,}075.47\text{ MiB}$ ($3{,}224.86\text{ MB}$) to $< 17.5\text{ MB}$. | `scene/__init__.py:84-87`, `scene/dataset_readers.py` |
| **VRAM Accounting** | Compound Footprint | Footprint is defined as $\max(\text{allocated}, \text{reserved}) + 256.0\text{ MiB allowance}$. Controller ceiling is $4{,}096.0\text{ MiB}$ ($3{,}840.0\text{ MiB}$ PyTorch limit). Units are strictly binary MiB ($1\text{ MiB} = 1{,}048{,}576\text{ bytes}$). | `zerogs/memory_telemetry.py:41-45`, `zerogs/admission_controller.py:180` |
| **Optimizer Stability** | `Int8GaussianAdam` | Variance quantization uses dynamic logarithmic scaling ($[\exp(-36), \exp(4)]$). While $v_{\min} = 2.3195 \times 10^{-16} < 10^{-15}$, $\sqrt{v_{\min}} = 1.523 \times 10^{-8} \gg 10^{-15}$, dominating the denominator and mathematically preventing underflow division by zero. | `zerogs/optimizers.py:15-32`, KaTeX proof in Deliverable 4 |
| **Causal Isolation** | Arm D vs. Arm E | Complete parity in dataset, resolution, random seed, camera device (`cpu`), admission controller hyperparameters, and pruning mode (`legacy_as_is`). The **sole independent variable** is `--optimizer_type default` (FP32) vs. `--optimizer_type int8`. | `arguments/__init__.py`, `train.py`, Deliverable 2 diff table |
| **CLI Argument Parsing** | All Arms (A through E) | Static parser check confirms zero syntax errors, valid argument types, correct flags (`--disable_zero_oom` vs. `--enable_zero_oom`), and zero path collisions across all 5 arms. | `PHASE5_2_CLI_VALIDATION.md` |
| **Watchdog Intercept** | Monitoring Unconstrained Arms | Unconstrained arms (Arms A, B, C) lack an in-process tripwire when running on an 8GB GPU. A CPU-side watchdog polling `telemetry.jsonl` at $0.25\text{ s}$ intervals is specified to issue `SIGTERM` upon detecting footprint $> 4{,}096.0\text{ MiB}$. | `PHASE5_2_MONITOR_SAFETY_AUDIT.md` |

---

## 2. Unresolved Uncertainties

1. **Exact Step of Vanilla 4GB Breach ($T_{\text{breach}}$):**  
   Prior Phase 4/5 logs show Vanilla 3DGS under `--data_device cpu` reaching step 3,000 with $\sim 191\text{k}$ primitives and $\sim 3{,}094\text{ MiB}$ footprint. It is uncertain whether Arm B will breach $4{,}096\text{ MiB}$ before step 3,000 under seed 0 or survive into the $5{,}000\text{--}15{,}000$ densification window. The staged pilot is specifically designed to resolve this uncertainty empirically.
2. **First Observed ($T_{\text{obs}}$) vs. True First Violation ($T_{\text{true}}$):**  
   Because Vanilla 3DGS writes mutation telemetry at 100-step intervals (`densification_interval = 100`), any breach will be recorded at an upper-bound iteration $T_{\text{obs}} \in [T_{\text{true}}, T_{\text{true}} + 100)$.
3. **Background Display Server Jitter:**  
   Windows DWM / WSLg display server footprint fluctuates dynamically between $200\text{--}400\text{ MiB}$. While the software controller uses a fixed $256.0\text{ MiB}$ allowance, physical process memory measured via `nvidia-smi` may experience $\pm 50\text{ MiB}$ fluctuations independent of the algorithm.

---

## 3. Remaining Blockers

- **Algorithmic / Code Blockers:** **NONE.** All arguments, parameters, and paths are validated.
- **Operational Blocker:** The standalone watchdog daemon (`scripts/vram_watchdog.py`) must be written to disk and verified on CPU prior to launching unconstrained GPU processes (Arm B).
- **Execution Clearance:** Strict execution guidelines require static validation mode until explicit user authorization to execute GPU compute is received.

---

## 4. Final Execution Verdict

### Verdict: **B. READY CONDITIONAL**

The ZeroGS experiment suite is technically and scientifically ready for execution, subject to the following **three mandatory pre-execution conditions**:

1. **Watchdog Deployment:** Instantiate `scripts/vram_watchdog.py` on disk in accordance with the specification in `PHASE5_2_MONITOR_SAFETY_AUDIT.md`.
2. **Environment Verification:** Confirm execution occurs inside the established WSL2 Linux environment (`cuda_env`) where CUDA 12.8 and compiled extensions (`diff_gaussian_rasterization`, `simple_knn`) are functional.
3. **User Clearance:** Receive explicit authorization to transition from static audit mode to GPU execution.

Upon satisfaction of these conditions, the status transitions immediately to **`A. READY FOR CONTROLLED PILOT`**.

---

## 5. Approved First Pilot Command (Arm B: Vanilla 3DGS + CPU Cameras)

The approved pilot tests the hypothesis: *Does offloading cameras to host CPU memory allow standard 3D Gaussian Splatting to complete initial densification (3,000 steps) within a 4GB VRAM envelope, or does unconstrained Gaussian densification trigger OOM?*

### Command Line Execution:

```bash
# 1. Launch Training Process in Background (WSL2)
python train.py \
    -s data/garden \
    -i images_4 \
    -m experiments/results/ablation_4gb/arm_b_vanilla_cpu \
    --eval \
    --seed 0 \
    --data_device cpu \
    --disable_zero_oom \
    --iterations 3000 \
    --save_iterations 1000 3000 \
    --test_iterations 3000 \
    --telemetry_jsonl experiments/results/ablation_4gb/arm_b_vanilla_cpu/telemetry.jsonl \
    > experiments/results/ablation_4gb/arm_b_vanilla_cpu/train.log 2>&1 &

TRAIN_PID=$!

# 2. Launch CPU Watchdog Daemon
python scripts/vram_watchdog.py \
    --pid $TRAIN_PID \
    --telemetry_path experiments/results/ablation_4gb/arm_b_vanilla_cpu/telemetry.jsonl \
    --limit_mb 4096.0 \
    --poll_interval 0.25 \
    --output_log experiments/results/ablation_4gb/arm_b_vanilla_cpu/watchdog_intercept.json
```

---

## 6. Expected Output Files & Directory Structure

```
experiments/results/ablation_4gb/arm_b_vanilla_cpu/
├── cfg_args
├── input.ply
├── train.log
├── telemetry.jsonl
├── point_cloud/
│   ├── iteration_1000/
│   │   └── point_cloud.ply
│   └── iteration_3000/
│       └── point_cloud.ply
├── chkpnt1000.pth
├── chkpnt3000.pth                (Only if survived to 3,000 steps)
├── test/
│   └── ours_3000/
│       ├── gt/                   (24 test ground truth renders)
│       ├── renders/              (24 test synthesized renders)
│       └── results.json          (PSNR, SSIM, LPIPS metrics)
└── watchdog_intercept.json       (Only generated if 4096 MiB breached)
```

---

## 7. Estimated Runtime & Resource Envelope

- **Target Device:** NVIDIA GeForce RTX 5060 Laptop GPU (compute capability sm_90 / sm_89 compatible).
- **Subsystem:** WSL2 Linux (CUDA 12.8, GCC 13).
- **Estimated Runtime:** **5.5 to 8.0 minutes** for 3,000 iterations at $1297 \times 840$ resolution.
- **VRAM Resource Profile:**
  - *Step 0 (Initialization):* $\approx 1{,}150\text{ MiB}$ (PyTorch context + 138k initial primitives).
  - *Step 1,000:* $\approx 1{,}800\text{--}2{,}200\text{ MiB}$.
  - *Step 3,000:* $\approx 2{,}900\text{--}3{,}500\text{ MiB}$ (projected $\sim 190\text{k}\text{--}220\text{k}$ primitives).
- **Disk Footprint:** $\approx 450\text{ MB}$ (checkpoints + test renders).

---

## 8. Concrete Stop & Termination Conditions

| Condition | Observable Event | Action Taken | Scientific Conclusion |
| :--- | :--- | :--- | :--- |
| **SUCCESS (Survives 3k)** | Process completes step 3,000 with exit code 0; `chkpnt3000.pth` and `results.json` exist; peak footprint $< 4{,}096.0\text{ MiB}$. | Log completion; compute 3k metrics; evaluate decision tree for Stage 2.2 extension to 7k/15k. | Proves Vanilla 3DGS survives initial densification under CPU camera offloading; establishes valid baseline checkpoint for step-matched comparisons. |
| **TRIPWIRE (4GB Breach)** | Watchdog detects record with $\max(\text{reserved}, \text{footprint}) > 4{,}096.0\text{ MiB}$. | Watchdog sends `SIGTERM` to PID; writes `watchdog_intercept.json`; marks Arm B as "Terminated at step $T_{\text{obs}}$". | Decisively proves Vanilla 3DGS exceeds 4GB budget during densification even with camera offloading. Ablation Arm B complete. |
| **FAILURE / ABORT** | Process crashes with CUDA OOM, segmentation fault, or unhandled exception prior to 3k without watchdog trigger. | Abort execution; inspect `train.log` and system dmesg; preserve core dumps. | Engineering defect or hardware fault. Must be diagnosed before continuing. |

---

## 9. Scope of Scientific Claims

### What the 3,000-Step Pilot Proves:
1. **Validates Host-Camera Streaming (Wall 1):** Proves whether `--data_device cpu` eliminates the $3{,}225\text{ MB}$ camera barrier in clean isolation.
2. **Establishes Empirical Memory Growth Rate:** Quantifies the precise byte-per-iteration memory slope of unconstrained 3DGS densification.
3. **Produces Step-Matched Baseline Checkpoint:** If it survives, provides a verified, unpruned 3k baseline to compare directly with ZeroGS 3k checkpoints.

### What the Pilot CANNOT Prove:
1. **Cannot Prove 30k Convergence:** A 3k checkpoint cannot demonstrate final scene fidelity or prove that Vanilla would survive to 30k.
2. **Cannot Claim Final Publication Superiority:** PSNR at 3k is an early intermediate checkpoint and cannot be compared against 30,000-step converged models from literature.

---

## 10. Sequence of Experiments After Pilot

```
[Pilot Run: Arm B (0 -> 3,000 steps)]
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
[Arm B Breaches 4GB]          [Arm B Survives 3k]
  │                             │
  ├─────────────────────────────┼─────────────────────────────┐
  ▼                             ▼                             ▼
[Launch Arm A]                [Stage 2.2 Extension]          [Launch Arm D & Arm E]
Run steps 0 -> 605            Extend Arm B to 7k/15k         Head-to-head 30k comparison
to prove static camera        to locate exact               under 4096 MiB budget
VRAM lock (Wall 1 OOM).       densification breach.          (FP32 vs INT8 optimizer).
```

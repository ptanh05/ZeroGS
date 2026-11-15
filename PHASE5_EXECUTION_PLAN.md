# Phase 5 Prioritized Execution Plan: Controlled GPU Pilot & Validation

**Document Target:** Concrete, executable, and prioritized roadmap for GPU experiment execution following preflight verification.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Hardware Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13) on NVIDIA RTX 5060 Laptop GPU  
**Execution Policy:** Statically verified commands only (Do NOT execute until authorized)  

---

## 1. The Recommended First Controlled GPU Pilot

### **Pilot Target:** **Arm B — Camera-Offloaded Vanilla 3DGS on Garden (3,000 Steps)**

### Why Arm B is the Single Most Critical Pilot:
1. **Resolves the Core Baseline Dispute:**  
   Prior reports asserted Vanilla 3DGS OOMed at Step 605 due to camera memory lock (Wall 1). Arm B gives Vanilla 3DGS `--data_device cpu`, eliminating Wall 1. It directly answers the reviewer's central question: *"Does Vanilla 3DGS survive on 4GB once you give it CPU camera streaming?"*
2. **Quantifies Unconstrained Densification Growth:**  
   By running unconstrained Vanilla with CPU camera streaming to step 3,000, we observe the exact primitive inflation curve and determine whether Wall 2 is breached before or after step 3,000.
3. **Safe Execution Window:**  
   A 3,000-step run takes only $\approx 5\text{--}8\text{ minutes}$ in WSL2 ($10.5\text{ it/s}$), providing immediate empirical validation with minimal compute consumption.
4. **Validates Telemetry & Ingestion Pipelines:**  
   Verifies that `telemetry.jsonl`, `cameras.json`, and `metrics_3000.json` generate without error under clean WSL2 execution.

---

## 2. Statically Verified Command Suite (Ablation Arms A – E)

All commands have been audited to ensure every argument is supported by `train.py` and `arguments/__init__.py`:

### Arm A: Stock Vanilla 3DGS (Confirm Wall 1 Failure at Step $\le 605$)
```bash
python train.py \
    -s data/garden \
    -i images_4 \
    -m experiments/results/ablation_4gb/arm_a_vanilla_cuda \
    --eval \
    --seed 0 \
    --data_device cuda \
    --disable_zero_oom \
    --iterations 1000 \
    --save_iterations 605 1000 \
    --test_iterations 1000 \
    --telemetry_jsonl experiments/results/ablation_4gb/arm_a_vanilla_cuda/telemetry.jsonl
```
*Expected Outcome:* Telemetry captures Reserved VRAM $> 4{,}096\text{ MB}$ within first 605 steps due to $3{,}225\text{ MB}$ static camera lock.

---

### Arm B: Camera-Offloaded Vanilla 3DGS (First Controlled Pilot)
```bash
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
    --telemetry_jsonl experiments/results/ablation_4gb/arm_b_vanilla_cpu/telemetry.jsonl
```
*Expected Outcome:* Wall 1 eliminated ($< 42\text{ MB}$ camera VRAM). Primitives grow unconstrained. If memory breaches $4{,}096\text{ MB}$, record $T_{\text{breach}}$ and primitive count.

---

### Arm C: Optimizer Quantized Only (Int8GaussianAdam Isolated)
```bash
python train.py \
    -s data/garden \
    -i images_4 \
    -m experiments/results/ablation_4gb/arm_c_opt_int8_only \
    --eval \
    --seed 0 \
    --data_device cpu \
    --optimizer_type int8 \
    --disable_zero_oom \
    --iterations 3000 \
    --save_iterations 1000 3000 \
    --test_iterations 3000 \
    --telemetry_jsonl experiments/results/ablation_4gb/arm_c_opt_int8_only/telemetry.jsonl
```
*Expected Outcome:* Moment memory drops from $472\text{ B}$ to $118\text{ B/pt}$. Delays memory ceiling compared to Arm B.

---

### Arm D: Admission Controller Only (Admission Dynamics Isolated with FP32)
```bash
python train.py \
    -s data/garden \
    -i images_4 \
    -m experiments/results/ablation_4gb/arm_d_admit_fp32_only \
    --eval \
    --seed 0 \
    --data_device cpu \
    --hard_vram_limit_mb 4096.0 \
    --safety_headroom_mb 256.0 \
    --non_pytorch_allowance_mb 256.0 \
    --optimizer_type default \
    --allocator_strategy gain_per_byte \
    --inflow_mode dynamic \
    --iterations 3000 \
    --save_iterations 1000 3000 \
    --test_iterations 3000 \
    --telemetry_jsonl experiments/results/ablation_4gb/arm_d_admit_fp32_only/telemetry.jsonl
```
*Expected Outcome:* Memory strictly bounded $\le 4{,}096\text{ MB}$. Higher primitive cost ($944\text{ B}$ vs $590\text{ B}$) forces tighter primitive capping ($< 150\text{k}$ points at step 3k).

---

### Arm E: Full ZeroGS System
- Already executed and verified in Phase 4 (`experiments/results/garden_4gb/garden_zerogs_int8_full`).
- Replicated if needed using:
```bash
python train.py \
    -s data/garden \
    -i images_4 \
    -m experiments/results/ablation_4gb/arm_e_full_zerogs \
    --eval \
    --seed 0 \
    --data_device cpu \
    --hard_vram_limit_mb 4096.0 \
    --safety_headroom_mb 256.0 \
    --non_pytorch_allowance_mb 256.0 \
    --optimizer_type int8 \
    --allocator_strategy gain_per_byte \
    --inflow_mode dynamic \
    --iterations 3000 \
    --save_iterations 1000 3000 \
    --test_iterations 3000 \
    --telemetry_jsonl experiments/results/ablation_4gb/arm_e_full_zerogs/telemetry.jsonl
```

---

## 3. Phased Execution Roadmap

### Stage 1: Pre-Execution Prerequisites (No GPU Required)
1. Verify WSL2 environment symlinks:
   ```bash
   export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
   export CUDA_HOME=/home/theanh/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   ```
2. Unpack `bicycle` and `stump` from `data/360_v2.zip` into `data/bicycle` and `data/stump`.
3. Create output directory tree: `experiments/results/ablation_4gb/`.

### Stage 2: First Controlled Pilot Run (Estimated: 8 minutes)
- Execute **Arm B (3,000 steps)**.
- Audit output: check `telemetry.jsonl`, verify `chkpnt3000.pth`, inspect peak reserved memory and primitive count.

### Stage 3: Remaining Controlled Ablation Matrix (Estimated: 25 minutes)
- Execute Arm A (1,000 steps) to record Wall 1 breach.
- Execute Arm C (3,000 steps).
- Execute Arm D (3,000 steps).
- Compile comparative 3,000-step ablation table for the paper.

### Stage 4: Cross-Scene Generalization Trials (Estimated: ~3 hours total)
Prioritized order based on scientific significance:
1. **`flowers`** (173 cameras, delicate geometry, pre-extracted).
2. **`bicycle`** (194 cameras, unbounded 360, thin structures).
3. **`stump`** (125 cameras, natural irregular terrain).
4. **`bonsai`** (Full 30,000-step 4GB confirmatory run).

### Stage 5: Paper Integration & Final Compilation
1. Update `submission/main.tex`:
   - Incorporate 5-arm ablation matrix into Section 4.
   - Insert multi-scene 4GB benchmark table into Section 4.3.
   - Clarify $75\%$ optimizer vs $37.5\%$ model reduction in Abstract and Methodology.
   - Disclose limitations (RAM streaming, PCIe, WDDM allocator).
2. Compile finalized PDF: `python submission/compile_paper_pdf.py`.

---

## 4. Remaining User Decisions Required

Before launching Stage 2 (First Controlled Pilot), the user must decide:
1. **Ablation Horizon:** Confirm whether 3,000-step pilot horizon is acceptable for initial ablation comparison, or if Arms B–D should run directly to 30,000 steps. *(Recommendation: 3,000 steps first to catch early budget violations in $< 30\text{ min}$, then extend surviving configurations).*
2. **Authorization to Execute:** Explicitly provide the authorization command to launch the Arm B pilot script in WSL2.

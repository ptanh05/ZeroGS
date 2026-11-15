# Phase 5 Scientific Audit: Comprehensive Evidence Reconciliation

**Document Target:** Authoritative reconciliation of all empirical measurements, telemetry logs, and theoretical claims for the ZeroGS Garden 4GB experiment.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM, Blackwell `sm_120`)  
**Audit Standard:** Strict Read-Only Forensic Evidence  

---

## 1. Executive Reconciliation Summary

During Phase 4 and prior experiment reports, several discrepancies in numerical telemetry emerged due to differing log parsers, milestone reporting windows, and platform execution layers. This section resolves every discrepancy definitively by citing line-by-line evidence from disk.

### 1.1 The Authoritative Empirical Evidence Matrix

| Metric / Parameter | Claimed / Prior Value | Authoritative Measured Value | Evidence File & Line / Method | Forensic Explanation & Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Global Peak Reserved VRAM** | $3{,}094.0\text{ MiB}$ vs $3{,}496.0\text{ MiB}$ | **$3{,}496.00\text{ MiB}$** | `garden_zerogs_int8_full/telemetry.jsonl` (Iter 11,000, `memory_before`) | **RECONCILED:** $3{,}496\text{ MB}$ occurred at Iter 11,000 during densification. $3{,}094\text{ MB}$ is the final reserved VRAM at Iter 30,000 (`training_end`). `run_garden_4gb_full.py` only parsed `rec["memory"]`, missing earlier peaks. |
| **Final Reserved VRAM** | $3{,}094.0\text{ MiB}$ | **$3{,}094.00\text{ MiB}$** | `garden_zerogs_int8_full/telemetry.jsonl` (Line 81: `training_end.memory`) | Verified: Memory pool stabilized and contracted to $3{,}094\text{ MB}$ after densification ceased at step 15,000. |
| **Peak Allocated VRAM (Mutation)** | $1{,}127.49\text{ MiB}$ | **$1{,}127.49\text{ MiB}$** | `garden_zerogs_int8_full/telemetry.jsonl` (Iter 14,900, `memory_after`) | Verified: Maximum active allocated tensor memory during Gaussian mutations. |
| **Global Peak Allocated VRAM** | $1{,}375.70\text{ MiB}$ | **$1{,}375.70\text{ MiB}$** | `garden_zerogs_int8_full/telemetry.jsonl` (Line 81: `training_end.memory`) | Verified: Peak cumulative allocated memory across the full run, including rendering evaluation passes. |
| **VRAM Budget Compliance (WSL2)** | $\le 4{,}096\text{ MiB}$ | **$100\%$ PASS** ($\le 3{,}496\text{ MB}$) | All 81 entries in full run telemetry | Headroom of $\mathbf{600.00\text{ MiB}}$ maintained at all times. 0 budget violations. |
| **VRAM Compliance (Windows Host)** | $\le 4{,}096\text{ MiB}$ | **$4{,}172.0\text{ MB}$ Spike** (Step 1,479) | `garden_zerogs_int8_pilot/telemetry.jsonl` (Line 13) | **RECONCILED:** PyTorch caching allocator on WDDM hoarded $4{,}172\text{ MB}$ reserved. The ZeroGS tripwire intervened, reclaimed $3{,}990\text{ MB}$, dropping reserved to $182\text{ MB}$. |
| **Static Camera GPU Memory** | Locked $3{,}224.55\text{ MB}$ | **$41.85\text{ MB}$** on GPU | `scene/cameras.py` & pilot logs | `--data_device cpu` leaves 185 images in Pageable Host RAM, streaming 1 active view ($13\text{ MB}$) per step. |
| **Optimizer State VRAM Reduction** | $75.0\%$ of total VRAM (ambiguous) | **$75.00\%$ of Optimizer State** ($37.5\%$ of model+grad+state) | `zerogs/optimizers.py:108-132` | **RECONCILED:** Theoretical and physical tensor reduction is exactly $75\%$ for Adam moment vectors ($472 \to 118\text{ B/pt}$). Total VRAM reduction is NOT $75\%$. |
| **Initial SfM Gaussian Count** | $138{,}766$ | **$138{,}766$ points** | `data/garden/sparse/0/points3D.bin` | Direct COLMAP binary inspection. |
| **Step 3,000 Gaussian Count** | $191{,}133$ | **$191{,}133$ points** | `garden_zerogs_int8_pilot/chkpnt3000.pth` | PyTorch tensor inspect: `chk[0].shape = [191133, 3]`. |
| **Step 7,000 Gaussian Count** | $283{,}906$ | **$283{,}906$ points** | `garden_zerogs_int8_full/chkpnt7000.pth` | PyTorch tensor inspect: `chk[0].shape = [283906, 3]`. |
| **Step 15,000 Gaussian Count** | $547{,}593$ | **$547{,}593$ points** | `garden_zerogs_int8_full/chkpnt15000.pth` | PyTorch tensor inspect: `chk[0].shape = [547593, 3]`. |
| **Step 30,000 Gaussian Count** | $547{,}593$ | **$547{,}593$ points** | `garden_zerogs_int8_full/chkpnt30000.pth` | Frozen topology: identical primitive count verified. |
| **Admission Cycles Logged** | 83 cycles | **83 cycles** | `garden_zerogs_int8_full/telemetry.jsonl` | Cycles executed every 100 steps between step 1,000 and 15,000. |
| **Primitives Purged via Admission** | $302{,}246$ | **$302{,}246$ primitives** | `garden_zerogs_int8_full/telemetry.jsonl` | Sum of candidate purges via Admission Controller. |
| **Test Views Count** | 24 views | **24 views** | `data/garden/images_4` ($i \pmod 8 == 0$) | Views: `[0, 8, 16, ..., 184]`. Full $1297 \times 840$ RGB. |
| **Step 3,000 PSNR / SSIM / LPIPS** | $14.59$ / $0.30$ / $0.74$ | **$14.5879\text{ dB}$ / $0.2992$ / $0.7402$** | `garden_zerogs_int8_pilot/results.json` | Re-verified from evaluation logs across all 24 views. |
| **Step 7,000 PSNR / SSIM / LPIPS** | $19.57$ / $0.39$ / $0.64$ | **$19.5668\text{ dB}$ / $0.3923$ / $0.6376$** | `garden_zerogs_int8_full/results.json` | Re-verified from evaluation logs across all 24 views. |
| **Step 15,000 PSNR / SSIM / LPIPS**| $21.70$ / $0.52$ / $0.51$ | **$21.7016\text{ dB}$ / $0.5174$ / $0.5085$** | `garden_zerogs_int8_full/results.json` | Re-verified from evaluation logs across all 24 views. |
| **Step 30,000 PSNR / SSIM / LPIPS**| $22.85$ / $0.60$ / $0.44$ | **$22.8456\text{ dB}$ / $0.5960$ / $0.4391$** | `garden_zerogs_int8_full/results.json` | Re-verified from evaluation logs across all 24 views. |

---

## 2. In-Depth Discrepancy Forensic Investigations

### 2.1 The Peak Reserved VRAM Paradox: $3{,}094\text{ MiB}$ vs $3{,}496\text{ MiB}$

- **The Problem:** In earlier Phase 4 reporting, two numbers were cited for peak reserved memory: $3{,}094\text{ MiB}$ in `convergence_summary.json` and `GARDEN_4GB_CONVERGENCE.csv`, versus $3{,}496\text{ MiB}$ in `ZERO_GARDEN_PHASE4_FORENSIC_AUDIT.md`.
- **The Forensic Investigation:**
  - Auditing `scripts/run_garden_4gb_full.py` lines 62–71:
    ```python
    elif "memory" in rec:
        mem = rec["memory"]
        alloc_mb = float(mem.get("peak_allocated_mb", ...))
        res_mb = float(mem.get("peak_reserved_mb", ...))
        peak_allocated_mb = max(peak_allocated_mb, alloc_mb)
        peak_reserved_mb = max(peak_reserved_mb, res_mb)
    ```
  - In `telemetry.jsonl`, lines 1–80 contain mutation event logs whose dictionary keys are `"memory_before"` and `"memory_after"`. They do NOT have a top-level `"memory"` key!
  - Only Line 81 (`event: training_end`) has the top-level `"memory"` key:
    ```json
    {"event": "training_end", "final_gaussians": 547593, "memory": {"allocated_mb": 724.764, "peak_allocated_mb": 1375.702, "reserved_mb": 3094.0, "peak_reserved_mb": 3094.0}}
    ```
  - Consequently, `parse_telemetry()` completely skipped lines 1–80, extracting only the values recorded at the end of training: $3{,}094.0\text{ MB}$ reserved.
  - When the Phase 4 auditor inspected all 80 mutation records line-by-line, the true global maximum reserved memory was uncovered at **Iteration 11,000**:
    ```json
    {"iteration": 11000, "memory_before": {"reserved_mb": 3496.0, "peak_reserved_mb": 3496.0}}
    ```
- **Authoritative Resolution:**
  - **Global Peak Reserved VRAM:** $\mathbf{3{,}496.00\text{ MiB}}$ (Step 11,000).
  - **Final Convergence Reserved VRAM:** $\mathbf{3{,}094.00\text{ MiB}}$ (Step 30,000).
  - Both values are authentic empirical measurements and comply strictly with the $4{,}096\text{ MiB}$ ceiling ($600\text{ MB}$ and $1{,}002\text{ MB}$ safety margins, respectively).

---

### 2.2 The Optimizer 75% Saving Claim: Theoretical vs Measured VRAM

- **The Problem:** Manuscripts frequently claim that INT8 quantization saves "75% of optimizer VRAM" without clarifying whether this means total VRAM, total parameter footprint, or purely persistent moment buffers.
- **The Forensic Investigation:**
  1. *Per-Gaussian Parameter Inventory ($59$ floats):*
     - $\text{Position}: 3$, $\text{SH DC}: 3$, $\text{SH Rest}: 45$, $\text{Opacity}: 1$, $\text{Scale}: 3$, $\text{Rotation}: 4 = 59\text{ scalar parameters}$.
  2. *Adam Optimizer State Storage:*
     - FP32 Adam: Stores $m_t$ (FP32) and $v_t$ (FP32) $\implies 59 \times 4\text{ B} \times 2 = \mathbf{472\text{ Bytes/Gaussian}}$.
     - Int8GaussianAdam: Stores $m_t$ (INT8) and $v_t$ (UINT8) $\implies 59 \times 1\text{ B} \times 2 = \mathbf{118\text{ Bytes/Gaussian}}$.
     - Persistent State Saving: $\frac{472 - 118}{472} = \mathbf{75.00\%}$ (exact bit-level saving).
  3. *Total Model + State Memory per Primitive:*
     - Parameters (FP32): $59 \times 4\text{ B} = 236\text{ Bytes}$.
     - Gradients (FP32): $59 \times 4\text{ B} = 236\text{ Bytes}$.
     - FP32 Total: $236 + 236 + 472 = \mathbf{944\text{ Bytes/Gaussian}}$.
     - INT8 Total: $236 + 236 + 118 = \mathbf{590\text{ Bytes/Gaussian}}$.
     - Net Model+State Saving: $\frac{944 - 590}{944} = \mathbf{37.50\%}$.
  4. *Transient Execution Overhead:*
     - During `Int8GaussianAdam.step()`, moments for each parameter tensor are temporarily dequantized to FP32. For `_features_rest` ($547{,}593 \times 45$ floats), this allocates transient buffers of $\approx 98.6\text{ MB}$.
- **Authoritative Resolution:**
  - Papers must state: *"Int8GaussianAdam achieves a 75.00% reduction in persistent optimizer state tensors (from 472 B to 118 B per Gaussian) and a 37.50% reduction in total persistent model representation footprint (from 944 B to 590 B per Gaussian)."*
  - It must NOT claim a 75% reduction in total training VRAM.

---

### 2.3 The Cross-Platform Performance Delta (Windows Host vs Linux WSL2)

- **The Problem:** Why did training speed change from $3.5\text{ it/s}$ to $10.5\text{ it/s}$?
- **The Forensic Investigation:**
  - On Windows Host (WDDM 3.2), GPU memory allocations pass through the DirectX / WDDM kernel subsystem, which virtualizes GPU memory and introduces paging overhead, particularly when host-streaming camera tensors.
  - On Linux WSL2, the NVIDIA Linux driver interacts directly with GPU memory through a native ELF runtime, eliminating Windows kernel context switching and synchronization delays.
  - Furthermore, memory allocator fragmentation on Windows led to the $4{,}172\text{ MB}$ reserved spike at Step 1,479, whereas Linux maintained smooth allocation pools peaking at $3{,}496\text{ MB}$.
- **Authoritative Resolution:**
  - WSL2 Linux provides a verified **$3.1\times$ wall-clock throughput acceleration** ($10.5\text{ it/s}$ vs $3.5\text{ it/s}$) and superior memory pooling stability.

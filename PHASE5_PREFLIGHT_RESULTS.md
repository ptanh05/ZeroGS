# Phase 5 Preflight Inspection & Verification Results

**Document Target:** Static implementation preflight, CLI compatibility matrix, memory arithmetic validation, and publication claim cross-check.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Auditor:** Independent Scientific Audit Agent  
**Audit Standard:** Strict Static Code Inspection & File-Based Telemetry  

---

## 1. Budget Enforcement & Runtime Mechanism Audit

### 1.1 Code Implementation Analysis
In `train.py`, memory enforcement is dictated by `enable_zero_oom` (configured via `--enable_zero_oom` or `--disable_zero_oom`):

```python
# train.py:395-408
if enable_zero_oom and not is_simulation:
    try:
        admission_ctrl.assert_current_memory_within_budget(
            telemetry_writer=telemetry_writer,
            iteration=iteration,
        )
    except VramBudgetExceededError:
        if telemetry_writer is not None:
            telemetry_writer.write(
                "budget_exceeded",
                iteration=iteration,
                memory=capture_cuda_memory().to_dict(),
            )
        raise
```

In `zerogs/admission_controller.py:218-228`:
```python
if current_footprint_bytes > self.hard_vram_limit_bytes:
    raise VramBudgetExceededError(
        f"Peak VRAM footprint ({current_footprint_mb:.2f}MB) exceeded hard budget "
        f"({self.hard_vram_limit_mb:.2f}MB)."
    )
```

### 1.2 Behavior by Experimental Arm

| Arm | CLI Flag | `enable_zero_oom` | Assertion Called? | Outcome on 8GB Physical GPU |
| :---: | :---: | :---: | :---: | :--- |
| **Arm A** | `--disable_zero_oom` | `False` | No | Free-running on GPU; static cameras ($3.2\text{ GB}$) + densification will exceed 4GB immediately, running up to $\sim 6\text{--}7\text{ GB}$ unless terminated. |
| **Arm B** | `--disable_zero_oom` | `False` | No | Free-running on GPU; unconstrained densification exceeds 4GB around step $5\text{k}\text{--}8\text{k}$, running up to $\sim 7\text{ GB}$ unless terminated. |
| **Arm C** | `--disable_zero_oom` | `False` | No | Free-running on GPU; unconstrained densification exceeds 4GB around step $8\text{k}\text{--}12\text{k}$, running up to $\sim 6.5\text{ GB}$ unless terminated. |
| **Arm D** | `--hard_vram_limit_mb 4096.0` | `True` | Yes (every step) | Enforces 4GB ceiling. Raises `VramBudgetExceededError` if memory exceeds 4,096 MiB. |
| **Arm E** | `--hard_vram_limit_mb 4096.0` | `True` | Yes (every step) | Enforces 4GB ceiling. Maintained $3{,}496\text{ MB}$ max reserved across 30k steps. |

> [!CAUTION]
> **Definitive Finding:**  
> On an 8GB physical GPU, passing `--disable_zero_oom` disables software tripwires. Arms A, B, and C will NOT crash at 4,096 MiB. They will continue allocating memory up to 8,151 MiB.  
> Therefore, an external telemetry monitoring script or threshold log parser must monitor `telemetry.jsonl` to detect $T_{\text{breach}}$ and terminate the process cleanly, recording $T_{\text{breach}}$ as the budget failure point.

---

## 2. Command Compatibility & CLI Argument Audit

### 2.1 Critical Argument Compatibility Matrix

| Proposed Argument | Defined in Code? | Defining File & Line | Accepted Types / Values | Status & Remarks |
| :--- | :---: | :--- | :--- | :--- |
| `-s`, `--source_path` | Yes | `arguments/__init__.py:81` | Path string | **VALID** |
| `-i`, `--images` | Yes | `arguments/__init__.py:89` | String (`images_4`, `images`) | **VALID** |
| `-m`, `--model_path` | Yes | `arguments/__init__.py:82` | Output directory string | **VALID** |
| `--eval` | Yes | `arguments/__init__.py:94` | Flag | **VALID** |
| `--seed` | Yes | `train.py:1028` | Integer (e.g. `0`) | **VALID** |
| `--data_device` | Yes | `arguments/__init__.py:93` | `"cuda"`, `"cpu"` | **VALID** (Controls host camera offloading) |
| `--optimizer_type` | Yes | `arguments/__init__.py:142` | `"default"`, `"int8"`, `"fp16"` | **VALID** (Instantiates `Int8GaussianAdam` if `"int8"`) |
| `--hard_vram_limit_mb` | Yes | `train.py:990` | Float (e.g. `4096.0`) | **VALID** |
| `--memory_budget_mb` | **NO** | *Non-existent in repo* | N/A | **INVALID BUG:** Replaced by `--hard_vram_limit_mb`. |
| `--disable_zero_oom` | Yes | `train.py:1025` | Flag | **VALID** (Sets `enable_zero_oom = False`) |
| `--allocator_strategy` | Yes | `train.py:1008` | `"gain_per_byte"`, `"demand_only"` | **VALID** |
| `--inflow_mode` | Yes | `arguments/__init__.py:171` | `"dynamic"`, `"static"` | **VALID** |
| `--save_iterations` | Yes | `train.py:984` | List of ints (`7000 15000 30000`) | **VALID** |
| `--test_iterations` | Yes | `train.py:983` | List of ints (`7000 15000 30000`) | **VALID** |
| `run_generalization_scene.py`| **NO** | *Non-existent in repo* | N/A | **INVALID BUG:** Must invoke `train.py` directly. |

---

## 3. Mathematical & Memory Accounting Audit

### 3.1 Bit-Level Optimizer Memory Proof
- **Scalar parameters per Gaussian ($D = 59$):**
  $$\text{XYZ}(3) + \text{Color}(3) + \text{SH Rest}(45) + \text{Opacity}(1) + \text{Scale}(3) + \text{Rotation}(4) = 59$$
- **Moment vectors in standard FP32 Adam:**
  $$\text{State}_{\text{FP32}} = 59 \times 4\text{ Bytes} \times 2\text{ moments} = \mathbf{472\text{ Bytes/Gaussian}}$$
- **Moment vectors in Int8GaussianAdam:**
  $$\text{State}_{\text{INT8}} = 59 \times 1\text{ Byte } (m_t) + 59 \times 1\text{ Byte } (v_t) = \mathbf{118\text{ Bytes/Gaussian}}$$
- **Exact Optimizer State Reduction:**
  $$\frac{472 - 118}{472} = \mathbf{75.00\%}$$
- **Total Persistent Model Representation Footprint:**
  - Parameters (FP32): $59 \times 4\text{ B} = 236\text{ Bytes}$
  - Gradients (FP32): $59 \times 4\text{ B} = 236\text{ Bytes}$
  - Total FP32 Model+Grad+State: $236 + 236 + 472 = \mathbf{944\text{ Bytes/Gaussian}}$
  - Total INT8 Model+Grad+State: $236 + 236 + 118 = \mathbf{590\text{ Bytes/Gaussian}}$
- **Exact Persistent Model Reduction:**
  $$\frac{944 - 590}{944} = \mathbf{37.50\%}$$

---

### 3.2 Denominator & Numerical Stability Correction
The logarithmic dynamic range covers $\log(v) \in [-36.0, 4.0]$.

**Correction of Previous Mathematical Misstatement:**
- Minimum positive variance code ($q_v = 1$):
  $$v_{\min} = \exp(-36.0) \approx 2.31952 \times 10^{-16}$$
- Comparing $v_{\min}$ with $\epsilon = 10^{-15}$:
  $$2.31952 \times 10^{-16} < 10^{-15}$$
  *(Previous documentation incorrectly stated that $v_{\min} > 10^{-15}$.)*
- **Why Underflow is Prevented:**
  In the Adam denominator (`zerogs/optimizers.py:185`):
  $$\text{denom} = \frac{\sqrt{v_t}}{\sqrt{1 - \beta_2^t}} + \epsilon$$
  The parameter update is scaled by the **standard deviation $\sqrt{v_t}$**, not by $v_t$!
  For $q_v = 1$:
  $$\sqrt{v_{\min}} = \sqrt{\exp(-36.0)} = \exp(-18.0) \approx \mathbf{1.523 \times 10^{-8}}$$
  Notice that:
  $$\sqrt{v_{\min}} = 1.523 \times 10^{-8} \gg \epsilon = 10^{-15}$$
  **Result:** The standard deviation $\sqrt{v_{\min}}$ is **$7$ orders of magnitude larger than $\epsilon$**!  
  Therefore, the denominator is dominated by $\sqrt{v_t} \sim 1.5 \times 10^{-8}$ rather than collapsing to $\epsilon = 10^{-15}$. Update magnitudes remain bounded to $\Delta \theta \sim 10^{-2}$ instead of exploding to $10^9$.

---

### 3.3 Transient Dequantization Memory During `step()`
- PyTorch sequentially iterates over parameter tensors during `Int8GaussianAdam.step()`.
- Largest tensor: `_features_rest` ($N \times 45$ floats).
- At $N = 547{,}593$ primitives:
  $$\text{Single FP32 Buffer} = 547{,}593 \times 45 \times 4\text{ Bytes} = \mathbf{94.00\text{ MiB}}$$
- With intermediate calculations (`m`, `v`, `denom`), peak transient memory reaches $\approx \mathbf{188\text{--}280\text{ MiB}}$ before PyTorch reclaims it.
- **Scientific Conclusion:** The Admission Controller's `safety_headroom_mb = 256.0` is physically required to prevent this transient spike from triggering CUDA OOM under a 4GB budget.

---

### 3.4 Camera Memory Accounting (Exact File-Level Verification)

Because `scene/__init__.py:84-87` instantiates both `train_cameras` and `test_cameras` during scene initialization, all cameras are allocated on `data_device`:

| Scene | Total Cams | Train Cams | Test Cams | Resolution ($W \times H$) | Bytes / Cam (RGB+Alpha) | Total Camera VRAM on CUDA (MiB) | Total Camera VRAM on CUDA (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`garden`** | $185$ | $161$ | $24$ | $1297 \times 840$ | $17{,}431{,}680\text{ B}$ | **$3{,}075.47\text{ MiB}$** | **$3{,}224.86\text{ MB}$** |
| **`bonsai`** | $292$ | $255$ | $37$ | $780 \times 520$ | $6{,}489{,}600\text{ B}$ | **$1{,}807.18\text{ MiB}$** | **$1{,}894.96\text{ MB}$** |
| **`flowers`** | $173$ | $151$ | $22$ | $1256 \times 828$ | $16{,}639{,}488\text{ B}$ | **$2{,}745.28\text{ MiB}$** | **$2{,}878.63\text{ MB}$** |
| **`bicycle`** | $194$ | $169$ | $25$ | $1237 \times 822$ | $16{,}269{,}024\text{ B}$ | **$3{,}009.98\text{ MiB}$** | **$3{,}156.19\text{ MB}$** |
| **`stump`** | $125$ | $109$ | $16$ | $1245 \times 825$ | $16{,}434{,}000\text{ B}$ | **$1{,}959.09\text{ MiB}$** | **$2{,}054.25\text{ MB}$** |

*Correction on Flowers:* Prior estimates assumed `flowers` had resolution $1297 \times 840$ ($3{,}015\text{ MiB}$). Actual files on disk are $1256 \times 828$, yielding $2{,}745.28\text{ MiB}$ ($2{,}878.63\text{ MB}$).

---

## 4. Publication Claim Cross-Check (`submission/main.tex`)

| Manuscript Section & Line | Stated Claim in Paper | Forensic Audit Classification | Specific Correction Required |
| :--- | :--- | :---: | :--- |
| **Abstract (Line 36)** | 8-bit optimizer achieves 51.6% lower VRAM and 69.6% fewer primitives. | **CONFOUNDED** | Attribute to the integrated framework (Admission Controller + Inflow Governor + INT8 Optimizer), not optimizer alone. |
| **Intro (Line 65)** | 18 live Prune-First interventions (437k evicted primitives). | **CONDITIONALLY VERIFIED** | Valid for `bonsai`. Clarify that on `garden`, 3 interventions evicted 302,246 primitives. |
| **Method (Line 132)** | Reducing persistent memory to 590 B/Gaussian (a 75% reduction). | **CONTRADICTED** | Correct arithmetic: 75.00% reduction for optimizer states ($472 \to 118\text{ B}$); 37.50% reduction for total model ($944 \to 590\text{ B}$). |
| **Results (Sec 4.3)** | 4GB Ultra-Low Regime. | **CONDITIONALLY VERIFIED** | Currently only reports `bonsai` (29.10 dB). Must insert `garden` 4GB trajectory (22.85 dB, 548k pts, 3,496 MB VRAM). |
| **Results (Table 1 & 4)** | Vanilla 3DGS OOM comparison on outdoor scenes. | **UNSUPPORTED** | Disclose that default Vanilla fails due to Wall 1 (Camera Storage $>3.2\text{ GB}$), while Wall 2 is solved by Admission Control. |
| **Discussion (Sec 5)** | System limitations. | **NOT TESTED** | Must disclose: (1) Host RAM requirements ($>8\text{ GB}$), (2) PCIe 4.0 transfer speed, (3) Windows WDDM allocator fragmentation. |

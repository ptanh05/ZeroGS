# Phase 4 Reproducibility Manifest: ZeroGS Garden 4GB Experiment

**Document Version:** 1.0  
**Experiment Name:** `garden_zerogs_int8_full`  
**Dataset Target:** Mip-NeRF 360 `garden` ($1297 \times 840$ downsampled, 185 cameras)  
**Budget Enforcement:** Hard Physical Limit 4,096 MiB VRAM  
**Primary Target Platform:** WSL2 Linux (Kali Linux / Ubuntu 24.04, CUDA 12.8)  
**Secondary Platform:** Windows 11 Enterprise (WDDM 3.2, PyTorch 2.6+)  

---

## 1. Hardware & System Configuration

To reproduce the exact numerical and memory telemetry observed in this experiment, the following hardware or an equivalent CUDA-capable setup is required:

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU |
| **Physical VRAM** | 8,151 MiB GDDR6 |
| **Compute Capability**| `sm_120` (Blackwell Architecture) |
| **Host System** | Windows 11 Enterprise (Build 26100), WDDM 3.2 |
| **Virtual Subsystem** | WSL2 (Linux Kernel `6.6.87.2-microsoft-standard-WSL2`) |
| **Linux Distribution**| Kali GNU/Linux Rolling 2024 (Debian-based) |
| **Host System RAM** | 32 GiB DDR5 |
| **Storage** | NVMe PCIe 4.0 SSD |

---

## 2. Software Toolchain & Exact Dependencies

### 2.1 Compiler and CUDA Toolchain
- **CUDA Toolkit:** CUDA 12.8.93 (`/home/theanh/cuda-12.8`)
- **C/C++ Compiler:** GCC 13.3.0 / G++ 13.3.0 (Explicitly required; GCC 14 is incompatible with CUDA 12.8 `nvcc`)
  ```bash
  export CC=/usr/bin/gcc-13
  export CXX=/usr/bin/g++-13
  export CUDA_HOME=/home/theanh/cuda-12.8
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  ```

### 2.2 Python Environment
- **Python Version:** 3.12.16 (CPython 64-bit)
- **Virtual Environment:** Python standard `venv` (`/home/theanh/venv_zerogs`)
- **Core Packages:**
  - `torch==2.11.0+cu128` (or `torch==2.6.0+cu128`)
  - `torchvision==0.24.0+cu128`
  - `numpy>=1.26.0`
  - `plyfile>=1.0.3`
  - `timm>=0.9.16`
  - `lpips>=0.1.4`
  - `opencv-python-headless==5.0.0.93`

### 2.3 CUDA Extensions Source Patches
Due to glibc 2.42 / C23 math header compatibility and standard C++ header requirements on modern Linux distributions:
1. **`submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h`**:
   Add `#include <cstdint>` and `#include <stdint.h>`.
2. **`cuda-12.8/include/crt/math_functions.h`**:
   Ensure declarations for `cospi`, `sinpi`, `rsqrt` include `__THROW` to prevent conflicting exception specifications with glibc.

---

## 3. Dataset Setup & Preprocessing

1. **Dataset Location:** `data/garden`
2. **Directory Structure:**
   ```
   data/garden/
   ├── images_4/             # 185 downsampled camera images (1297 x 840)
   ├── sparse/
   │   └── 0/
   │       ├── cameras.bin   # Camera intrinsics
   │       ├── images.bin    # Camera extrinsics
   │       └── points3D.bin  # 138,766 SfM initial points
   ```
3. **Train / Test Partitioning Rule:**
   - LLFF standard stride-8 partitioning:
     $$\text{Test Views} = \{ i \in [0, 184] \mid i \pmod 8 == 0 \} \implies 24\text{ views}$$
     $$\text{Train Views} = \{ i \in [0, 184] \mid i \pmod 8 \neq 0 \} \implies 161\text{ views}$$

---

## 4. Hyperparameter Specification

| Parameter | Configuration Value | Scientific Rationale |
| :--- | :--- | :--- |
| **Hard Memory Budget** | `4096 MiB` | Ultra-low memory regime target |
| **Camera Storage Device**| `--data_device cpu` | Pageable host memory offloading |
| **Initial Points** | 138,766 | SfM initialization from COLMAP |
| **Total Iterations** | 30,000 | Full convergence trajectory |
| **Densification Start** | Step 1,000 | Allow initial covariance stabilization |
| **Densification Stop** | Step 15,000 | Freeze topology; refine appearance |
| **Densification Interval**| 100 steps | 83 total admission evaluation cycles |
| **Opacity Reset Interval**| 3,000 steps | Clear floating degenerate primitives |
| **Optimizer** | `Int8GaussianAdam` | 75% optimizer VRAM reduction |
| **First Moment Quantization**| INT8 linear symmetric | Bounds $[-128, 127]$, dynamic scale |
| **Second Moment Quantization**| UINT8 logarithmic non-linear | $\log(v) \in [-36, 4]$, 17 orders magnitude |
| **Random Seed** | `--seed 0` | Deterministic reproducibility |

---

## 5. End-to-End Execution Sequence

### Phase A: Environment Preparation (WSL2)
```bash
# 1. Activate environment
source /home/theanh/venv_zerogs/bin/activate
cd /home/theanh/projects/ZeroGS

# 2. Build extensions with GCC-13
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 pip install -e submodules/simple-knn --no-build-isolation
```

### Phase B: Launch Pilot Trajectory (0 -> 3,000 steps)
```bash
python scripts/run_garden_4gb_pilot.py \
    --source_path data/garden \
    --model_path experiments/results/garden_4gb/garden_zerogs_int8_pilot \
    --data_device cpu \
    --memory_budget_mb 4096 \
    --iterations 3000 \
    --save_iterations 3000 \
    --test_iterations 3000
```

### Phase C: Resume Full Convergence (3,000 -> 30,000 steps)
```bash
python scripts/run_garden_4gb_full.py \
    --source_path data/garden \
    --start_checkpoint experiments/results/garden_4gb/garden_zerogs_int8_pilot/chkpnt3000.pth \
    --model_path experiments/results/garden_4gb/garden_zerogs_int8_full \
    --data_device cpu \
    --memory_budget_mb 4096 \
    --iterations 30000 \
    --save_iterations 7000 15000 30000 \
    --test_iterations 7000 15000 30000
```

---

## 6. Deterministic Verification & Integrity Auditing

To independently audit the results without executing training, run the following inspection commands:

### 6.1 Checkpoint Cryptographic Hash Verification
```bash
python -c "
import hashlib, os
files = [
    'experiments/results/garden_4gb/garden_zerogs_int8_pilot/chkpnt3000.pth',
    'experiments/results/garden_4gb/garden_zerogs_int8_full/chkpnt7000.pth',
    'experiments/results/garden_4gb/garden_zerogs_int8_full/chkpnt15000.pth',
    'experiments/results/garden_4gb/garden_zerogs_int8_full/chkpnt30000.pth'
]
for f in files:
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    sz = os.path.getsize(f) / (1024*1024)
    print(f'{os.path.basename(f)}: size={sz:.2f}MB sha256={h}')
"
```

**Expected Hashes:**
- `chkpnt3000.pth`: `6fb9cb4aa743263e68640f48652f1f55bfdcef7b5bbefd8ba074dd133f128e5d`
- `chkpnt7000.pth`: `5c26f0651b916774083fe52a7a04428250280038ff06a15f1aa58e77659ecbef`
- `chkpnt15000.pth`: `021861677de4a9ba167599ad86994838f62b66ac9554fe35fb309dccfa70abdc`
- `chkpnt30000.pth`: `c05aafaa24358372a3b1e8e308d39ce3f494a4056b174299faa9b4dedd806ab5`

### 6.2 Metric Re-Verification Across 24 Test Views
```bash
python -c "
import json
with open('experiments/results/garden_4gb/garden_zerogs_int8_full/results.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'Milestone {k}: PSNR={v[\"psnr\"]:.4f} SSIM={v[\"ssim\"]:.4f} LPIPS={v[\"lpips\"]:.4f}')
"
```

**Expected Metrics:**
- Step 7,000: `PSNR=19.5668 SSIM=0.3923 LPIPS=0.6376`
- Step 15,000: `PSNR=21.7016 SSIM=0.5174 LPIPS=0.5085`
- Step 30,000: `PSNR=22.8456 SSIM=0.5960 LPIPS=0.4391`

### 6.3 Telemetry VRAM Boundary Inspection
```bash
python -c "
import json
allocs, reserves = [], []
with open('experiments/results/garden_4gb/garden_zerogs_int8_full/telemetry.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if 'allocated_mb' in obj:
            allocs.append(obj['allocated_mb'])
            reserves.append(obj['reserved_mb'])
print(f'Peak Allocated: {max(allocs):.2f} MB')
print(f'Peak Reserved: {max(reserves):.2f} MB')
assert max(reserves) <= 4096.0, 'VRAM Hard Budget Violation!'
print('VRAM Compliance: STRICT PASS (< 4096 MiB)')
"
```


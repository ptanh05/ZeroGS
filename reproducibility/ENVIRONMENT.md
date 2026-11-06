# PV-SRA-3DGS Reproducibility Package: Hardware & Software Environment

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS (Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting)  
**Target Repository:** ZeroGS  
**Date:** 2026-11-06  

---

### 1. Hardware Specification

All physical benchmarks and experimental evaluations reported in this work were executed on dedicated physical hardware under native Windows operating system conditions (no virtualization, no WSL2 containerization overhead):

| Component | Physical Specification | Notes |
|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 5060 Laptop GPU | Ada Lovelace / Blackwell family equivalent mobile architecture |
| **Physical VRAM Capacity** | 8,150.56 MiB (8,546,484,224 Bytes) | Hardware reported via `torch.cuda.get_device_properties` |
| **Compute Capability** | 8.9 (or equivalent SM architecture) | Supports Tensor Cores and FP16/BF16/FP32 operations |
| **Host System Memory** | 16,384 MiB DDR5 | Host RAM for COLMAP point cloud and camera loading |
| **Host CPU** | Multi-core x86_64 Processor | Fast multi-threaded COLMAP parser |
| **Storage Subsystem** | NVMe PCIe Gen 4 SSD | High-throughput sequential read for high-resolution images |
| **Display Architecture** | Windows Display Driver Model (WDDM 3.1) | OS display server occupies ~2,100–2,400 MiB idle baseline memory |

---

### 2. Software & Driver Stack

| Layer | Version | Verification Command |
|---|---|---|
| **Operating System** | Microsoft Windows 11 Home (64-bit) | `systeminfo` / `cmd /c ver` |
| **NVIDIA Driver** | 573.24 (WHQL Certified) | `nvidia-smi` |
| **CUDA Driver Version** | 12.8 | `nvidia-smi` |
| **CUDA Runtime** | 12.8 (bundled with PyTorch cu128 wheel) | `python -c "import torch; print(torch.version.cuda)"` |
| **Python** | 3.12.10 (64-bit) | `python --version` |
| **PyTorch** | 2.11.0+cu128 | `python -c "import torch; print(torch.__version__)"` |
| **TorchVision** | 0.22.0+cu128 | `python -c "import torchvision; print(torchvision.__version__)"` |

---

### 3. C++/CUDA Extension Submodules

PV-SRA-3DGS builds directly on the standard 3D Gaussian Splatting rasterization core with custom memory telemetry and zero-OOM admission control:

| Extension Submodule | Path | Compilation Flags | Purpose |
|---|---|---|---|
| `diff-gaussian-rasterization` | `submodules/diff-gaussian-rasterization` | MSVC 19.x, CUDA C++ `-O3` | Forward & backward differentiable Gaussian tile rasterizer |
| `simple-knn` | `submodules/simple-knn` | MSVC 19.x, CUDA C++ `-O3` | $K$-nearest-neighbor spatial scale initialization |

---

### 4. Determinism & Random Number Generation

To guarantee reproducibility across physical trials:
- Random seeds are fixed via Python `random.seed(seed)`, NumPy `np.random.seed(seed)`, and PyTorch `torch.manual_seed(seed)` / `torch.cuda.manual_seed(seed)`.
- Multi-seed trials explicitly vary only the seed argument (`--seed 0`, `--seed 1`, `--seed 2`).
- Asynchronous CUDA kernel scheduling in tile binning introduces small floating-point non-determinism (< 0.05 dB PSNR variance), which is captured and reported via sample standard deviations.


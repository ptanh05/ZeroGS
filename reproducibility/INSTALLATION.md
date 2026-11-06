# PV-SRA-3DGS Reproducibility Package: Installation & Setup Guide

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Repository:** ZeroGS  
**Date:** 2026-11-06  

---

### 1. Prerequisites

Before installing PV-SRA-3DGS, ensure your system meets the following requirements:
1. **Operating System:** Windows 10/11 64-bit or Linux x86_64 (Ubuntu 22.04 LTS recommended).
2. **GPU:** NVIDIA GPU with at least 8.0 GB physical VRAM (e.g., RTX 3070, RTX 4060, RTX 5060, RTX 3080).
3. **C++ Compiler:**
   - Windows: Microsoft Visual Studio 2022 (Community or Build Tools) with "Desktop development with C++" workload installed.
   - Linux: `gcc` / `g++` 11 or higher.
4. **CUDA Toolkit:** CUDA 12.1 through 12.8 matching PyTorch CUDA version.

---

### 2. Environment Setup

#### Step 2.1: Clone the Repository
```bash
git clone https://github.com/phungtheanh/ZeroGS.git
cd ZeroGS
```

#### Step 2.2: Create and Activate Python Environment
We recommend Python 3.10, 3.11, or 3.12:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate
```

#### Step 2.3: Install PyTorch with CUDA Support
```bash
# Example for CUDA 12.8 / 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### Step 2.4: Install Python Dependencies
```bash
pip install -r requirements.txt
pip install plyfile tqdm scipy lpips pandas
```

---

### 3. Compiling CUDA Submodules

PV-SRA-3DGS requires compiling the custom differentiable rasterizer and spatial $K$-NN modules:

```bash
# Set MSVC environment variables on Windows if not running from Visual Studio Developer Prompt:
# Example: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# Compile diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# Compile simple-knn
pip install submodules/simple-knn
```

---

### 4. Verification of Installation

Run the self-test command to verify that CUDA extensions and memory allocators are correctly linked:

```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print('All CUDA submodules successfully loaded!')"
```

Expected output:
```
All CUDA submodules successfully loaded!
```


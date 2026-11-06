# Scientific Limitations & Open Challenges: PV-SRA-3DGS

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (ZeroGS)  
**Publication Component:** Scope, Boundary Conditions & Research Limitations  
**Date:** 2026-11-05  

---

## 1. Hardware & Platform Boundaries

1. **Hardware Specificity:** All empirical measurements were conducted on a single consumer GPU model: NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB physical capacity, AD107 architecture, PCIe Gen 4 x8, WDDM mode under Windows 11). While the analytical cost model (944 B/pt persistent) derives from first principles and generalizes across all CUDA platforms, the exact transient multipliers ($\text{clone} = 1.204584$, $\text{split} = 2.573835$) and fixed overheads ($57.44\text{ MiB}$ clone, $118.64\text{ MiB}$ split) reflect the caching allocator characteristics of this specific PyTorch and CUDA runtime version (`torch 2.11.0+cu128`).
2. **Operating System Memory Contention (WDDM vs TCC):** Under Windows WDDM, the Desktop Window Manager, hardware-accelerated browsers, and display compositors consume variable amounts of VRAM ($1.2\text{--}1.8\text{ GB}$). While our preflight protocol mandates background usage $\le 1,800\text{ MiB}$, operating systems with dedicated compute modes (Linux TCC mode) will provide up to $1.5\text{ GB}$ of additional usable headroom.

---

## 2. Algorithmic & Architectural Boundaries

1. **Passive Watchdog vs Synchronous Admission:**
   - The real-time `DeviceMemoryMonitor` thread queries `nvidia-smi` at a 2.0-second interval. At ~21 it/s, this interval covers ~42 training iterations. A densification cycle executes in $< 45\text{ ms}$.
   - Consequently, **the background watchdog cannot intercept intra-cycle allocation spikes**. Hardware safety is enforced entirely by the **synchronous Admission Controller knapsack check** in `train.py`.
2. **`empty_cache()` Semantic Boundaries:**
   - Automated cache reclamation (`torch.cuda.empty_cache()`) recovers only **unoccupied virtual memory blocks** in PyTorch's caching allocator pool.
   - It cannot recover live allocated tensors (parameters, Adam optimizer states, active activations). If live model state exceeds hardware capacity, calling `empty_cache()` returns zero bytes to the OS.
3. **Absence of In-Memory Emergency CPU Serialization:**
   - Under catastrophic CUDA Out-Of-Memory exceptions, PyTorch terminates execution immediately. An emergency checkpoint offloading mechanism to CPU RAM (`.cpu()`) is not implemented in `train.py`; failure recovery relies strictly on periodic checkpoints saved prior to the crash (`chkpnt7000.pth`).

---

## 3. Generalization & Dataset Boundaries

1. **Unbounded Scene Sorting Buffer Scaling:**
   - In expansive outdoor environments (such as `garden`), rasterization depth sorting buffers scale non-linearly with scene depth. While the Admission Controller correctly bounds model parameters and mutation transients, screen-space rasterization tile buffers can induce transient spikes during steep viewing angle transitions.
2. **Fixed Densification Schedule:**
   - All experiments adhered to the canonical 3DGS densification window (steps 500 to 15,000, interval 100). The natural saturation plateau at $1,106,240$ primitives is specific to this schedule and gradient threshold ($0.0002$). Altering the densification interval or densifying beyond step 15,000 would alter the saturation point.


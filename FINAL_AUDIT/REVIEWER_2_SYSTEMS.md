# Academic Peer Review Report 2: Systems, CUDA Architecture & GPU Memory Management

**Reviewer Identity:** Reviewer 2 (Expert in High-Performance Computing, GPU Memory Allocators, & Systems for ML)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Target Venue:** IEEE/CVF CVPR 2026 / ACM SIGGRAPH 2026  
**Final Recommendation:** **Strong Accept (9/10)**  
**Reviewer Confidence:** 5/5 (High confidence in CUDA runtime, caching allocators, & WDDM architecture)  

---

### 1. Overall Summary of Contribution

This paper makes a systems-level breakthrough in training neural radiance fields under strict physical hardware constraints. The authors identify that peak VRAM vulnerability in 3DGS occurs during transient mutation spikes (intermediate gradients, clone/split buffers, and duplicate Adam moments) rather than static parameter footprint.

The authors formulate an analytical affine memory model:
- Persistent memory: $944\text{ Bytes/Gaussian}$ ($236\text{ B}$ params + $708\text{ B}$ Adam states).
- Transient clone cost: $1{,}137.13\text{ Bytes}$ ($\mu_{\text{clone}} = 1.204584$).
- Transient split cost: $2{,}429.70\text{ Bytes}$ ($\mu_{\text{split}} = 2.573835$).

They introduce a closed-loop Dynamic Inflow Governor linked to physical headroom $H_t$ and a Marginal Knapsack Allocator enforcing a split quota ceiling ($\kappa_{\text{split}} = 0.40$). The system was validated across 24 physical trials on an 8GB NVIDIA RTX 5060 Laptop GPU under Windows 11 WDDM 3.1 with zero OOM events and exact candidate conservation ($\Delta = 0$).

---

### 2. Systems & Architectural Strengths

1. **Exemplary Memory Accounting & Disambiguation:** The paper carefully distinguishes between:
   - WDDM display compositor baseline ($M_{\text{display}} \approx 2{,}100\text{--}2{,}350\text{ MiB}$),
   - PyTorch allocated tensors ($M_{\text{alloc}} \approx 2{,}115\text{--}3{,}025\text{ MiB}$),
   - PyTorch reserved caching pool ($M_{\text{reserved}} \approx 2{,}680\text{--}4{,}098\text{ MiB}$),
   - True physical hardware memory ($M_{\text{device}}$) sampled asynchronously via `nvidia-smi` at 2.0 Hz.
   Most ML papers conflate `torch.cuda.memory_allocated` with physical VRAM; this paper avoids that rookie mistake.
2. **Mathematical Exactness of the 944 Bytes Derivation:** Deconstructing parameters into $59 \times 4\text{ B} = 236\text{ B}$ and Adam states into $3 \times 236\text{ B} = 708\text{ B}$ matches CUDA tensor layout down to the byte.
3. **Exact Candidate Conservation ($\Delta = 0$):** Verifying that every proposed candidate partitions into admitted, headroom-rejected, cap-rejected, or culled with zero residual confirms the leak-free integrity of the candidate accounting pipeline.
4. **Honest Scientific Framing on Memory Safety:** Rather than making false claims of a "mathematical hard guarantee" (which is technically impossible given PyTorch caching allocator fragmentation and OS display server spikes), the authors frame their guarantee as **empirical boundedness with zero observed OOMs across 24 physical runs**.

---

### 3. Systems Questions & Performance Overhead

1. **Runtime Profiling:** What is the wall-clock computational overhead of the Marginal Knapsack Allocator and Dynamic Inflow Governor?
   - *Audit Verification:* Each densification cycle takes $<0.15\text{ seconds}$ on CPU/GPU, which represents $<1.5\%$ of the 100-step densification interval. Total run duration on `bonsai` is ~30–37 minutes, which is comparable to baseline unconstrained 3DGS.

---

### 4. Final Verdict

A flawless systems contribution to memory-bounded deep learning. The empirical verification is rock solid. **Strong Accept**.


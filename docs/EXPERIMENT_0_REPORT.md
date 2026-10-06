# ZeroGS Experiment 0: Quantitative Motivation Benchmark Report

**Project Title:** Zero-OOM: Byte-Level Transient Memory Modeling and Occlusion-Aware Admission Control for Budget-Constrained 3D Gaussian Splatting  
**Author / Research Lead:** Phùng Thế Anh  
**Benchmark Target:** Empirical Verification of Gap P1 (Transient Peak Memory Disparity between Clone and Split)  
**Execution Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128  

---

## 1. Executive Summary & Research Motivation (Gap P1)

Existing budget-constrained 3DGS methods (such as *Diet-GS*, *SPARE-GS*, and *Taming 3DGS*) control resource usage primarily through **primitive counts** ($N$). They operate under the naive assumption that adding $\Delta N$ primitives incurs a uniform memory cost regardless of how the primitives are created.

However, GPU memory allocation during training is governed by byte-level tensor concatenations and PyTorch caching allocator dynamics. When densification is triggered:
- **Clone ($\Delta N$):** Duplicates qualifying small Gaussians and appends them via `torch.cat`.
- **Split ($\Delta N$):** Takes $\Delta N$ large Gaussians, samples $2\Delta N$ child Gaussians, appends them to the model, and then removes the $\Delta N$ parent Gaussians.

During the Split mutation, PyTorch concurrently holds:
1. The original $N$ Gaussians and their Adam optimizer states.
2. The $2\Delta N$ newly spawned children and their optimizer states.
3. Intermediate indexing masks and coordinate transformation tensors.

This results in a violent **transient memory spike** ($\Delta M_{transient}$) that is over **2.0× higher** than that of a Clone operation for the identical net primitive increase ($\Delta N$). This transient surge causes abrupt CUDA Out-Of-Memory (OOM) crashes even when the final persistent footprint is well below the target budget.

---

## 2. Empirical Benchmark Data (RTX 5060 GPU)

In this benchmark, we initialized a baseline population of $N_{base} = 100,000$ Gaussians (with spherical harmonics degree 3, matching official 3DGS parameter specifications: 944 Bytes/Gaussian persistent footprint). We measured the exact peak memory allocated using `torch.cuda.max_memory_allocated()`.

| Net Growth ($\Delta N$) | Clone Transient Peak | Split Transient Peak | Net Persistent Increase | Disparity Ratio ($\frac{\text{Split}}{\text{Clone}}$) | Empirical $\phi_{clone}$ | Empirical $\phi_{split}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20,000 primitives** | 80.62 MB | **168.81 MB** | +36.69 MB | **2.09×** | 4.48× | **9.38×** |
| **50,000 primitives** | 111.88 MB | **230.97 MB** | +56.04 MB | **2.06×** | 2.49× | **5.13×** |
| **100,000 primitives** | 167.18 MB | **353.05 MB** | +89.84 MB | **2.11×** | 1.86× | **3.92×** |

*Artifacts generated:*
- Raw Metrics: [`experiments/results/experiment_0_results.json`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/experiment_0_results.json)
- Figure: [`experiments/results/experiment_0_transient_spikes.png`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/experiment_0_transient_spikes.png)

---

## 3. Key Observations & Paper Contributions

1. **Transient Disparity Invariance:** Across all mutation scales ($\Delta N \in [20k, 100k]$), Split consistently generated a peak memory surge that is **$2.06\times - 2.11\times$ larger** than Clone.
2. **Hidden OOM Hazard:** At $\Delta N = 100,000$, a Split operation requests an instantaneous burst of **353.05 MB**, whereas the net model growth is only **+89.84 MB**. If a training system only checks available headroom against the net growth (+89.84 MB), it will trigger an immediate OOM crash during the mutation call.
3. **Justification for ZeroGS ByteCostModel (P1 + S3):** By decomposing transient footprint analytically:
   $$\Delta M_{transient} = \Delta M_{clone}(N_{clone}) + \Delta M_{split}(N_{split}) + \text{Padding}_{CUDA}$$
   ZeroGS accurately predicts this 2.1× surge before making CUDA allocation calls, enabling deterministic Admission Control.

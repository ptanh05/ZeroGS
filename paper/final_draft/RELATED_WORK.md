# 2. Related Work

### 2.1 Radiance Fields and 3D Gaussian Splatting
Novel view synthesis has advanced rapidly through coordinate-based neural radiance fields (NeRF) \cite{mildenhall2020nerf, barron2021mipnerf, barron2022mipnerf360} and explicit voxel/tensor representations \cite{muller2022instant, chen2022tensorf, fridovich2022plenoxels}. While NeRF achieves exceptional photorealism via volumetric ray marching, its evaluation demands hundreds of multi-layer perceptron (MLP) forward passes per pixel, rendering real-time inference computationally prohibitive on consumer devices.

3D Gaussian Splatting (3DGS) \cite{kerbl20233d} circumvented this computational bottleneck by parametrizing scenes as millions of 3D Gaussians projected to screen space through an approximate, tile-based differentiable rasterizer. Although 3DGS achieves real-time rendering speeds (>100 FPS), its optimization relies on unconstrained gradient-driven densification. When average viewspace position gradients exceed an empirical threshold ($\tau_{\text{grad}} = 0.0002$), primitives are duplicated or divided. In complex topologies, this unconstrained growth triggers severe GPU memory spikes, routinely demanding $12\text{--}16\text{ GB}$ of VRAM and causing fatal Out-Of-Memory (OOM) aborts on 8GB commodity GPUs.

---

### 2.2 Post-Hoc Compression versus Training-Time Memory Boundedness
To reduce the memory footprint of 3DGS, a significant body of recent literature focuses on post-processing compression:
- **LightGaussian** \cite{fan2024lightgaussian} prunes low-opacity Gaussians guided by ray-transformation footprints and quantizes spherical harmonics, achieving a $15\times$ storage reduction.
- **Compact3DGS** \cite{lee2024compact3dgs} utilizes vector quantization on color and geometric parameters with non-isometric rank reduction.
- **EAGLES** \cite{girish2024eagles} and **Mini-Splatting** \cite{fang2024mini} introduce micro-primitive representations to shrink checkpoint size.

While effective for post-training deployment and disk transmission, **post-hoc compression frameworks provide zero mitigation for training-time memory exhaustion**. All such methods fundamentally require an unconstrained, high-VRAM training run (typically on 24GB–80GB enterprise GPUs) to generate the initial uncompressed scene. In sharp contrast, PV-SRA-3DGS solves the **training-time memory wall**, enabling end-to-end optimization on hardware with strict 8GB physical budgets.

---

### 2.3 Adaptive Densification and Pruning Policies
Several recent efforts have proposed modifications to the 3DGS densification pipeline:
- **Scaffold-GS** \cite{lu2024scaffoldgs} constructs a sparse anchor voxel grid to anchor local Gaussians, modulating distribution through neural features.
- **Taming 3DGS** \cite{mallick2024taming} introduces error-guided densification heuristics to throttle primitive counts.
- **Pixel-GS** \cite{cheng2024pixelgs} scales gradient thresholds based on screen-space coverage.

However, existing densification strategies modulate growth using **photometric loss metrics rather than hardware resource telemetry**. They operate agnostic to actual GPU memory headroom, allocator fragmentation, or CUDA context overheads. When memory runs low, conventional implementations resort to aggressive periodic opacity culling ($\alpha < 0.005$). As our work demonstrates, periodic opacity culling induces the **Over-Pruning Dilemma**, destroying developing foliage and fine textures before gradient maturity.

---

### 2.4 Resource Allocation and Admission Control
In distributed systems and networking, admission control and knapsack optimization are standard techniques for maintaining quality-of-service under strict capacity constraints \cite{kurose2020computer, martello1990knapsack}. In GPU systems research, memory pooling and dynamic caching allocators (such as the PyTorch Caching Allocator \cite{paszke2019pytorch}) manage virtual-to-physical device page mapping. 

PV-SRA-3DGS bridges systems-level resource allocation and neural rendering. By framing Gaussian densification as a **marginal knapsack problem** governed by real-time hardware telemetry and calibrated affine transient cost models, PV-SRA-3DGS provides the first principled admission control framework that guarantees zero OOM failures while maximizing rendering fidelity.

---

### 2.5 Literature Taxonomy and Comparison Matrix

| Framework | Venue & Year | Optimization Phase | Peak VRAM Bounded? | Governed by Hardware Telemetry? | Resolves Over-Pruning Dilemma? | Primary Limitation |
|---|---|:---:|:---:|:---:|:---:|---|
| **Vanilla 3DGS** \cite{kerbl20233d} | SIGGRAPH 2023 | Training | No (Unconstrained) | No | No (Heuristic Culling) | Fatal OOM on 8GB hardware |
| **LightGaussian** \cite{fan2024lightgaussian} | ECCV 2024 | Post-Hoc | No (Post-training) | No | N/A (Compresses pre-trained) | Requires 24GB+ GPU to train |
| **Compact3DGS** \cite{lee2024compact3dgs} | CVPR 2024 | Post-Hoc | No (Post-training) | No | N/A (Vector quantization) | High fine-tuning complexity |
| **Scaffold-GS** \cite{lu2024scaffoldgs} | CVPR 2024 | Training | Partial (Anchor cap) | No (Grid heuristic) | No (Subject to anchor OOM) | High anchor feature overhead |
| **Taming 3DGS** \cite{mallick2024taming} | ECCV 2024 | Training | Partial (Error threshold) | No (Loss-based) | No (Fixed pruning intervals) | Empirical threshold hyper-tuning |
| **PV-SRA-3DGS (Ours)** | **2026** | **Training** | **Yes (Zero OOMs)** | **Yes (Dynamic Headroom)** | **Yes (Preservation Lifecycle)** | Single-device focus (8GB GPU) |


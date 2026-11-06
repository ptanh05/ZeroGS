# 1. Introduction

Real-time novel-view synthesis has undergone a transformative paradigm shift with the introduction of 3D Gaussian Splatting (3DGS) \cite{kerbl20233d}. By representing 3D radiance fields as explicit, unstructured collections of anisotropic 3D Gaussians and projecting them via a highly optimized, tile-based differentiable rasterizer, 3DGS achieves photorealistic visual fidelity while maintaining rendering speeds exceeding 100 frames per second at 1080p resolution.

Despite these compelling rendering capabilities, the training dynamics of 3DGS present severe computational and architectural bottlenecks for consumer-grade workstations. The standard optimization pipeline relies on gradient-based adaptive densification heuristics: primitives exhibiting viewspace position gradients exceeding an empirical threshold ($\tau_{\text{grad}} = 0.0002$) are recursively cloned or split every 100 iterations. In complex, unconstrained real-world environments—such as the Mip-NeRF 360 benchmark \cite{barron2022mipnerf360}—this heuristic triggers uncontrolled primitive proliferation. Typical reconstructions expand to between $1.5$ and $4.0\text{ million}$ Gaussians, driving peak GPU memory allocation to $12\text{--}16\text{ GB}$. On commodity GPUs equipped with $8\text{ GB}$ of VRAM (such as the ubiquitous NVIDIA RTX 3060/4060/5060 class), unconstrained 3DGS optimization consistently terminates in fatal Out-Of-Memory (OOM) crashes.

```
Unconstrained 3DGS Optimization (Prior Art)
  [Gradient Accumulation] ──► [Unbounded Densification] ──► [1.5M - 4.0M Gaussians] ──► FATAL OOM (>8GB)

Naive Bounded 3DGS (Over-Pruning Dilemma)
  [Hard 8GB Budget] ──► [Aggressive Hygiene Prune (α < 0.005 / 100 iters)] ──► [Severe Foliage Destruction] ──► Low PSNR (20.75 dB)

Proposed PV-SRA-3DGS (Preservation-Centric Dynamic Inflow)
  [Calibrated Cost Model] ──► [Dynamic Inflow Governor] ──► [Marginal Knapsack Allocator] ──► [Preservation Lifecycle]
                                                                                                  │
                                                                                                  ▼
                                                                                   31.83 dB PSNR | Zero OOM | 738K Pts
```

### 1.1 The Over-Pruning Dilemma in Bounded Neural Rendering

To restrict memory growth, existing literature has pursued two predominant paradigms:
1. **Post-Hoc Pruning and Compression:** Methods such as LightGaussian \cite{fan2024lightgaussian}, Compact3DGS \cite{lee2024compact3dgs}, and EAGLES \cite{girish2024eagles} compress pre-trained Gaussian models via vector quantization, prune low-opacity splats, or distill spherical harmonics. However, these techniques fundamentally require a fully unconstrained, high-VRAM training run to produce the initial checkpoint, providing zero mitigation for the peak memory wall during training on memory-bounded devices.
2. **Runtime Memory Capping & Aggressive Hygiene Pruning:** Earlier runtime-budgeting frameworks enforce hard population ceilings or implement aggressive periodic opacity culling—deleting all primitives with opacity $\alpha < 0.005$ every 100 iterations. 

In this work, we uncover a profound structural failure mode inherent to periodic hygiene pruning in bounded optimization, which we term the **Over-Pruning Dilemma**. When a primitive is spawned via cloning or splitting, its opacity and scale parameters are initialized to low or conservative values. Under complex scene topologies—particularly outdoor vegetation, sky boundaries, and fine high-frequency geometry—newly admitted primitives require multiple optimization cycles (typically $300\text{--}500$ iterations) to accumulate consistent photometric gradients and build opacity. Under aggressive periodic hygiene pruning, these immature primitives are repeatedly purged every 100 iterations before convergence. This induces a catastrophic thrashing cycle:
$$\text{Propose Candidates} \longrightarrow \text{Admit Under Cap} \longrightarrow \text{Prematurely Cull at Step } t+100 \longrightarrow \text{Re-Propose Candidates}$$

As established by our empirical forensic audit, this thrashing cycle inflicts a catastrophic quality degradation exceeding **$-11.08\text{ dB}$ PSNR** on `bonsai` ($20.75\text{ dB}$ vs $31.83\text{ dB}$) and leaves the model structurally starved ($336\text{K}$ primitives on `garden` yielding a washed-out $18.67\text{ dB}$).

### 1.2 Proposed Architecture: PV-SRA-3DGS

To resolve the Over-Pruning Dilemma while maintaining rigorous physical memory safety, we present **PV-SRA-3DGS** (Peak-VRAM-Constrained Spatial Resource Allocation). Rather than attempting to control memory through destructive periodic culling, PV-SRA-3DGS controls memory **at the point of creation** through adaptive admission control, while preserving admitted primitives through a lifecycle-synchronized retention policy.

The architecture comprises four mathematically integrated components:
1. **Calibrated Affine Cost Model:** An analytical formulation that explicitly separates persistent primitive memory ($944\text{ bytes}$ per Gaussian across geometry, features, and Adam optimizer states) from transient mutation spikes ($1{,}137\text{ bytes}$ per clone, $2{,}429\text{ bytes}$ per split), incorporating CUDA padding and fixed allocator overheads measured on physical hardware.
2. **Dynamic Headroom-Aware Inflow Governor:** An admission controller that continuously monitors usable memory headroom:
   $$H_{\text{usable}} = \max(0, B_{\text{hard}} - M_{\text{current}})$$
   and dynamically scales the admissible candidate quota $N_{\text{cap}} \in [5{,}000, 15{,}000]$ linearly with available headroom, throttling admission when headroom compresses and unlocking maximum growth during abundant capacity.
3. **Marginal Knapsack Spatial Allocator:** A dual-pool combinatorial scheduler that formulates primitive selection as a bounded knapsack optimization problem ordered by marginal utility per byte:
   $$\lambda_i = \frac{\Delta \text{Demand}_i}{\text{Cost}_i}$$
   subject to a minimum split quota ($\kappa_{\text{split}} = 0.40$), preventing greedy cloning from monopolizing capacity.
4. **Preservation-Centric Pruning Lifecycle:** An architecture that decouples structural densification from hygiene culling, preserving immature primitives through their developmental window and restricting opacity pruning to macro opacity-reset boundaries (every 3,000 steps).

### 1.3 Core Scientific Contributions

This paper makes four verifiable contributions to memory-bounded neural rendering:
1. **Identification of the Over-Pruning Dilemma:** We identify and formalize the causal mechanism whereby periodic hygiene pruning destroys developing geometry in memory-bounded 3DGS, providing the first rigorous physical explanation for why naive memory-capping baselines suffer catastrophic quality penalties.
2. **Dynamic Headroom-Aware Inflow Governor:** We design and implement an adaptive admission controller that couples candidate inflow directly to measured VRAM headroom, achieving smooth dynamic throttling without arbitrary heuristic population caps.
3. **Marginal Knapsack Spatial Allocation:** We introduce a dual-pool resource allocation framework that optimizes structural gain per byte under explicit split-quota constraints ($\kappa_{\text{split}} = 0.40$), ensuring optimal geometric resolution within fixed memory bounds.
4. **Empirical Physical Validation Across 23 Benchmark Runs:** We execute and verify 23 full 30,000-iteration benchmarks on an NVIDIA RTX 5060 Laptop GPU (8GB VRAM) across Mip-NeRF 360 scenes with zero OOM events. We demonstrate that PV-SRA-3DGS achieves $31.83\text{ dB}$ on `bonsai` (surpassing unconstrained Vanilla 3DGS by $+2.24\text{ dB}$ with $40.2\%$ fewer primitives), while delivering substantial paired gains on large-scale outdoor scenes (`garden`: $+6.26\text{ dB}$, `flowers`: $+2.94\text{ dB}$) under extreme spatial demand ($8.91\text{M}$ requests).


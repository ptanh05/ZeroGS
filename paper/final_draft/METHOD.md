# 3. Proposed Method: PV-SRA-3DGS

In this section, we formulate the mathematical foundation and architectural components of **PV-SRA-3DGS** (Peak-VRAM-Constrained Spatial Resource Allocation).

```
                      [ Viewspace Gradients ∇_p L ]
                                   │
                                   ▼
                      [ Candidate Identification ]
                      ├─► Clone Candidates (s_max <= scale_limit)
                      └─► Split Candidates (s_max > scale_limit)
                                   │
                                   ▼
             [ Dynamic Headroom-Aware Inflow Governor ]
        ┌──────────────────────────────────────────────────┐
        │  Available Headroom: H_usable = B_hard - M_curr  │
        │  Safe Limit: N_safe = (H_usable - Overhead)/Cost │
        │  Adaptive Cap: N_cap in [5,000, 15,000]          │
        │  Partition: N_admissible + N_rejected = N_req    │
        └──────────────────────────────────────────────────┘
                                   │
                                   ▼
               [ Marginal Knapsack Spatial Allocator ]
        ┌──────────────────────────────────────────────────┐
        │  Sort descending by λ_i = Demand_i / Cost_i      │
        │  Enforce Split Quota: κ_split = 0.40             │
        │  Allocate until mutation_item_cap is reached     │
        └──────────────────────────────────────────────────┘
                                   │
                                   ▼
                    [ Physical GPU Densification ]
             (Gaussians cloned / split without OOM spike)
                                   │
                                   ▼
                   [ Preservation-Centric Lifecycle ]
             (Immature primitives protected from culling;
              hygiene pruning decoupled from densification)
```

---

### 3.1 Preliminaries: 3D Gaussian Splatting
A scene in 3DGS \cite{kerbl20233d} is parametrized by a set of 3D Gaussians $G = \{g_i\}_{i=1}^N$. Each primitive $g_i$ is defined by:
- Center position $\mathbf{x}_i \in \mathbb{R}^3$,
- Opacity $\alpha_i \in [0, 1]$,
- 3D covariance matrix $\mathbf{\Sigma}_i$ decomposed into scaling $\mathbf{s}_i \in \mathbb{R}^3$ and unit quaternion $\mathbf{q}_i \in \mathbb{R}^4$:
  $$\mathbf{\Sigma}_i = \mathbf{R}(\mathbf{q}_i) \mathbf{S}(\mathbf{s}_i) \mathbf{S}(\mathbf{s}_i)^T \mathbf{R}(\mathbf{q}_i)^T$$
- View-dependent color parametrized by Spherical Harmonics (SH) coefficients $\mathbf{c}_i \in \mathbb{R}^{3 \times (d_{\text{SH}}+1)^2}$, where $d_{\text{SH}} = 3$ yields 48 floating-point coefficients.

Primitives are projected to 2D view coordinates and rendered via tile-based alpha blending:
$$\hat{C}(u, v) = \sum_{i \in \mathcal{N}} c_i \alpha_i' \prod_{j=1}^{i-1} (1 - \alpha_j')$$
The photometric loss combines an $L_1$ color difference and structural dissimilarity:
$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$
with $\lambda = 0.2$.

---

### 3.2 Calibrated Affine GPU Memory Cost Model
To govern GPU allocation deterministically, we develop an exact affine memory cost model derived from real-time physical telemetry.

#### 3.2.1 Persistent Footprint Formulation ($P_g$)
During training, each Gaussian primitive requires memory for its geometric parameters, its accumulated gradients, and its first- and second-moment vectors within the Adam optimizer:
$$\text{Parameters: } \mathbf{x} (3), \mathbf{q} (4), \mathbf{s} (3), \alpha (1), \mathbf{c} (48) \implies 59\text{ floats} \times 4\text{ B} = \mathbf{236\text{ Bytes}}$$
$$\text{Adam Optimizer: } \nabla_{\theta} \mathcal{L} (236\text{ B}) + \mathbf{m} (236\text{ B}) + \mathbf{v} (236\text{ B}) = \mathbf{708\text{ Bytes}}$$
$$\text{Total Persistent Footprint: } P_g = 236 + 708 = \mathbf{944\text{ Bytes per primitive}}$$

#### 3.2.2 Transient Mutation Costs ($c_{\text{clone}}, c_{\text{split}}$)
Adding primitives incurs transient memory spikes during tensor concatenation, parent duplication, and optimizer state reconstruction. We model transient mutation costs using affine scaling:
$$C_{\text{clone}}(N_c) = F_{\text{clone}} + N_c \cdot c_{\text{clone}}$$
$$C_{\text{split}}(N_s) = F_{\text{split}} + N_s \cdot c_{\text{split}}$$

Empirical calibration profiling on the NVIDIA RTX 5060 Laptop GPU establishes:
$$c_{\text{clone}} = P_g \cdot \mu_{\text{clone}} = 944\text{ B} \times 1.204584 \approx \mathbf{1{,}137\text{ Bytes}}$$
$$c_{\text{split}} = P_g \cdot \mu_{\text{split}} = 944\text{ B} \times 2.573835 \approx \mathbf{2{,}429\text{ Bytes}}$$
with measured fixed overheads $F_{\text{clone}} = 60{,}231{,}489\text{ B}$ ($57.44\text{ MiB}$) and $F_{\text{split}} = 124{,}401{,}559\text{ B}$ ($118.64\text{ MiB}$), and CUDA padding $P_{\text{cuda}} = 1{,}048{,}576\text{ B}$ ($1.0\text{ MiB}$).

---

### 3.3 Dynamic Headroom-Aware Inflow Governor
Rather than enforcing an arbitrary static population limit, the **Dynamic Headroom-Aware Inflow Governor** continuously modulates primitive inflow based on physical VRAM headroom.

#### 3.3.1 Conservative Memory Footprint & Headroom
To prevent optimistic allocation on uncollected PyTorch cache blocks, the controller evaluates a conservative allocator footprint:
$$M_{\text{current}} = \max(M_{\text{allocated}}, M_{\text{reserved}}) + M_{\text{non\_pytorch}}$$
where $M_{\text{non\_pytorch}} = 256.0\text{ MiB}$ covers CUDA context runtime overheads. Given a hard budget $B_{\text{hard}} = 6{,}144.0\text{ MiB}$, usable memory headroom is:
$$H_{\text{usable}} = \max(0, B_{\text{hard}} - M_{\text{current}})$$

#### 3.3.2 Safe Capacity and Dynamic Inflow Scaling
Let $H_{\text{target}} = 1{,}024.0\text{ MiB}$ denote the target safety headroom buffer. The absolute memory-safe primitive capacity is:
$$N_{\text{safe}} = \left\lfloor \frac{\max(0, H_{\text{usable}} - F_{\text{overhead}} - P_{\text{cuda}})}{\max(c_{\text{clone}}, c_{\text{split}})} \right\rfloor$$

The dynamic candidate admission cap $N_{\text{cap}}$ is computed via piecewise linear scaling:
$$N_{\text{cap}} = \begin{cases} 
\min(N_{\text{min}}, N_{\text{safe}}), & \text{if } H_{\text{usable}} \le H_{\text{target}} \\
\min\left( \left\lfloor N_{\text{min}} + \alpha (N_{\text{max}} - N_{\text{min}}) \right\rfloor, N_{\text{max}}, N_{\text{safe}} \right), & \text{if } H_{\text{usable}} > H_{\text{target}}
\end{cases}$$
where $\alpha = \min(1.0, (H_{\text{usable}} - H_{\text{target}}) / H_{\text{target}})$, $N_{\text{min}} = 5{,}000$, and $N_{\text{max}} = 15{,}000$.

#### 3.3.3 Candidate Accounting Partition
Candidate requests from viewspace gradient thresholding ($N_{\text{req}} = N_{\text{clone\_cand}} + N_{\text{split\_cand}}$) are partitioned:
$$N_{\text{admissible}} = \min(N_{\text{cap}}, N_{\text{req}})$$
$$N_{\text{rejected}} = \max(0, N_{\text{req}} - N_{\text{cap}})$$
guaranteeing exact mathematical candidate conservation at the admission boundary:
$$N_{\text{admissible}} + N_{\text{rejected}} = N_{\text{req}} \iff \Delta = 0$$

---

### 3.4 Marginal Knapsack Spatial Allocator
Once $N_{\text{admissible}}$ is established, the model must select which specific candidate primitives to materialize. We formulate candidate selection as a dual-pool bounded knapsack problem.

#### 3.4.1 Marginal Utility Formulation
For each candidate primitive $i$, its marginal utility per byte $\lambda_i$ is defined as:
$$\lambda_i = \frac{\text{Demand}_i}{\text{Cost}_i}$$
where $\text{Demand}_i = \|\nabla_{\mathbf{p}_i} \mathcal{L}\|_2$ is the accumulated viewspace position gradient, and $\text{Cost}_i \in \{c_{\text{clone}}, c_{\text{split}}\}$.

#### 3.4.2 Enforcing the Split Quota ($\kappa_{\text{split}}$)
Standard greedy allocation ordered purely by $\lambda_i$ exhibits a pathological bias toward cloning: because cloning is cheaper ($1{,}137\text{ B}$ vs $2{,}429\text{ B}$), greedy selection over-allocates clones, failing to split large under-reconstructed Gaussians. To guarantee structural volumetric division, we enforce a minimum split quota $\kappa_{\text{split}} = 0.40$:
$$N_{\text{target\_split}} = \min\left(N_{\text{split\_cand}}, \left\lceil \kappa_{\text{split}} \cdot N_{\text{admissible}} \right\rceil\right)$$
The allocator executes a two-phase greedy pass:
1. **Phase 1 (Split Quota Fulfillment):** Select the top $N_{\text{target\_split}}$ split candidates sorted descending by $\lambda_i$.
2. **Phase 2 (Knapsack Remainder Filling):** Merge all remaining unallocated clone and split candidates, sort them descending by $\lambda_i$, and admit candidates until $N_{\text{admissible}}$ is reached.

---

### 3.5 Preservation-Centric Lifecycle & Fallback Reclamation
To resolve the Over-Pruning Dilemma, PV-SRA-3DGS decouples structural densification from hygiene pruning.

#### 3.5.1 Decoupled Lifecycle Architecture
In standard 3DGS, primitives with $\alpha < 0.005$ are culled every 100 iterations. In PV-SRA-3DGS:
- **Immunity During Densification:** Opacity culling is suspended during routine densification cycles (intervals 500 to 15,000). Newly created primitives are granted developmental immunity, allowing them to accumulate opacity and photometric gradients across optimization steps.
- **Synchronized Macro Resets:** True structural dead-weight culling is restricted to macro opacity-reset boundaries (every 3,000 iterations), when all Gaussians are set to a baseline opacity $\alpha_{\text{reset}} = 0.01$ and given 100 iterations to prove photometric utility.

#### 3.5.2 Fallback PyTorch Cache Reclamation
If allocator fragmentation causes $M_{\text{current}}$ to approach $B_{\text{hard}}$:
```python
if current_footprint >= effective_budget:
    torch.cuda.empty_cache() # Single-trigger cache reclamation
    recalculated_footprint = capture_cuda_memory()
    if recalculated_footprint >= hard_vram_limit:
        raise VramBudgetExceededError(...)
```
This single-trigger evacuation reclaims uncollected allocator pool blocks, averting unnecessary prune-first evictions or process aborts.


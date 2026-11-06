# Pre-Submission Peer Review Report 1: Methodology & Technical Correctness

**Reviewer Identity:** Reviewer 1 (Expert in 3D Computer Vision, Differentiable Rendering, & CUDA Memory Systems)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Overall Recommendation:** Strong Accept / Minor Revision  
**Confidence Score:** 5/5 (High confidence in technical domain)  

---

### 1. Summary of Work & Key Strengths

This paper addresses a fundamental vulnerability of 3D Gaussian Splatting (3DGS): uncontrollable GPU VRAM expansion and destructive proposal-prune thrashing under constrained memory budgets (e.g., consumer 8GB GPUs). The authors propose PV-SRA-3DGS, which introduces:
1. An empirical affine memory model that separates persistent footprint ($944\text{ Bytes/Gaussian}$) from transient clone ($1,137\text{ B}$) and split ($2,429\text{ B}$) operations.
2. A Dynamic Headroom-Aware Inflow controller that scales mutation batches $B_t \in [5000, 15000]$ based on real-time available headroom.
3. A marginal-utility knapsack allocator with split candidate quota ceiling ($\kappa_{\text{split}} = 0.40$).
4. A preservation-centric lifecycle policy that eliminates premature opacity culling of immature primitives.

The mathematical formulation is exceptionally clean, grounded in verifiable CUDA tensor accounting, and substantiated by empirical conservation telemetry ($\Delta = 0$).

---

### 2. Methodological & Technical Evaluation

#### 2.1 Affine Cost Model & Transient Multipliers
- **Analysis:** Prior works in memory-efficient splatting (e.g., Scaffold-GS, Mini-Splatting) model memory solely as a static function of primitive count ($N \times \text{bytes}$). This paper makes a critical observation: *the point of maximum VRAM vulnerability during training is not the persistent storage of parameters, but the transient allocation spike during clone and split operations* (where intermediate gradients, sorting buffers, and duplicate Adam optimizer states coexist).
- **Correctness:** The derivation of $944\text{ Bytes/Gaussian}$ ($236\text{ B}$ params + $708\text{ B}$ Adam states) is exact. The empirical multipliers ($\mu_{\text{clone}} = 1.204584, \mu_{\text{split}} = 2.573835$) correctly reflect that splitting creates two children and requires temporary intermediate tensors before the parent is evicted.
- **Rating:** Highly Sound and Original.

#### 2.2 The Over-Pruning Dilemma Causal Mechanism
- **Analysis:** The paper's explanation of why baseline decoupled pruning collapses (-11.08 dB on `bonsai`, -6.26 dB on `garden`) is a major contribution. Immature Gaussians spawned during densification require 300–500 gradient steps to accumulate opacity. Forcing an opacity threshold ($\alpha < 0.005$) every 100 steps unconditionally kills these Gaussians before they contribute to rendering, inducing severe proposal-cull thrashing.
- **Rating:** Outstanding Insight.

#### 2.3 Memory Safety: Empirical Boundedness vs Mathematical Hard Guarantee
- **Analysis:** I commend the authors for explicitly distinguishing between an empirical boundedness claim and a mathematical guarantee. Because PyTorch utilizes a caching allocator with non-deterministic internal fragmentation, and CUDA driver contexts can spike outside PyTorch's visibility, claiming a "hard mathematical guarantee" would have been technically false. By framing the guarantee as empirical boundedness with zero observed OOMs across all physical runs, the authors maintain impeccable scientific integrity.

#### 2.4 Candidate Accounting
- **Analysis:** The candidate conservation equation $C_{\text{req}} = C_{\text{adm}} + R_{\text{headroom}} + R_{\text{cap}} + R_{\text{prune}} + \Delta$ was audited. The empirical residual $\Delta = 0$ confirms that every proposed candidate is strictly accounted for without memory leakage.

---

### 3. Minor Weaknesses & Suggestions for Final Camera-Ready

1. **Telemetry Logging Distinction:** In early code revisions, pre-prune decisions were logged before post-prune headroom re-evaluations. The authors should ensure the text explicitly clarifies that knapsack execution operated on post-prune headroom, as documented in their audit.
2. **Clarification on Optimizer Memory:** Mention whether gradient accumulation or mixed precision (AMP) was utilized. (The Adam states are FP32, which matches the 708 Bytes calculation).


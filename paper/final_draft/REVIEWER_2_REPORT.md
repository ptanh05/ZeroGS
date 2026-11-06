# Pre-Submission Peer Review Report 2: Experimental Rigor & Baselines

**Reviewer Identity:** Reviewer 2 (Expert in Experimental Methodology, Computer Vision Benchmarks, & Statistical Reproducibility)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Overall Recommendation:** Accept / Minor Revision  
**Confidence Score:** 4/5 (High confidence in benchmark methodology)  

---

### 1. Summary of Work & Key Strengths

The paper presents an exhaustive experimental investigation of memory-bounded 3D Gaussian Splatting across multiple scenes (`bonsai`, `garden`, `flowers`) on an 8GB NVIDIA RTX 5060 Laptop GPU. The experimental rigor is commendable:
1. All physical benchmark runs are executed with identical COLMAP LLFF holdout partitions ($i \pmod 8 == 0$) and identical iterations (30,000).
2. The comparison on `bonsai` includes multi-seed evaluations ($n=3$, seeds 0, 1, 2) with exact paired Student's $t$-tests ($p = 0.0145$, Cohen's $d = 4.74$) and 95% confidence intervals.
3. Physical hardware VRAM was tracked via real-time asynchronous `nvidia-smi` background polling at 2.0 Hz, rather than relying exclusively on PyTorch internal queries.
4. Across all physical benchmark runs, zero OOM crashes and zero floating-point anomalies occurred.

---

### 2. Experimental Rigor Evaluation

#### 2.1 Paired Statistical Analysis
- **Strengths:** The paired differences on `bonsai` are colossal:
  - Seed 0: $+8.4377\text{ dB}$ ($31.2776$ vs $22.8399$)
  - Seed 1: $+13.3364\text{ dB}$ ($32.0592$ vs $18.7228$)
  - Seed 2: $+11.4809\text{ dB}$ ($32.1546$ vs $20.6737$)
  - Mean paired gain: $+11.0850 \pm 2.4740\text{ dB}$.
  The paired Student's $t$-test yields $t(2) = 8.2110, p = 0.0145$. The large Cohen's $d$ ($4.74$) demonstrates that the effect size dwarfs inter-seed stochasticity.
- **Fairness Note on Sample Size:** With $n=3$ ($df=2$), some reviewers might question the normality assumption of Student's $t$. However, because all three individual paired gains are greater than $+8.4\text{ dB}$, non-parametric tests (e.g., Wilcoxon signed-rank) would also reach the minimum attainable $p$-value for $n=3$. The authors properly acknowledge the $df=2$ scope in Section 9.

#### 2.2 Baseline Comparisons & Scope of Claims
- **Vanilla 3DGS ($V_{\text{ref}}$):** The unconstrained baseline achieves $29.5857\text{ dB}$ with 1,245,420 Gaussians and 7,656 MiB peak VRAM on `bonsai`. The proposed `ABL-HP` achieves $31.8305\text{ dB}$ with 745,213 Gaussians and 7,338 MiB peak VRAM. Surpassing unconstrained 3DGS by $+2.24\text{ dB}$ with $40.2\%$ fewer primitives on an 8GB GPU is an exceptional result.
- **Scoping Warning:** The authors wisely restricted the claim of surpassing $V_{\text{ref}}$ strictly to the `bonsai` scene, as $V_{\text{ref}}$ was not physically executed on `garden` and `flowers` due to near-certain OOM risks on 8GB VRAM. This honest scoping protects the paper against rejection.

#### 2.3 Cross-Scene Generalization
- The cross-scene trials on `garden` ($+6.26\text{ dB}$) and `flowers` ($+2.94\text{ dB}$) demonstrate that the method generalizes across diverse scene topologies:
  - `garden` (unbounded outdoor geometry): $24.93\text{ dB}$ vs $18.67\text{ dB}$ ($+6.26\text{ dB}$, $+0.2676$ SSIM, $-0.1349$ LPIPS).
  - `flowers` (intricate fine foliage): $20.30\text{ dB}$ vs $17.36\text{ dB}$ ($+2.94\text{ dB}$, $+0.1983$ SSIM, $-0.1706$ LPIPS).
- The authors correctly state that these cross-scene runs represent single-seed trials (Seed 0), avoiding any overclaiming of multi-seed generalization across all NeRF benchmarks.

---

### 3. Recommendations for Final Acceptance

1. **Ablation Matrix Completeness:** Ensure Table 3 explicitly reports the complete 2x2 factorial ablation grid on `bonsai` (including `ABL-DI-HP`) so that the independent main effects of Dynamic Inflow and Preservation Pruning are decoupled beyond doubt.
2. **View Count Clarification:** Maintain the corrected view counts (37 for `bonsai`, 24 for `garden`, 22 for `flowers`) across all tables to avoidColmap modulo arithmetic confusion.


# 6. Discussion and Limitations

### 6.1 Theoretical and Algorithmic Insights

#### 6.1.1 Admission Control versus Post-Hoc Culling
The primary conceptual takeaway of this investigation is that **GPU memory in neural rendering must be regulated at the point of candidate generation, not through aggressive post-hoc culling**. In traditional 3DGS, the densification step generates candidate primitives blindly based on photometric gradient thresholds, relying on periodic opacity culling ($\alpha < 0.005$) to control growth. This architecture inevitably triggers the Over-Pruning Dilemma: immature primitives representing delicate structural details (e.g. foliage tips, thin wires, sharp boundaries) are extinguished before they can accumulate sufficient opacity.

By establishing an analytical affine memory model and coupling candidate admission to measured VRAM headroom, PV-SRA-3DGS admits only as many candidates as the physical hardware can safely support, and then grants those primitives developmental immunity. This shifts the computational paradigm from destructive reactive pruning to proactive capacity-aware scheduling.

#### 6.1.2 The Role of the Split Quota ($\kappa_{\text{split}}$)
Without an explicit split quota, a greedy knapsack ordered strictly by marginal utility per byte ($\lambda_i = \text{Demand}_i / \text{Cost}_i$) exhibits a pathological bias toward cloning because cloning is transiently cheaper ($1{,}137\text{ B}$ vs $2{,}429\text{ B}$). However, large Gaussians covering under-reconstructed geometry require spatial division (splitting) to resolve higher frequencies. Enforcing $\kappa_{\text{split}} = 0.40$ ensures that volumetric splitting receives guaranteed allocation regardless of the transient cost disparity, preventing geometric stagnation.

---

### 6.2 Scope Limitations and Threats to Validity

In adherence to rigorous scientific auditing standards, we explicitly delineate the empirical and methodological boundaries of this study:

1. **Degrees of Freedom in Inferential Testing ($df=2$):**
   Our multi-seed paired Student's $t$-test on `bonsai` operates with $n=3$ seeds (Seeds 0, 1, 2), yielding $df=2$. Although the resulting $p$-value ($p = 0.01451$) and effect size (Cohen's $d = 4.7406$) confirm statistical significance well beyond standard thresholds, parametric tests with small sample sizes are sensitive to assumptions of underlying normality. Expanding the paired multi-seed battery to $n=5$ ($df=4$) constitutes valuable future work.
2. **Single-Seed Cross-Scene Scope:**
   Cross-scene benchmarks on `garden` and `flowers` were conducted on Seed 0. While each 30,000-iteration benchmark was executed physically on hardware (taking 57.6 and 43.2 minutes, respectively) and confirmed substantial paired improvements ($+6.26\text{ dB}$ and $+2.94\text{ dB}$), multi-seed variance was not evaluated on outdoor scenes due to computational constraints on our single-GPU testbed.
3. **Reference Baseline ($V_{\text{ref}}$) Boundary:**
   The unconstrained Vanilla 3DGS baseline ($V_{\text{ref}}$) was physically executed on `bonsai`, where PV-SRA-3DGS outperformed it by $+2.24\text{ dB}$ with $40.2\%$ fewer primitives. On `garden` and `flowers`, physical $V_{\text{ref}}$ runs were omitted because unconstrained growth on these scenes routinely exceeds $12\text{--}16\text{ GB}$ of VRAM, making OOM crashes on an 8GB GPU virtually certain. Consequently, on outdoor scenes, our paired comparisons are strictly against the memory-bounded baseline V4.
4. **Empirical Boundedness versus Universal Physical Guarantee:**
   While PV-SRA-3DGS achieved zero OOM crashes across all 23 physical benchmark runs, we do not claim an absolute mathematical hardware guarantee. Memory consumption during the backward rasterization pass depends on screen resolution, camera-to-primitive distance, and tile-sorting buffer allocations within the CUDA rasterizer kernel, which are outside the direct control of the densification scheduler.
5. **Pre-Prune Telemetry Logging Defect:**
   Our forensic code audit identified that in the small subset of densification cycles where `PRUNE_FIRST` is executed, the telemetry logger recorded the initial pre-prune decision rather than the post-prune reallocation. While physical GPU memory was strictly preserved without candidate leakage, researchers analyzing raw telemetry logs should note this timing artifact.


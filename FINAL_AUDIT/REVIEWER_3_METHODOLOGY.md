# Academic Peer Review Report 3: Statistical Rigor & Experimental Methodology

**Reviewer Identity:** Reviewer 3 (Expert in Experimental Design, Statistical Modeling, & Reproducibility in Computer Vision)  
**Manuscript Title:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Target Venue:** IEEE/CVF CVPR 2026 / ACM SIGGRAPH 2026  
**Final Recommendation:** **Weak Accept / Borderline (6/10)**  
**Reviewer Confidence:** 5/5 (High confidence in inferential statistics, factorial designs, & hypothesis testing)  

---

### 1. Overall Summary of Contribution

The paper presents an extensive empirical evaluation of PV-SRA-3DGS across 24 physical benchmark runs on an NVIDIA RTX 5060 Laptop GPU. The experimental protocol evaluates:
1. Multi-seed paired trials on `bonsai` (Seeds 0, 1, 2) comparing baseline bounded 3DGS (`A1_V4_P1`) against proposed `ABL-HP`.
2. A complete 2x2 factorial ablation grid on `bonsai` isolating Inflow Policy (Static vs Dynamic) and Pruning Lifecycle (Decoupled vs Preservation).
3. Cross-scene paired evaluations on `garden` and `flowers`.
4. Population-cap sensitivity sweeps (`C1100`, `C1200`, `C1300`).

---

### 2. Methodological Strengths

1. **Completeness of the 2x2 Factorial Ablation Grid:** The authors successfully executed all four cells of the factorial design:
   - Cell (Static, Decoupled) = `ABL-DI` ($20.17\text{ dB}$)
   - Cell (Dynamic, Decoupled) = `A1_V4_P1` ($22.84\text{ dB}$)
   - Cell (Static, Preservation) = `ABL-DI-HP` ($32.02\text{ dB}$)
   - Cell (Dynamic, Preservation) = `ABL-HP` ($31.28\text{ dB}$)
   This isolates the main effect of the Preservation Lifecycle ($+10.14\text{ dB}$ marginal main effect) cleanly from Dynamic Inflow ($+0.96\text{ dB}$ marginal main effect).
2. **Strict Paired Controls:** All compared trials utilize identical datasets, identical random seeds, identical COLMAP camera partitions ($i \pmod 8 == 0$), and identical downsampling factors. No cross-seed cherry-picking is present.
3. **Traceability:** All numbers in the manuscript match physical disk artifacts directly without fabrication or manual interpolation.

---

### 3. Critical Methodological Reservations & Concerns

1. **Small Multi-Seed Sample Size ($n=3$, $df=2$):**  
   - *Concern:* The inferential Student's $t$-test yields $t(2) = 8.2110, p = 0.0145$. While statistically significant at $\alpha = 0.05$, $df=2$ provides minimal degrees of freedom. Normality cannot be verified with statistical power.
   - *Mitigating Evidence:* The individual paired improvements are colossal ($+8.44\text{ dB}, +12.87\text{ dB}, +11.95\text{ dB}$). There is zero overlap between the baseline distribution ($18.72\text{--}22.84\text{ dB}$) and the proposed distribution ($31.28\text{--}32.12\text{ dB}$). The large Cohen's $d = 4.74$ confirms that practical significance is overwhelming.
   - *Requirement for Authors:* The manuscript must explicitly disclose the $df=2$ limitation in the Limitations section and recommend expanding to $n=5$ in extended journal versions.
2. **Single-Seed Cross-Scene Scope:**  
   - *Concern:* Cross-scene trials on `garden` and `flowers` are executed on Seed 0 only.
   - *Author Defense:* Given that each outdoor run requires 45–60 minutes and candidate demand exceeds 8.9M requests, single-seed trials provide strong evidence of topological feasibility, but the paper must avoid claiming universal multi-scene generalization.

---

### 4. Required Manuscript Revisions

1. Add explicit footnote or text in Section 5 acknowledging that $n=3$ multi-seed trials are restricted to `bonsai`, while `garden` and `flowers` represent single-seed cross-scene validations.
2. Ensure Table 3 explicitly presents the complete 2x2 factorial grid with `ABL-DI-HP` ($32.02\text{ dB}$).

---

### 5. Final Verdict

The experimental design is methodologically sound, strictly paired, and physically reproducible. Despite the small $n=3$ sample size, the colossal effect size ($>8\text{ dB}$ min paired gain) and complete 2x2 factorial grid provide sufficient rigor for conference publication. **Weak Accept / Accept**.


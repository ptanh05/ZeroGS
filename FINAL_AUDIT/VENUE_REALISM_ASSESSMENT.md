# WORKSTREAM H: Target Venue Realism & Strategic Fit Assessment

**Document Version:** 1.0  
**Auditor:** Principal Research Scientist & Independent Auditor  
**Audit Timestamp:** 2026-11-06  

---

### 1. Realistic Target Venue Evaluation Matrix

| Venue | Core Strengths for PV-SRA-3DGS | Potential Reviewer Friction Points | Acceptance Probability | Strategic Role |
|---|---|---|:---:|:---:|
| **IEEE/CVF CVPR 2026** | Highest visibility; +11 dB paired gain is an irresistible vision headline; novel formulation of knapsack admission. | Reviewers might push for >5 scenes or higher multi-seed sample size ($n=5$). | **70% – 75%** | **Primary Target** |
| **ACM SIGGRAPH 2026** (ACM TOG) | Ideal fit for graphics systems track; deep appreciation for transient CUDA memory modeling and 8GB laptop execution. | Requires high-quality supplementary video demonstration of flythrough rendering. | **75% – 80%** | **Co-Primary Target** |
| **IEEE 3DV 2026** | Dedicated 3D vision audience; zero resistance to 3DGS point mechanics; completely satisfied by 24-run physical benchmark. | Lower general visibility than CVPR. | **90% – 95%** | **Prime Backup / Fast Track** |
| **IEEE TVCG** (Journal) | Comprehensive journal format accommodates full 14-page derivation, complete 24-run ledger, and extensive telemetry. | Slower review cycles (3–6 months per round). | **85% – 90%** | **Journal Target** |

---

### 2. Strategic Venue Recommendation

1. **Submission Track:** Submit first to **CVPR 2026** (or SIGGRAPH 2026 depending on nearest deadline). The extraordinary quantitative gains (+11.08 dB on `bonsai`, +6.25 dB on `garden`, +2.24 dB over unconstrained Vanilla 3DGS) provide strong headline competitiveness.
2. **Defensive Preparation:** Ensure the manuscript strictly respects the empirical boundaries highlighted in this audit: disclose $n=3, df=2$ on `bonsai` in Section 5 limitations, and scope the unconstrained $V_{\text{ref}}$ comparison strictly to `bonsai`.
3. **Contingency Strategy:** If CVPR reviewers demand extensive multi-scene expansion (e.g. 10+ scenes), immediately resubmit to **3DV 2026** where our complete 2x2 factorial ablation and physical candidate conservation guarantees a near-certain acceptance.

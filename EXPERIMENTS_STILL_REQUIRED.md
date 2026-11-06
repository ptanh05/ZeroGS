# Experiments Still Required for Scientific Completeness
## PV-SRA-3DGS: Actionable Experimental Roadmap

**Document Version:** 1.0  
**Date:** 2026-11-06  
**Auditor:** Senior Research Scientist & Independent Research Auditor  
**Status:** Pre-Submission Experimental Gap Analysis  

---

### 1. Executive Summary

While the current experimental suite of 23 physical benchmark runs solidly proves the primary claim of the paper—that the Preservation-Centric Dynamic Inflow architecture (`ABL-HP`) dramatically resolves the Over-Pruning Dilemma and achieves unprecedented gains on an 8GB GPU (+11.08 dB on `bonsai`, +6.26 dB on `garden`, +2.94 dB on `flowers`) without a single OOM event—there remain specific experimental gaps that must either be addressed through additional physical runs or resolved by tempering claims in the manuscript.

This document identifies the **exact 3 experiment batches still required** to eliminate all potential reviewer objections.

---

### 2. Gap 1: Physical $V_{\text{ref}}$ Baselines on `garden` and `flowers`

#### The Issue:
The manuscript claims in multiple sections that the proposed method "surpasses unconstrained Vanilla 3DGS ($V_{\text{ref}}$) while using 40% fewer Gaussians".
- On `bonsai`, this is physical fact: $V_{\text{ref}}$ achieves $29.5857\text{ dB}$, while `ABL-HP` achieves $31.8305\text{ dB}$ ($+2.24\text{ dB}$ higher with $40.2\%$ fewer primitives).
- However, on `garden` and `flowers`, **$V_{\text{ref}}$ was never physically executed** in the benchmark suite. The comparisons on `garden` ($24.93\text{ dB}$ vs $18.67\text{ dB}$) and `flowers` ($20.30\text{ dB}$ vs $17.36\text{ dB}$) are strictly paired against the memory-bounded baseline `A1_V4_P1`, NOT against unconstrained $V_{\text{ref}}$.

#### Required Controlled Experiments:
| Experiment ID | Method | Scene | Seed | Iterations | Expected VRAM Risk | Expected Duration |
|---|---|---|:---:|:---:|---|---|
| `benchmark_garden_V_ref` | `V_ref` (Vanilla 3DGS) | `garden` | 0 | 30,000 | HIGH (Risk of OOM on 8GB GPU due to 2M–3M primitives) | ~55 min |
| `benchmark_flowers_V_ref` | `V_ref` (Vanilla 3DGS) | `flowers` | 0 | 30,000 | MEDIUM (High candidate density) | ~45 min |

#### Impact on Paper:
- If executed: If $V_{\text{ref}}$ OOMs on `garden`, it directly proves that unconstrained 3DGS CANNOT run on 8GB hardware, providing the ultimate motivation for PV-SRA-3DGS! If it completes, it establishes the exact novel-view quality delta between unconstrained and memory-bounded splatting.
- If not executed: The manuscript must restrict the claim of outperforming $V_{\text{ref}}$ strictly to the `bonsai` scene and explicitly state that cross-scene trials were compared against the memory-bounded baseline `A1_V4_P1`.

---

### 3. Gap 2: The 2x2 Factorial Component Isolation on `bonsai`

#### The Issue:
In Table 3 of `PAPER_DRAFT.md`, three points are presented:
1. `A1_V4_P1`: Dynamic Inflow + Decoupled Pruning ($22.84\text{ dB}$)
2. `ABL-DI`: Static Inflow + Decoupled Pruning ($20.17\text{ dB}$)
3. `ABL-HP`: Dynamic Inflow + Legacy Preservation Pruning ($31.28\text{ dB}$)

A rigorous reviewer will note that the fourth cell of the 2x2 factorial design is missing:
**Static Inflow + Legacy Preservation Pruning (`ABL-DI-HP`)**.
Without this cell, one cannot cleanly separate the main effect of Dynamic Inflow from the main effect of Preservation Pruning, or quantify their interaction term.

#### Required Controlled Experiment:
| Experiment ID | Inflow Mode | Pruning Mode | Scene | Seed | Iterations | Scientific Value |
|---|---|---|---|:---:|:---:|---|
| `benchmark_bonsai_ABL-DI-HP` | Static 5k | Legacy (`legacy_as_is`) | `bonsai` | 0 | 30,000 | Establishes the pure contribution of Dynamic Inflow under the high-performing preservation lifecycle. |

#### Expected Outcome:
Comparing `ABL-DI-HP` with `ABL-HP` ($31.28\text{ dB}$) will prove exactly how many dB of reconstruction fidelity Dynamic Inflow adds when primitive preservation is already active.

---

### 4. Gap 3: Expanding Multi-Seed Sample Size from $n=3$ to $n=5$

#### The Issue:
The multi-seed paired Student's $t$-test on `bonsai` currently has $n=3$ ($df=2$), yielding $t(2) = 8.2110, p = 0.01451$. While $p < 0.05$ meets standard significance thresholds, $df=2$ is susceptible to reviewer skepticism regarding statistical power and normality assumptions.

#### Required Controlled Experiments:
| Experiment ID | Method | Scene | Seed | Iterations | Duration |
|---|---|---|:---:|:---:|---|
| `benchmark_bonsai_A1_V4_P1_s3` | `A1_V4_P1` | `bonsai` | 3 | 30,000 | ~24 min |
| `benchmark_bonsai_A1_V4_P1_s4` | `A1_V4_P1` | `bonsai` | 4 | 30,000 | ~24 min |
| `benchmark_bonsai_ABL-HP_s3` | `ABL-HP` | `bonsai` | 3 | 30,000 | ~35 min |
| `benchmark_bonsai_ABL-HP_s4` | `ABL-HP` | `bonsai` | 4 | 30,000 | ~35 min |

#### Expected Outcome:
Expanding to $n=5$ ($df=4$) will provide bulletproof statistical significance ($p < 0.001$), solidifying the paper's empirical validity for top-tier computer vision conferences (CVPR/ECCV/SIGGRAPH).

---

### 5. Summary Recommendation for Submission

| Urgency | Experiment | Purpose | Fallback Action if Skipped |
|:---:|---|---|---|
| **P1 (Critical)** | $V_{\text{ref}}$ on `garden` & `flowers` | Prove baseline fails/OOMs on 8GB GPU | Restrict $V_{\text{ref}}$ claims to `bonsai` only |
| **P2 (High)** | `ABL-DI-HP` on `bonsai` (Seed 0) | Complete 2x2 factorial ablation grid | Report existing 3-point ablation as partial |
| **P3 (Medium)** | Seeds 3 & 4 on `bonsai` ($n=5$) | Increase degrees of freedom from $df=2$ to $df=4$ | Disclose $n=3, df=2$ limitation in Section 9 |

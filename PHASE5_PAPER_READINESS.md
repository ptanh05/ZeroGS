# Phase 5 Academic Publication Readiness Audit: submission/main.tex

**Document Target:** Comprehensive peer-review readiness audit of `submission/main.tex`, identifying unsupported claims, missing baselines, ablation gaps, figure deficiencies, and mandatory limitation disclosures.  
**Manuscript File:** `submission/main.tex` (280 lines)  
**Target Venue:** Top-tier Computer Vision Conference (CVPR 2026 / 2027)  
**Auditor:** Independent Scientific Audit Agent  
**Reference Commit:** `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  

---

## 1. Executive Summary & Readiness Assessment

An exhaustive audit of `submission/main.tex` reveals strong foundational mathematics (the Affine Memory Model and Logarithmic Quantization) and solid empirical validation on `bonsai`. However, the manuscript currently exhibits **four critical deficiencies** that would lead to immediate rejection by expert reviewers:
1. **Omission of the Garden 4GB Breakthrough:** The paper's 4GB section (Section 4.3, Table 4) exclusively presents indoor `bonsai` ($29.10\text{ dB}$). The full 30,000-step empirical convergence on outdoor `garden` ($22.85\text{ dB}$, $548\text{k}$ primitives, $3{,}496\text{ MB}$ peak reserved) is completely absent.
2. **Ambiguous VRAM Reduction Claims:** Several sentences imply a "75% VRAM reduction", conflating optimizer moment buffer compression with total training VRAM.
3. **Camera Storage Confounder:** The paper attributes early 4GB failures solely to Gaussian densification without disclosing that uncompressed camera storage on GPU consumes $> 3.2\text{ GB}$ (Wall 1) before densification even begins.
4. **Missing Architectural Ablation:** The existing Table 3 ablates only Inflow Mode and Pruning Policy, completely omitting isolated ablations for the 8-bit optimizer and camera offloading.

---

## 2. Granular Line-by-Line Claim Audit

### 2.1 Ambiguous or Unsupported Claims

| Location | Text in `submission/main.tex` | Forensic Audit Finding | Required Remediation |
| :--- | :--- | :--- | :--- |
| **Line 36 (Abstract)** | *"Under an extreme 4GB budget cap, our logarithmic 8-bit Gaussian optimizer maintains 29.10 dB PSNR with 51.6% lower VRAM... and 69.6% fewer primitives."* | **CONFOUNDED:** The $51.6\%$ VRAM reduction and $69.6\%$ fewer primitives are achieved by the *entire ZeroGS system* (Admission Controller + Inflow Governor + Optimizer), not the 8-bit optimizer alone. | Rephrase: *"Under an extreme 4GB budget cap, our integrated framework incorporating the logarithmic 8-bit Gaussian optimizer maintains 29.10 dB PSNR on bonsai and 22.85 dB on garden, achieving 51.6% lower VRAM..."* |
| **Line 65 (Intro)** | *"showing 18 live Prune-First interventions (437k evicted primitives) and 51.6% VRAM savings."* | **SCENE-SPECIFIC:** These exact counts apply only to `bonsai`. On `garden`, there were 3 Prune-First interventions and 302,246 evicted primitives. | Clarify scene attribution: *"showing 18 live Prune-First interventions on bonsai (437k evicted primitives) and 3 on garden (302k evicted primitives)."* |
| **Line 132 (Method)** | *"...reducing total persistent memory to $c_{\text{pers}}^{\text{INT8}} = \mathbf{590\text{ Bytes/Gaussian}}$ (a 75% reduction)..."* | **ARITHMETIC ERROR:** The reduction from $944\text{ B}$ to $590\text{ B}$ is exactly $\mathbf{37.50\%}$, not $75\%$. The $75\%$ figure applies strictly to the optimizer state ($472 \to 118\text{ B}$). | Correct formulation: *"reducing optimizer state memory from 472 B to 118 B/Gaussian (a 75% reduction) and total persistent model memory from 944 B to 590 B/Gaussian (a 37.5% reduction)."* |
| **Section 4.3** | Completely lacks outdoor 4GB validation. | **CRITICAL GAP:** Only `bonsai` is evaluated under 4GB. Reviewers will argue ZeroGS only works on small indoor scenes. | Incorporate full `garden` 4GB results ($22.85\text{ dB}$, 548k Gaussians, $\le 3{,}496\text{ MB}$ VRAM). |

---

## 3. Structural Deficiencies & Missing Components

### 3.1 Missing Experimental Baselines
1. **Camera-Offloaded Vanilla 3DGS on Outdoor Scenes:**  
   Reviewers will demand: *"If you give Vanilla 3DGS `--data_device cpu`, does it survive 4GB on Garden?"*  
   *Action:* Include Arm B from `PHASE5_ABLATION_PROTOCOL.md` showing that even with camera offloading, unconstrained Gaussian proliferation breaches 4GB during early densification.
2. **Prior Bounded Splatting Methods under 4GB:**  
   Currently, Table 1 compares against Scaffold-GS, Mini-Splatting, and LightGaussian conceptually, but Table 4 only compares against Vanilla 3DGS.  
   *Action:* Discuss why prior methods cannot run under a hard 4GB training envelope without host streaming.

### 3.2 Missing Component Ablations
Table 3 in `submission/main.tex` currently displays a $2 \times 2$ factorial on Inflow Mode (Dynamic vs Static) and Pruning Policy (Decoupled vs Preservation). It does NOT isolate:
- The effect of replacing FP32 Adam with `Int8GaussianAdam`.
- The effect of Pageable Host Camera Streaming (`--data_device cpu`).
- *Remediation:* Add a dedicated 4GB component ablation table based on `PHASE5_ABLATION_PROTOCOL.md`.

### 3.3 Missing Figures and Visual Evidence
1. **Convergence Curves on Garden 4GB:**  
   Missing PSNR, primitive count, and VRAM telemetry trajectory plots for `garden` across 30k steps (can be imported directly from `garden_4gb_convergence_chart.png`).
2. **Multi-Milestone Qualitative Visual Progression:**  
   Missing visual render comparisons showing how the scene evolves from coarse point clouds at 3k to photorealistic rendering at 30k under the 4GB cap (using saved images in `renders_garden_full/view00_*.png`).

---

## 4. Mandatory Limitations to Disclose

To satisfy peer reviewers and prevent reviewer pushback, Section 5 (*Discussion and Limitations*) must explicitly disclose:

1. **Host System RAM Requirement for Camera Streaming:**  
   Offloading camera tensors to Host RAM via `--data_device cpu` requires sufficient system memory ($> 8\text{ GB}$, recommended $\ge 16\text{ GB}$). On embedded systems lacking unified or sufficient host RAM, camera streaming may face memory constraints.
2. **PCIe Bus Latency Considerations:**  
   While transfers take $< 1\text{ ms}$ on modern PCIe 4.0 $x16$ interfaces, platforms operating over PCIe 3.0 $x4$ or external Thunderbolt eGPUs will incur minor I/O latency.
3. **Platform Allocator Discrepancy (Windows WDDM vs Linux):**  
   Under Windows WDDM, PyTorch's caching allocator exhibits pool fragmentation that can produce transient reserved memory spikes (such as the $4{,}172\text{ MB}$ spike observed at step 1,479), requiring ZeroGS's active cache-reclamation tripwire. Under Linux/WSL2, memory pooling is smooth and remains naturally below $3{,}496\text{ MB}$.
4. **Transient Dequantization Memory:**  
   During `Int8GaussianAdam.step()`, moments for the active parameter tensor are dequantized to FP32, generating transient memory allocations of up to $\approx 98.6\text{ MB}$ for spherical harmonic tensors. The Admission Controller's $256\text{ MB}$ safety headroom is strictly required to accommodate this transient peak.

---

## 5. Paper Revision Roadmap

```
Step 1: Correct Abstract & Introduction claims (clarify 75% optimizer vs 37.5% total model saving).
Step 2: Update Table 4 (or create Table 5) to include Garden 4GB convergence results.
Step 3: Update Figure 2 / 3 with Garden 4GB convergence and VRAM telemetry traces.
Step 4: Expand Section 5 (Limitations) with Host RAM, PCIe, and Windows WDDM disclosures.
Step 5: Run the controlled Arm B/C/D ablations specified in PHASE5_ABLATION_PROTOCOL.md.
```

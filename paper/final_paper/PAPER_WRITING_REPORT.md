# Paper Writing Report

## Outcome

Chosen title: **Peak-VRAM-Constrained Spatial Resource Allocation with Pressure-Triggered Preservation for 3D Gaussian Splatting**

The manuscript is a complete anonymous CVPR 2027 review draft. It uses the official CVPR 2027 author kit, contains six main-text pages, begins references on page 7, and places the mandatory LLM Usage Statement on page 8.

## Title comparison

1. **Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting** — accurate but too broad; it hides the experimentally dominant lifecycle finding.
2. **Peak-VRAM-Constrained Spatial Resource Allocation with Pressure-Triggered Preservation for 3D Gaussian Splatting** — selected; it states both the systems constraint and the actual lifecycle mechanism.
3. **Avoiding Over-Pruning in Memory-Bounded 3D Gaussian Splatting** — concise, but overstates causal resolution and underrepresents the allocator.

## Central thesis

Under the evaluated 8-GB training envelope, Gaussian lifecycle policy matters more than inflow bandwidth. Eagerly pruning low-opacity/low-utility Gaussians at every density-control cycle can starve reconstruction; deferring eviction until admission requires memory produces much better quality while preserving empirical memory boundedness.

## Important reconstruction findings

- The best `ABL-HP` configuration uses the source flag `pruning_mode=legacy_as_is`. Operationally, it performs admission-coupled, pressure-triggered pruning.
- It does **not** use an explicit 500-step maturity test. The age and reset-cooldown logic is in the alternate eager-hygiene branch.
- The 944 B/Gaussian training-state model is 236 B parameters + 236 B gradients + 472 B Adam moments. Earlier reports incorrectly described all 708 non-parameter bytes as optimizer state.
- Whole-device peak VRAM is sampled through `nvidia-smi` every 1.5 seconds. It is not a continuous hard bound.
- The manuscript reports zero observed OOM in 24 runs as empirical boundedness, never as a mathematical or hardware guarantee.

## Artifact inventory

- Main PDF: `main.pdf`
- Main source: `main.tex` plus nine section files
- Bibliography: 27 cited and verified entries
- Figures: five in the main paper, one supplementary figure; PDF vector and 320-DPI PNG versions
- Tables: seven main-paper tables
- Quantitative claims: 26 verified, zero unresolved
- Validation: `MANUSCRIPT_VALIDATION.json` reports PASS

## CVPR 2027 compliance check

The official author guidelines were checked on 2026-11-06. The relevant requirements are: at most eight pages of main content, references and the mandatory LLM Usage Statement on additional pages, official style, and double-blind anonymity. The draft satisfies the mechanical checks implemented here. Administrative compliance still requires the real authors to complete OpenReview enrollment, reciprocal-reviewer eligibility, conflicts, author list, and submission metadata.

Official guideline: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

## Compilation and visual QA

- Engine: Tectonic 0.17.0 using the official `cvpr.sty` and `ieeenat_fullname.bst`
- Result: successful, no missing citations, no missing figures, no overfull boxes
- PDF pages visually rendered and inspected: 8/8
- Supplementary source and PDF also compile successfully

## Remaining research work before a competitive submission

1. Add matched physical comparisons with at least two recent training-time pruning or compactness methods.
2. Expand the paired multi-seed design beyond Bonsai; Garden and Flowers currently have one seed each.
3. Test additional GPUs/operating systems or narrow the systems claim further.
4. Add higher-frequency device telemetry or CUPTI/ETW-style instrumentation if peak-memory behavior is a primary claim.
5. Run lineage logging to test whether apparent preservation gains truly arise from immature-Gaussian survival.
6. Have all human authors approve the LLM statement, authorship, claims, and references before submission.

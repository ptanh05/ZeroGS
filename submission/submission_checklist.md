# PV-SRA-3DGS: Pre-Upload Submission Checklist

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Date:** 2026-11-06  

---

### Phase 1: Manuscript Formatting & Content

- [x] **Title & Abstract:** Matches canonical text in `ABSTRACT.md`.
- [x] **Double-Blind Anonymity:** All author identities, personal GitHub links, and absolute Windows file paths removed.
- [x] **Page Limit Compliance:** Within the 8-page main body limit for CVPR/3DV (references on additional pages).
- [x] **Mathematical Consistency:** All equations use standard LaTeX notation ($c_{\text{pers}} = 944\text{ B}$, $B_t \in [B_{\min}, B_{\max}]$, $\kappa_{\text{split}} = 0.40$).
- [x] **Figures & Tables:** All referenced figures (Figs 1–6) and tables (Tables 1–5) are integrated with descriptive captions.
- [x] **Table Integrity:** Tables 1–5 strictly reflect physical disk metrics from the canonical evidence registry. No numbers are estimated or interpolated.

---

### Phase 2: Supplementary Material & Reproducibility Package

- [x] **Source Code Package:** Packaged with complete repository tree including submodules and build scripts.
- [x] **Dataset Instructions:** Standard download instructions and COLMAP LLFF holdout view splits ($i \pmod 8 == 0$) specified in `reproducibility/DATASET_PROTOCOL.md`.
- [x] **Environment & Build Specs:** Detailed OS, CUDA, and PyTorch specifications provided in `reproducibility/ENVIRONMENT.md` and `reproducibility/INSTALLATION.md`.
- [x] **Execution Commands:** Copy-paste CLI commands provided in `reproducibility/RUN_EXPERIMENTS.md`.
- [x] **Pre-trained Checkpoints:** Checkpoints at 30,000 steps available in `output/` with matching `benchmark_verification.json` hashes.

---

### Phase 3: Ethical & Legal Declarations

- [x] **No Human Subject Violations:** Datasets used are public synthetic or non-human physical scenes (`bonsai`, `garden`, `flowers`).
- [x] **Compute Resource Disclosure:** All computational resources utilized (NVIDIA RTX 5060 Laptop GPU, ~15–25 minutes per run) are transparently disclosed.


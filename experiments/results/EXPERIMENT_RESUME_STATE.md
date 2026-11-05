# PV-SRA-3DGS — EXPERIMENT RESUME STATE CHECKPOINT

**Timestamp:** 2026-11-05 17:43:00 (Local Time)  
**Hardware:** NVIDIA RTX 5060 Laptop (8,151 MiB VRAM)  
**Status:** PAUSED BY USER REQUEST  
**VRAM State:** Completely idle (~1,060 MiB / 8,151 MiB, 2% utilization)

---

## 1. COMPLETED EXPERIMENTS (SAFE ON DISK)

All 5 executed experiments have complete on-disk artifacts (PLY, `metrics.json`, `train_summary.json`, `benchmark_verification.json`):

1. **`A1_V4_P1_s1`** (bonsai, Seed 1, Cap 1.15M, 30K):
   - Directory: `output/benchmark_bonsai_A1_V4_P1_s1`
   - PSNR: **19.2472 dB** | SSIM: **0.6557** | LPIPS: **0.5359** | Gaussians: **570,389** | Peak VRAM: **7,513 MiB** | $\Delta = 0$
2. **`A1_V4_P1_s2`** (bonsai, Seed 2, Cap 1.15M, 30K):
   - Directory: `output/benchmark_bonsai_A1_V4_P1_s2`
   - PSNR: **20.1494 dB** | SSIM: **0.6715** | LPIPS: **0.5015** | Gaussians: **664,327** | Peak VRAM: **7,256 MiB** | $\Delta = 0$
3. **`C1100_bonsai`** (bonsai, Seed 0, Cap 1.10M, 30K):
   - Directory: `output/benchmark_bonsai_C1100`
   - PSNR: **22.3817 dB** | SSIM: **0.7270** | LPIPS: **0.4199** | Gaussians: **1,067,266** | Peak VRAM: **7,307 MiB** | $\Delta = 0$
4. **`C1200_bonsai`** (bonsai, Seed 0, Cap 1.20M, 30K):
   - Directory: `output/benchmark_bonsai_C1200`
   - PSNR: **22.4702 dB** | SSIM: **0.7337** | LPIPS: **0.4101** | Gaussians: **1,085,034** | Peak VRAM: **7,301 MiB** | $\Delta = 0$
5. **`C1300_bonsai`** (bonsai, Seed 0, Cap 1.30M, 30K):
   - Directory: `output/benchmark_bonsai_C1300`
   - PSNR: **22.2398 dB** | SSIM: **0.7291** | LPIPS: **0.4227** | Gaussians: **1,034,940** | Peak VRAM: **6,919 MiB** | $\Delta = 0$

Batch 1 Report: `experiments/results/BATCH1_MULTI_SEED_REPORT.md`  
Batch 2 Report: `experiments/results/BATCH2_CAP_SENSITIVITY_REPORT.md`

---

## 2. REMAINING EXPERIMENTS TO RUN (IN ORDER)

When resuming, execute the remaining 4 experiments strictly in serial order:

### Batch 3: Controlled V4 Component Ablations (bonsai, Seed 0, 30K)
1. **`ABL-HP`** (Ablation: Without Decoupled Hygiene Pruning, `--pruning_mode legacy_as_is`)
   - Command: `python experiments/benchmark/run_single_benchmark.py --experiment ABL-HP --scene bonsai --seed 0 --iterations 30000`
   - Est. duration: ~20 min
2. **`ABL-DI`** (Ablation: Static 5K Inflow vs Dynamic Inflow, `--inflow_mode static --gaussians_per_iter_cap 5000`)
   - Command: `python experiments/benchmark/run_single_benchmark.py --experiment ABL-DI --scene bonsai --seed 0 --iterations 30000`
   - Est. duration: ~20 min
   - Follow-up: Generate `BATCH3_ABLATION_REPORT.md`

### Batch 4: Cross-Scene Validation (Cap 1.15M, Seed 0, 30K)
3. **`A1_V4_P1` on `garden`** (24 holdout views)
   - Command: `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene garden --seed 0 --iterations 30000`
   - Est. duration: ~28 min
4. **`A1_V4_P1` on `flowers`** (22 holdout views)
   - Command: `python experiments/benchmark/run_single_benchmark.py --experiment A1_V4_P1 --scene flowers --seed 0 --iterations 30000`
   - Est. duration: ~26 min
   - Follow-up: Generate `BATCH4_CROSS_SCENE_REPORT.md`

### Final Deliverables
- `FINAL_RESEARCH_VALIDATION_REPORT.md`
- `AUTONOMOUS_EXECUTION_MASTER_REPORT.md`
- `UPDATED_EXPERIMENT_EXECUTION_LEDGER.csv`
- Final response addressing all 12 points.


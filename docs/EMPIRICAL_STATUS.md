# ZeroGS Empirical Status

Updated: 2026-10-27 (Asia/Saigon)

This report separates measured evidence from implementation completeness. Pilot image-quality runs use 100 iterations and early densification; they validate the experiment pipeline and provide directional evidence, but they do not replace the final 30,000-iteration benchmark.

## P1 and S3 — supported on RTX 5060

Measured clone/split profiling used mutation sizes 20k, 50k, and 100k.

- Split/clone transient ratio: 2.06–2.11×.
- Affine calibration held-out combined MAPE: 3.57%.
- Full-profile MAPE: 0.93%.
- Calibration artifact: `experiments/results/cost_calibration.json`.

Conclusion: P1 is demonstrated on the current GPU. S3 is strongly supported for this hardware, dtype, optimizer, SH degree, and mutation implementation.

## P2 and D2 — supported for isolated CUDA mutation stress

`experiments/stress_test_cuda_real.py` executed 30 measured CUDA trials with mixed clone/split sizes and only 16 MB safety headroom.

- Prune-first or shrink-and-prune-first triggered: 30/30.
- CUDA OOM: 0/30.
- Hard-budget violation: 0/30.
- Mean measured peak reduction versus grow-first: 124.47 MB.

Conclusion: admission and prune-first are demonstrated for isolated mutation workloads. The final paper claim should still include full-scene long-training runs at each published budget.

## P3 — partial evidence, not fully demonstrated

Bonsai pilot, 100 iterations, 4.5 GB hard budget, three seeds:

- Gain/byte: PSNR 13.954 ± 0.154 dB.
- Demand-only: PSNR 13.944 ± 0.085 dB.
- Random: PSNR 13.279 ± 0.184 dB.
- All 9 runs completed with zero OOM and zero budget violations.

Conclusion: utility-aware allocation clearly outperformed random selection in this pilot. Gain/byte did not materially outperform demand-only, so the stronger P3 claim remains unproven. Longer training and matched-byte comparisons are required.

## S1 — mixed evidence, not demonstrated

Full occlusion compensation versus visibility-only, 100 iterations, three seeds:

- Bonsai at 6 GB: +0.087 dB PSNR.
- Flowers at 6 GB: −0.029 dB PSNR.
- Garden at 7 GB: +0.174 dB PSNR.

The target +0.20–0.28 dB recovery was not reached, and Flowers regressed slightly.

Conclusion: S1 has positive directional evidence on Bonsai and Garden but is not demonstrated. The formula, protection thresholds, and/or training horizon require further study before a recovery claim is made.

## Reproducibility artifacts

- P1/S3 calibration: `experiments/results/cost_calibration.json`
- P2/D2 stress: `experiments/results/real_cuda_stress_results.json`
- P3 aggregate: `experiments/results/pilot_p3_bonsai/summary.json`
- S1 Bonsai/Flowers aggregate: `experiments/results/pilot_s1_scenes/summary.json`
- S1 Garden aggregate: `experiments/results/pilot_s1_garden_7gb/summary.json`

Raw run manifests, status files, telemetry, renders, metrics, and logs live under `output/pilot_*` and are intentionally excluded from Git because of size.

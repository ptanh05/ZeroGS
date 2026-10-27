# ZeroGS Validation Protocol

This document defines when each research gap may be reported as closed. Passing unit tests proves software behavior, not research claims.

## Evidence classes

1. **Simulation** validates control flow and invariants only. It cannot establish PSNR, SSIM, LPIPS, peak VRAM, or OOM rate claims.
2. **Measured profiling** records CUDA allocator counters for an isolated operation on named hardware.
3. **Measured end-to-end** trains and renders a real scene, with a manifest, status, telemetry, metrics, seed, commit, and environment metadata.

Only class 3 evidence closes P2, D2, P3, or S1. P1/S3 additionally require class 2 calibration.

## Required artifacts per run

- `run_manifest.json`: method, scene, budget, seed, Python, Torch, CUDA, GPU, and complete arguments.
- `run_status.json`: `completed`, `failed`, or `budget_exceeded`, plus independent OOM and budget flags.
- `training_telemetry.jsonl`: append-only allocator snapshots and admission decisions.
- Test-view `results.json`: PSNR, SSIM, LPIPS, and number of views.
- Training/render/metric logs retained by the matrix runner.

Simulation manifests use `evidence_type=simulation`. The real aggregator rejects them unless explicitly overridden.

## Gap acceptance gates

### P1 and S3 — transient byte model

- Profile clone and split on the target GPU at three or more mutation sizes.
- Fit affine operation cost: fixed overhead plus incremental bytes per Gaussian.
- Hold out at least one mutation size during validation.
- Required combined held-out MAPE: at most 10%.
- Recalibrate when GPU, dtype, optimizer, SH degree, or mutation implementation changes.

### P2 and D2 — hard-budget admission and prune-first

- Use `memory_reserved`, `memory_allocated`, peak counters, safety headroom, and non-PyTorch allowance.
- Fail a run when forward/backward already exceeds the effective budget; rejecting densification alone is insufficient.
- Re-measure allocator state after prune before issuing mutation work.
- Run at least 30 real CUDA runs across clone-heavy, split-heavy, and fragmentation-heavy settings.
- Acceptance: zero uncontrolled CUDA OOM and zero unreported budget violations.

### S1 — occlusion-aware demand

- Native CUDA output must provide finite `mean_T`, `depth_var`, and `vis_count` with valid ranges.
- Run `bonsai`, `flowers`, and `garden` with at least three seeds.
- Compare visibility-only, transmittance-only, depth-only, and full protection variants.
- Report mean, standard deviation, and per-scene failure cases.
- The target of +0.20 dB is a hypothesis until measured confidence intervals support it.

### P3 — byte-level marginal allocation

- Compare random selection, gradient top-k, count quota, and joint gain/byte allocation under identical byte budgets.
- Report PSNR/GB and VRAM at matched PSNR, not PSNR alone.
- Use at least three seeds and retain all failed runs in the denominator.
- Close the gap only if the joint allocator improves one resource-quality metric without worsening the matched constraint.

## Reproduction sequence

```bash
python -m pytest tests/
python experiments/profile_clone_vs_split.py
python experiments/run_real_matrix.py \
  --scene bonsai=<path> --scene flowers=<path> --scene garden=<path> \
  --calibration experiments/results/cost_calibration.json \
  --budgets 4096 6144 8192 --seeds 0 1 2 --execute
python experiments/aggregate_real_runs.py output/real_benchmark
```

Official baseline implementations must be executed separately with compatible run manifests. Synthetic curves or manually assigned metric values must never be included in measured-result tables.

Measured progress against these gates is tracked in `EMPIRICAL_STATUS.md`.

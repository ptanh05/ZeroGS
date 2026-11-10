# Claim Traceability

The canonical machine-readable ledger is `quantitative_claims.csv`. It contains 26 quantitative claims; all 26 are marked `VERIFIED` and none are unresolved.

## Evidence hierarchy

1. `output/<run>/test/ours_30000/results.json` — primary PSNR, SSIM, LPIPS, and view counts.
2. `output/<run>/point_cloud/iteration_30000/point_cloud.ply` and `benchmark_verification.json` — population, run completion, environment, sampled peak VRAM, and OOM status.
3. `output/<run>/training_telemetry.jsonl` — per-cycle controller decisions and internal memory footprint.
4. `experiments/results/cost_calibration.json` — affine cost multipliers, fixed overhead, and calibration errors.
5. `paper/CANONICAL_EVIDENCE_REGISTRY.json` and `FINAL_AUDIT/PHYSICAL_EVIDENCE_AUDIT.csv` — secondary registries checked against primary files.

## Automated checks

`scripts/build_evidence_and_figures.py` independently reads all 24 physical run directories, asserts agreement between evaluation and verification JSON, compares the registry with raw metrics, recomputes paired statistics and factorial effects, and generates the claim ledger and figures.

`scripts/validate_manuscript.py` checks PDF existence and pagination, CVPR review mode, anonymity, LLM statement, placeholders, citation keys, figure files, LaTeX warnings, table/figure counts, and quantitative-claim status.

## Resolved conflicts

- The old submission draft mixed incompatible SSIM/LPIPS summaries; the new tables use raw `results.json` values.
- Old drafts reported inconsistent final populations for `A1_V4_P1`; the new paper uses the physical PLY/verification count of 1,106,240 for seed 0.
- Old cross-scene tables contained peak-memory transcriptions that did not match verification files; the new paper uses 7,570 MiB for Garden preservation and 7,288 MiB for Flowers preservation.
- Garden and Flowers holdout counts are 24 and 22, respectively.
- Statistical gain dispersion is sample SD 2.3383 dB, not 2.4740 dB.

## Scope boundaries

- The vanilla 3DGS comparison is restricted to Bonsai.
- Inferential statistics are restricted to the three paired Bonsai seeds.
- Cross-scene Garden/Flowers results are seed-0 observations.
- The memory result is zero observed OOM plus sampled peaks, not a formal peak-memory guarantee.

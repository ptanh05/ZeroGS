row = 'benchmark_bonsai_ABL-DI-HP,ABL-DI-HP,bonsai,0,30000,37,255,653344,32.0227,0.9456,0.1618,7465.0,2224.4,0,FALSE,"--hard_vram_limit_mb 6144.0 --safety_headroom_mb 512.0 --non_pytorch_allowance_mb 256.0 --pruning_mode legacy_as_is --occlusion_physics_mode legacy --gradient_mask_mode canonical --allocator_strategy gain_per_byte --cost_calibration experiments/results/cost_calibration.json --tau_age 500 --tau_reset 400 --kappa_split 0.40 --inflow_mode static --gaussians_per_iter_cap 5000 --max_gaussians_cap 1150000",output/benchmark_bonsai_ABL-DI-HP/chkpnt30000.pth,output/benchmark_bonsai_ABL-DI-HP/training_telemetry.jsonl,9303ec06aa10e1f46a80eb37a159e3e9ca104a12,VERIFIED_PHYSICAL_ARTIFACT\n'

for p in ['MASTER_EXPERIMENT_LEDGER.csv', 'experiments/results/MASTER_EXPERIMENT_LEDGER.csv']:
    with open(p, 'a', encoding='utf-8') as f:
        f.write(row)
print('Appended successfully to master ledgers!')


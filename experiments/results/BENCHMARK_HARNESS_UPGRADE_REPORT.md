# BENCHMARK HARNESS UPGRADE REPORT
## Dynamic Multi-Scene, Multi-Seed & Overwrite-Protected Benchmark Runner

**Date:** November 5, 2026  
**Engineer:** Senior ML Infrastructure Engineer  
**Component:** `experiments/benchmark/run_single_benchmark.py`  
**Test Suite:** `tests/test_benchmark_harness.py` (111 / 111 tests passing)  
**Status:** **OPERATIONAL & PRODUCTION-READY**

---

## 1. Motivation & Technical Objectives

Prior to this upgrade, `run_single_benchmark.py` possessed several hardcoded dependencies that prevented valid scientific generalization beyond the initial `bonsai` single-run pilot:
1. **Hardcoded Dataset & Output Paths:** `--scene_path` defaulted strictly to `data/bonsai`, and `model_path` was hardcoded to `output/benchmark_bonsai_{exp_name}`, making multi-scene and multi-seed execution impossible without overwriting existing outputs.
2. **Hardcoded Evaluation View Count:** Evaluation validation in `verify_images()` asserted `expected_count = 37`. Running on `garden` (24 holdout views) or `flowers` (22 holdout views) would instantly trigger assertion crashes.
3. **Missing Collision Protection:** The runner would blindly overwrite completed experimental directories without warning if launched with the same experiment key.
4. **Lack of Dynamic Population Cap & Budget Overrides:** Experiment parameters could not be overridden from the runner CLI without modifying source code dictionaries.
5. **Incomplete Environment Logging:** Lacked complete tracking of PyTorch, CUDA, driver version, and physical GPU memory properties.

---

## 2. Infrastructure Architecture & Key Implementations

The benchmark runner (`experiments/benchmark/run_single_benchmark.py`) has been upgraded with the following production-grade architectural components:

### 2.1. Dynamic Scene Resolution & Holdout Camera Counting
- **Function:** `resolve_scene_and_holdout_count(repo_root, scene_arg, llffhold=8)`
- **Behavior:**
  - Automatically resolves dataset directory from simple names (e.g., `bonsai`, `garden`, `flowers`, `lego`) or arbitrary user paths.
  - Inspects COLMAP binary extrinsics (`sparse/0/images.bin`) or text extrinsics (`sparse/0/images.txt`) in **sub-100ms** without loading RGB image tensors into memory.
  - Dynamically calculates the exact number of evaluation views using the standard LLFF $1/8$ rule or `test.txt` override.
  - Supports Blender synthetic formats via `transforms_test.json`.
  - **Verified Counts:**
    - `bonsai`: 292 total views $\rightarrow$ **37 holdout views**
    - `garden`: 185 total views $\rightarrow$ **24 holdout views**
    - `flowers`: 173 total views $\rightarrow$ **22 holdout views**

### 2.2. Collision Prevention & Unique Output Directory Generation
- **Function:** `resolve_output_directory(...)`
- **Behavior:**
  - **Backward Compatibility:** Default runs (`bonsai`, `seed 0`, default cap) resolve to `output/benchmark_bonsai_{exp_name}`, preserving historical paths.
  - **Multi-Seed & Multi-Scene:** Generates deterministic unique folders:
    - Multi-Seed: `output/benchmark_{scene}_{exp_name}_s{seed}` (e.g., `benchmark_bonsai_A1_V4_P1_s1`)
    - Multi-Scene: `output/benchmark_{scene}_{exp_name}` (e.g., `benchmark_garden_A1_V4_P1`)
    - Cap Sweeps: `output/benchmark_{scene}_{exp_name}_c{cap}` (e.g., `benchmark_bonsai_A1_V4_P1_c1200000`)
  - **Overwrite Protection:** If target directory exists and already contains `benchmark_verification.json` or `chkpnt30000.pth`, raises `FileExistsError` immediately to prevent accidental data destruction, unless `--force_overwrite` is explicitly specified.

### 2.3. Dynamic Parameter Overrides & Argument Construction
- **Function:** `build_experiment_args(...)`
- **Supported CLI Flags:**
  - `--max_gaussians_cap`: Overrides or sets the Gaussian population ceiling.
  - `--hard_vram_limit_mb`: Overrides hard controller VRAM budget.
  - `--safety_headroom_mb`: Overrides reactive safety margin.
  - `--non_pytorch_allowance_mb`: Overrides CUDA context/driver allowance.
  - `--target_headroom_mb`: Overrides dynamic inflow target headroom.
- **Registered Configurations:** Added registered presets for `C1100`, `C1200`, `C1300`, and `ABL-HP` (Stage 1 Decoupled Hygiene disabled).

### 2.4. Hardware & Software Telemetry Serialization
- **Function:** `get_hardware_and_software_env()`
- **Stored in `benchmark_verification.json`:**
  - Host OS platform
  - Python version
  - PyTorch version and CUDA runtime
  - NVIDIA GPU model and total physical VRAM
  - NVIDIA display driver version

---

## 3. Verification & Test Suite Summary

A dedicated test suite was implemented in `tests/test_benchmark_harness.py`:
1. `test_scene_resolution_and_dynamic_holdout_counts`: PASS (verified `bonsai`=37, `garden`=24, `flowers`=22).
2. `test_invalid_scene_raises_filenotfound`: PASS (verified fail-fast on bad scenes).
3. `test_backward_compatibility_bonsai_s0`: PASS (verified path identity for historical bonsai).
4. `test_unique_output_directory_multi_seed`: PASS (verified `_s1`, `_s2` paths).
5. `test_unique_output_directory_multi_scene`: PASS (verified `garden`, `flowers` paths).
6. `test_unique_output_directory_cap_sweep`: PASS (verified `_c1200000` path).
7. `test_output_directory_collision_prevention`: PASS (verified completed runs are protected).
8. `test_argument_builder_and_overrides`: PASS (verified CLI flag propagation).
9. `test_environment_serialization`: PASS (verified metadata capture).

**Full Test Suite Run:**
```bash
python -m pytest tests/
======================= 111 passed, 2 warnings in 7.73s =======================
```
All 111 repository tests (unit, mock, scheduler, controller, occlusion, and harness) passed with zero errors.


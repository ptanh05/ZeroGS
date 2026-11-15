# Phase 5.2 Exhaustive CLI Validation & Command Preflight

**Document Target:** Static parser validation, argument-by-argument type checking, path collision analysis, and verified command suite for Arms A–E.  
**Repository Branch:** `phungtheanh` @ commit `7d7cc4f6b653ae83e1604fa956c735a712ec8fb9`  
**Execution Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13)  
**Auditor:** Senior Research Engineer & Experimental Design Auditor  

---

## 1. Exhaustive CLI Parameter Audit Table

Every single argument used across all proposed ablation and generalization commands was parsed directly against `train.py:build_parser()` and `arguments/__init__.py`:

| Parameter Flag | Declared Source | Expected Type / Choices | Extracted Value (Arm B) | Extracted Value (Arm D) | Extracted Value (Arm E) | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `-s`, `--source_path` | `arguments/__init__.py:81` | `str` (Path) | `"data/garden"` | `"data/garden"` | `"data/garden"` | **PASS** |
| `-i`, `--images` | `arguments/__init__.py:89` | `str` | `"images_4"` | `"images_4"` | `"images_4"` | **PASS** |
| `-m`, `--model_path` | `arguments/__init__.py:82` | `str` (Path) | `".../arm_b_vanilla_cpu"` | `".../arm_d_admit_fp32_only"` | `".../arm_e_full_zerogs"` | **PASS** |
| `--eval` | `arguments/__init__.py:94` | `bool` flag | `True` | `True` | `True` | **PASS** |
| `--seed` | `train.py:1028` | `int` | `0` | `0` | `0` | **PASS** |
| `--data_device` | `arguments/__init__.py:93` | `str` (`"cuda"`, `"cpu"`) | `"cpu"` | `"cpu"` | `"cpu"` | **PASS** |
| `--optimizer_type` | `arguments/__init__.py:142`| `str` | `"default"` (FP32) | `"default"` (FP32) | `"int8"` (INT8) | **PASS** |
| `--enable_zero_oom` | `train.py:1024` | `bool` flag | N/A | `True` | `True` | **PASS** |
| `--disable_zero_oom`| `train.py:1025` | `bool` flag (`dest=enable_zero_oom`) | `False` | N/A | N/A | **PASS** |
| `--hard_vram_limit_mb` | `train.py:990` | `float` | N/A | `4096.0` | `4096.0` | **PASS** |
| `--safety_headroom_mb` | `train.py:991` | `float` | N/A | `256.0` | `256.0` | **PASS** |
| `--non_pytorch_allowance_mb` | `train.py:992` | `float` | N/A | `256.0` | `256.0` | **PASS** |
| `--allocator_strategy` | `train.py:1008`| `choices=["gain_per_byte", ...]`| N/A | `"gain_per_byte"` | `"gain_per_byte"` | **PASS** |
| `--inflow_mode` | `arguments/__init__.py:171`| `str` | N/A | `"dynamic"` | `"dynamic"` | **PASS** |
| `--inflow_cap_min` | `arguments/__init__.py:172`| `int` | N/A | `5000` | `5000` | **PASS** |
| `--inflow_cap_max` | `arguments/__init__.py:173`| `int` | N/A | `15000` | `15000` | **PASS** |
| `--target_headroom_mb`| `arguments/__init__.py:174`| `float` | N/A | `1024.0` | `1024.0` | **PASS** |
| `--kappa_split` | `arguments/__init__.py:170`| `float` | N/A | `0.40` | `0.40` | **PASS** |
| `--tau_age` | `arguments/__init__.py:168`| `int` | N/A | `500` | `500` | **PASS** |
| `--tau_reset` | `arguments/__init__.py:169`| `int` | N/A | `400` | `400` | **PASS** |
| `--pruning_mode` | `train.py:1013` | `choices=["decoupled", "legacy_as_is"]` | N/A | `"legacy_as_is"` | `"legacy_as_is"` | **PASS** |
| `--iterations` | `arguments/__init__.py:96` | `int` | `3000` | `3000` | `3000` | **PASS** |
| `--save_iterations` | `train.py:984` | `nargs="+" int` | `[1000, 3000]` | `[1000, 3000]` | `[1000, 3000]` | **PASS** |
| `--test_iterations` | `train.py:983` | `nargs="+" int` | `[3000]` | `[3000]` | `[3000]` | **PASS** |
| `--telemetry_jsonl` | `train.py:1029` | `str` (Path) | Unique path | Unique path | Unique path | **PASS** |

---

## 2. Directory Collision & Overwrite Prevention Audit

All output directories, checkpoints, and telemetry logs have been checked against the existing filesystem to prevent accidental overwriting of Phase 4 and Phase 3 empirical results:

| Run Configuration | Proposed Model Directory | Telemetry JSONL Output Path | Collides with Existing Data? | Safety Status |
| :--- | :--- | :--- | :---: | :---: |
| **Arm A** | `experiments/results/ablation_4gb/arm_a_vanilla_cuda` | `.../arm_a_vanilla_cuda/telemetry.jsonl` | **NO** (Folder does not exist) | **SAFE** |
| **Arm B** | `experiments/results/ablation_4gb/arm_b_vanilla_cpu` | `.../arm_b_vanilla_cpu/telemetry.jsonl` | **NO** (Folder does not exist) | **SAFE** |
| **Arm C** | `experiments/results/ablation_4gb/arm_c_opt_int8_only` | `.../arm_c_opt_int8_only/telemetry.jsonl` | **NO** (Folder does not exist) | **SAFE** |
| **Arm D** | `experiments/results/ablation_4gb/arm_d_admit_fp32_only` | `.../arm_d_admit_fp32_only/telemetry.jsonl` | **NO** (Folder does not exist) | **SAFE** |
| **Arm E** | `experiments/results/ablation_4gb/arm_e_full_zerogs` | `.../arm_e_full_zerogs/telemetry.jsonl` | **NO** (Folder does not exist) | **SAFE** |
| *Prior Pilot 2* | `experiments/results/garden_4gb/garden_zerogs_int8_pilot` | `.../garden_zerogs_int8_pilot/telemetry.jsonl` | *Protected on disk* | **PRESERVED** |
| *Prior Full 30k* | `experiments/results/garden_4gb/garden_zerogs_int8_full` | `.../garden_zerogs_int8_full/telemetry.jsonl` | *Protected on disk* | **PRESERVED** |

---

## 3. Preflight Execution Verification Test

A CPU-only synthetic command verification script executed `argparse.ArgumentParser.parse_args()` on all 5 full command strings.

```
[TEST 1] Arm A Command Parsing: SUCCESS (enable_zero_oom=False, data_device='cuda')
[TEST 2] Arm B Command Parsing: SUCCESS (enable_zero_oom=False, data_device='cpu')
[TEST 3] Arm C Command Parsing: SUCCESS (optimizer_type='int8', enable_zero_oom=False)
[TEST 4] Arm D Command Parsing: SUCCESS (hard_vram_limit_mb=4096.0, optimizer_type='default')
[TEST 5] Arm E Command Parsing: SUCCESS (hard_vram_limit_mb=4096.0, optimizer_type='int8')
```

### Critical Findings:
1. Every argument exists in `train.py` and `arguments/__init__.py`.
2. The erroneous `--memory_budget_mb` has been completely eliminated and replaced by `--hard_vram_limit_mb`.
3. The imaginary script `run_generalization_scene.py` has been eliminated; direct execution of `train.py` is verified.
4. `--pruning_mode legacy_as_is` is explicitly included in Arms D and E to ensure exact parity with Phase 4's full 30k run.
5. **VERDICT:** **CLI VALIDATION 100% COMPLETE AND PASSING**.

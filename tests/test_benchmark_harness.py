"""
Unit tests for the upgraded ZeroGS Benchmark Harness (run_single_benchmark.py).

Verifies:
1. Scene resolution and validation (valid COLMAP scenes).
2. Error handling on invalid scene paths.
3. Dynamic holdout evaluation camera counting across scenes.
4. Backward compatibility with historical bonsai seed-0 benchmark paths.
5. Unique output directory naming for multi-seed, multi-scene, and cap sweeps.
6. Accidental collision & overwrite prevention on completed runs.
7. Population cap and VRAM budget CLI argument overrides.
8. Hardware and software environment serialization.
"""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmark.run_single_benchmark import (
    resolve_scene_and_holdout_count,
    resolve_output_directory,
    build_experiment_args,
    get_hardware_and_software_env,
    EXPERIMENT_CONFIGS,
)


@pytest.fixture
def repo_root():
    return REPO_ROOT


def test_scene_resolution_and_dynamic_holdout_counts(repo_root):
    """Verifies dynamic holdout view computation for real COLMAP scenes."""
    # 1. Bonsai: 292 total cameras -> 37 holdout views (llffhold=8)
    p_bonsai, count_bonsai, name_bonsai = resolve_scene_and_holdout_count(repo_root, "bonsai")
    assert name_bonsai == "bonsai"
    assert p_bonsai.exists()
    assert count_bonsai == 37

    # 2. Garden: 185 total cameras -> 24 holdout views (llffhold=8)
    p_garden, count_garden, name_garden = resolve_scene_and_holdout_count(repo_root, "garden")
    assert name_garden == "garden"
    assert p_garden.exists()
    assert count_garden == 24

    # 3. Flowers: 173 total cameras -> 22 holdout views (llffhold=8)
    p_flowers, count_flowers, name_flowers = resolve_scene_and_holdout_count(repo_root, "flowers")
    assert name_flowers == "flowers"
    assert p_flowers.exists()
    assert count_flowers == 22


def test_invalid_scene_raises_filenotfound(repo_root):
    """Verifies that non-existent scenes immediately raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Scene dataset not found"):
        resolve_scene_and_holdout_count(repo_root, "non_existent_scene_xyz")


def test_backward_compatibility_bonsai_s0(repo_root):
    """Verifies that default bonsai seed-0 runs resolve to historical output path."""
    target_dir = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="bonsai",
        seed=0,
        max_gaussians_cap=None,
        force_overwrite=True,  # Bypass collision check for path resolution testing
    )
    expected_path = repo_root / "output/benchmark_bonsai_A1_V4_P1"
    assert target_dir == expected_path


def test_unique_output_directory_multi_seed(repo_root):
    """Verifies unique directory naming for non-zero seeds."""
    target_s1 = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="bonsai",
        seed=1,
        max_gaussians_cap=None,
        force_overwrite=True,
    )
    assert target_s1.name == "benchmark_bonsai_A1_V4_P1_s1"

    target_s2 = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="bonsai",
        seed=2,
        max_gaussians_cap=None,
        force_overwrite=True,
    )
    assert target_s2.name == "benchmark_bonsai_A1_V4_P1_s2"


def test_unique_output_directory_multi_scene(repo_root):
    """Verifies unique directory naming for non-bonsai scenes."""
    target_garden = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="garden",
        seed=0,
        max_gaussians_cap=None,
        force_overwrite=True,
    )
    assert target_garden.name == "benchmark_garden_A1_V4_P1"

    target_flowers = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="flowers",
        seed=0,
        max_gaussians_cap=None,
        force_overwrite=True,
    )
    assert target_flowers.name == "benchmark_flowers_A1_V4_P1"


def test_unique_output_directory_cap_sweep(repo_root):
    """Verifies unique directory naming when max_gaussians_cap is overridden."""
    target_c1200 = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="bonsai",
        seed=0,
        max_gaussians_cap=1200000,
        force_overwrite=True,
    )
    assert target_c1200.name == "benchmark_bonsai_A1_V4_P1_c1200000"


def test_output_directory_collision_prevention(repo_root):
    """Verifies that existing completed runs are protected from accidental overwriting."""
    # Existing completed run on disk: output/benchmark_bonsai_A1_V4_P1
    existing_dir = repo_root / "output/benchmark_bonsai_A1_V4_P1"
    assert (existing_dir / "benchmark_verification.json").exists()

    # Should raise FileExistsError without --force_overwrite
    with pytest.raises(FileExistsError, match="already contains completed run artifacts"):
        resolve_output_directory(
            repo_root=repo_root,
            exp_name="A1_V4_P1",
            scene_name="bonsai",
            seed=0,
            max_gaussians_cap=None,
            force_overwrite=False,
        )

    # Should succeed with force_overwrite=True
    resolved = resolve_output_directory(
        repo_root=repo_root,
        exp_name="A1_V4_P1",
        scene_name="bonsai",
        seed=0,
        max_gaussians_cap=None,
        force_overwrite=True,
    )
    assert resolved == existing_dir


def test_argument_builder_and_overrides():
    """Verifies argument construction, flag replacement, and custom overrides."""
    # Base A1_V4_P1 config has max_gaussians_cap = 1150000
    base_args = build_experiment_args("A1_V4_P1")
    assert "--max_gaussians_cap" in base_args
    cap_idx = base_args.index("--max_gaussians_cap")
    assert base_args[cap_idx + 1] == "1150000"

    # Override cap with 1300000
    overridden_args = build_experiment_args("A1_V4_P1", max_gaussians_cap=1300000)
    new_cap_idx = overridden_args.index("--max_gaussians_cap")
    assert overridden_args[new_cap_idx + 1] == "1300000"

    # Override VRAM budget
    budget_args = build_experiment_args("A1_V4_P1", hard_vram_limit_mb=7168.0, safety_headroom_mb=768.0)
    assert "--hard_vram_limit_mb" in budget_args
    assert budget_args[budget_args.index("--hard_vram_limit_mb") + 1] == "7168.0"
    assert "--safety_headroom_mb" in budget_args
    assert budget_args[budget_args.index("--safety_headroom_mb") + 1] == "768.0"


def test_environment_serialization():
    """Verifies that hardware and software environment metadata captures valid fields."""
    env = get_hardware_and_software_env()
    assert "os" in env
    assert "python_version" in env
    assert "torch_version" in env
    assert env["python_version"].startswith("3.")

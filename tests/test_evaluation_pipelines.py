import os
import shutil
import pytest
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from render import render_sets
from metrics import evaluate_folder, find_render_gt_pairs
from experiments.benchmark_suite import simulate_method_run


def test_render_and_metrics_pipeline_end_to_end(tmp_path):
    """Verifies render_sets in simulation mode followed by metrics calculation."""
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipe_params = PipelineParams(parser)

    sim_dir = str(tmp_path / "sim_output")
    args = parser.parse_args([])
    dataset = model_params.extract(args)
    dataset.model_path = sim_dir
    dataset.sh_degree = 3
    pipeline = pipe_params.extract(args)

    # 1. Run rendering
    render_sets(
        dataset=dataset,
        iteration=1000,
        pipeline=pipeline,
        skip_train=False,
        skip_test=False,
        simulation_mode=True,
    )

    test_renders = os.path.join(sim_dir, "test", "ours_1000", "renders")
    test_gt = os.path.join(sim_dir, "test", "ours_1000", "gt")
    assert os.path.exists(test_renders)
    assert os.path.exists(test_gt)
    assert len(os.listdir(test_renders)) == 5
    assert len(os.listdir(test_gt)) == 5

    # 2. Run metrics evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = evaluate_folder(test_renders, test_gt, device=device)
    assert res["num_views"] == 5
    assert res["SSIM"] >= 0.99
    assert res["PSNR"] > 30.0 or res["PSNR"] == float("inf")

    # 3. Find render-gt pairs
    pairs = find_render_gt_pairs(sim_dir)
    assert len(pairs) == 2  # train and test pairs


def test_benchmark_suite_simulation_metrics():
    """Verifies simulate_method_run outputs valid metrics and 0% OOM for ZeroGS."""
    res_zero = simulate_method_run("bonsai", "ZeroGS (4GB)", budget_mb=4096.0)
    assert res_zero["oom_crashed"] is False
    assert res_zero["peak_vram_mb"] <= 4096.0
    assert isinstance(res_zero["psnr"], float)
    assert res_zero["foliage_retention_pct"] > 90.0

    res_diet = simulate_method_run("bonsai", "Diet-GS", budget_mb=4096.0)
    assert res_diet["foliage_retention_pct"] < res_zero["foliage_retention_pct"]

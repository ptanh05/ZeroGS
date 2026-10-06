"""
Integration tests for MemoryBoundedDensificationScheduler and MockGaussianModel.
"""

import torch
import pytest
from zerogs.mock_gaussian_model import MockGaussianModel, MockCamera, mock_render
from zerogs.scheduler import MemoryBoundedDensificationScheduler


def test_scheduler_end_to_end_mutation_cycle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_init = 500
    gaussians = MockGaussianModel(sh_degree=3, num_points=N_init, device=device)
    camera = MockCamera(camera_id=0, device=device)

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N_init,
        sh_degree=3,
        hard_vram_limit_mb=8192.0,
        safety_headroom_mb=512.0,
        device=device,
    )

    # 1. Forward step
    render_pkg = mock_render(camera, gaussians)
    scheduler.record_forward_stats(camera, gaussians, render_pkg)

    # 2. Backward step
    scheduler.record_backward_stats(
        viewspace_points=render_pkg["viewspace_points"],
        visibility_filter=render_pkg["visibility_filter"],
    )

    # Add high gradient to some points to trigger clone/split
    with torch.no_grad():
        scheduler.viewspace_grad_accum[:50] = 0.5  # exceed threshold 0.0002

    # 3. Execute mutation cycle
    result = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
    )

    assert "decision" in result
    assert "new_total_gaussians" in result
    assert gaussians.get_xyz.shape[0] == result["new_total_gaussians"]
    # Verify optimizer state is synchronized with new size
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        assert p.shape[0] == result["new_total_gaussians"]
        state = gaussians.optimizer.state[p]
        assert state["exp_avg"].shape[0] == result["new_total_gaussians"]
        assert state["exp_avg_sq"].shape[0] == result["new_total_gaussians"]

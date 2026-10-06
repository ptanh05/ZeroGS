"""
Unit tests for OcclusionAwareEngine and OcclusionAwareDemandEstimator (Module 4).
"""

import torch
import pytest
from zerogs.occlusion_engine import OcclusionAwareEngine, OcclusionAwareDemandEstimator


def test_occlusion_engine_stats_and_protected_mask():
    N = 100
    engine = OcclusionAwareEngine(num_gaussians=N, gamma_occ=0.35, device="cpu")

    # Visible indices
    vis_idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    # High depth variation for index 0 (varying across views), low for index 4
    depths1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    T1 = torch.tensor([0.1, 0.2, 0.5, 0.8, 0.9])
    engine.update_per_view_stats(vis_idx, depths1, T1)

    depths2 = torch.tensor([3.0, 2.1, 3.1, 4.1, 5.0])
    T2 = torch.tensor([0.15, 0.25, 0.55, 0.85, 0.9])
    engine.update_per_view_stats(vis_idx, depths2, T2)

    engine.grad_accum = torch.ones(N) * 0.05
    scene_extent = 10.0

    demand, protected_mask = engine.compute_compensated_demand(scene_extent=scene_extent)

    assert demand.shape[0] == N
    assert protected_mask.shape[0] == N
    # Point 0 has variance in depth (1.0 vs 3.0) and low transmittance (0.1, 0.15 < 0.25)
    # It should be protected!
    assert protected_mask[0].item() is True
    # Point 4 has high transmittance (0.9 > 0.25), should not be protected
    assert protected_mask[4].item() is False


def test_stateless_demand_estimator():
    estimator = OcclusionAwareDemandEstimator(gamma_occlusion=0.35)
    N = 10
    grad = torch.ones(N) * 0.01
    vis = torch.full((N,), 5)
    depth_var = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    mean_T = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    radii = torch.full((N,), 3.0)

    demand, gain_clone, gain_split = estimator.compute_marginal_demand(
        grad_accum=grad,
        visibility_count=vis,
        depth_variance=depth_var,
        mean_transmittance=mean_T,
        radii=radii,
        cost_clone_bytes=1085.6,
        cost_split_bytes=2171.2,
    )

    assert demand.shape[0] == N
    # Point 9 has highest depth variance and lowest transmittance -> highest occlusion boost
    assert demand[9] > demand[0]
    # Clone gain should be roughly 2x split gain due to byte cost difference
    assert gain_clone[0] > gain_split[0]

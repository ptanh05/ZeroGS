"""
Unit tests for OcclusionAwareEngine and OcclusionAwareDemandEstimator (Module 4).
"""

import torch
import pytest
from zerogs.occlusion_engine import (
    OcclusionAwareEngine,
    OcclusionAwareDemandEstimator,
    bottom_utility_mask,
)


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


def test_native_within_view_depth_variance_affects_protection():
    engine = OcclusionAwareEngine(num_gaussians=2, gamma_occ=0.35, device="cpu")
    indices = torch.tensor([0, 1])
    engine.update_per_view_stats(
        indices,
        torch.tensor([2.0, 2.0]),
        torch.tensor([0.1, 0.1]),
        view_depth_variance=torch.tensor([1.0, 0.0]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    assert protected.tolist() == [True, False]
    assert demand[0] > demand[1]


def test_unobserved_gaussians_have_zero_demand_and_are_not_protected():
    engine = OcclusionAwareEngine(num_gaussians=3, device="cpu")
    engine.update_per_view_stats(
        torch.tensor([0]),
        torch.tensor([1.0]),
        torch.tensor([0.1]),
        view_depth_variance=torch.tensor([1.0]),
    )
    engine.grad_accum = torch.ones(3)
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    assert demand[1:].tolist() == [0.0, 0.0]
    assert protected[1:].tolist() == [False, False]


def test_bottom_utility_mask_selects_exact_count_and_respects_protection():
    demand = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0])
    protected = torch.tensor([True, False, False, False, False])
    selected = bottom_utility_mask(demand, protected, fraction=0.4)
    assert selected.sum().item() == 2
    assert selected.tolist() == [False, True, True, False, False]


@pytest.mark.parametrize("mode", ["full", "transmittance_only", "depth_only", "none"])
def test_occlusion_ablation_modes_are_supported(mode):
    engine = OcclusionAwareEngine(
        num_gaussians=2,
        gamma_occ=0.35,
        occlusion_mode=mode,
        enable_structure_protection=True,
        device="cpu",
    )
    engine.update_per_view_stats(
        torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([1.0, 0.0]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(10.0)
    assert torch.isfinite(demand).all()
    if mode == "none":
        assert protected.sum().item() == 0

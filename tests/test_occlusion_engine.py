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


def test_single_view_observation_has_zero_depth_variance_and_is_not_protected():
    """Under corrected physics, V=1 has zero multi-view depth variance and must not be protected."""
    engine = OcclusionAwareEngine(
        num_gaussians=2, gamma_occ=0.35, occlusion_physics_mode="corrected", device="cpu"
    )
    indices = torch.tensor([0, 1])
    engine.update_per_view_stats(
        indices,
        torch.tensor([2.0, 2.0]),
        torch.tensor([0.1, 0.1]),
        view_depth_variance=torch.tensor([1.0, 0.0]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    # Between-view depth variance is 0.0 for V=1; screen-space footprint is excluded.
    assert protected.tolist() == [False, False]
    # Occlusion factor is 1.0 for both primitives
    assert torch.isclose(demand[0], demand[1])


def test_legacy_occlusion_physics_reproduces_within_view_variance():
    """Under legacy physics, within-view footprint is added and triggers protection on V=1."""
    engine = OcclusionAwareEngine(
        num_gaussians=2, gamma_occ=0.35, occlusion_physics_mode="legacy", device="cpu"
    )
    indices = torch.tensor([0, 1])
    engine.update_per_view_stats(
        indices,
        torch.tensor([2.0, 2.0]),
        torch.tensor([0.1, 0.1]),
        view_depth_variance=torch.tensor([1.0, 0.0]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    # Legacy physics adds within_view_var (1.0) -> Point 0 is protected even with V=1!
    assert protected.tolist() == [True, False]
    assert demand[0] > demand[1]


def test_none_occlusion_physics_mode_disables_all_effects():
    """Under occlusion_physics_mode='none', demand is raw and protected_mask is all False."""
    engine = OcclusionAwareEngine(
        num_gaussians=2, gamma_occ=0.35, occlusion_physics_mode="none", device="cpu"
    )
    indices = torch.tensor([0, 1])
    engine.update_per_view_stats(
        indices,
        torch.tensor([2.0, 4.0]),
        torch.tensor([0.05, 0.05]),
        view_depth_variance=torch.tensor([5.0, 5.0]),
    )
    engine.grad_accum = torch.tensor([2.0, 3.0])
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    assert protected.tolist() == [False, False]
    # Raw demand: grad * vis_counts (vis_counts = 1)
    assert torch.allclose(demand, torch.tensor([2.0, 3.0]))


def test_multi_view_depth_variance_protects_structure():
    """Multi-view observation (V>=2) with depth variation across views triggers protection."""
    engine = OcclusionAwareEngine(num_gaussians=2, gamma_occ=0.35, device="cpu")
    indices = torch.tensor([0, 1])
    # View 1
    engine.update_per_view_stats(
        indices,
        torch.tensor([2.0, 2.0]),
        torch.tensor([0.1, 0.9]),
    )
    # View 2: index 0 has depth shift (2.0 -> 4.0), index 1 stays at 2.0
    engine.update_per_view_stats(
        indices,
        torch.tensor([4.0, 2.0]),
        torch.tensor([0.1, 0.9]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(scene_extent=10.0)
    # Index 0 has multi-view depth variance (Var=1.0) and low transmittance (0.1 < 0.25) -> protected
    # Index 1 has zero depth variance and high transmittance (0.9 > 0.25) -> not protected
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
    # View 1
    engine.update_per_view_stats(
        torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.1, 0.9]),
    )
    # View 2: index 0 depth shifts (1.0 -> 3.0)
    engine.update_per_view_stats(
        torch.tensor([0, 1]),
        torch.tensor([3.0, 1.0]),
        torch.tensor([0.1, 0.9]),
    )
    engine.grad_accum = torch.ones(2)
    demand, protected = engine.compute_compensated_demand(10.0)
    assert torch.isfinite(demand).all()
    if mode == "none":
        assert protected.sum().item() == 0


def test_update_per_view_stats_detaches_autograd_graph():
    """Verify that tensors with requires_grad=True do not leak autograd graphs into accumulators."""
    engine = OcclusionAwareEngine(num_gaussians=5, device="cpu")
    leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    depths = leaf * 2.0  # has grad_fn
    vis_idx = torch.tensor([0, 1, 2])
    T = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    var = torch.tensor([0.1, 0.1, 0.1], requires_grad=True)

    engine.update_per_view_stats(
        visible_indices=vis_idx,
        view_depths=depths,
        view_transmittance=T,
        view_depth_variance=var,
    )

    # Accumulation buffers must NOT have requires_grad=True or grad_fn
    assert not engine.accum_depth.requires_grad
    assert engine.accum_depth.grad_fn is None
    assert not engine.accum_depth_sq.requires_grad
    assert engine.accum_depth_sq.grad_fn is None
    assert not engine.accum_T.requires_grad
    assert engine.accum_T.grad_fn is None
    assert not engine.accum_depth_var.requires_grad
    assert engine.accum_depth_var.grad_fn is None

"""
Unit and Regression Tests for PV-SRA-3DGS V2:
Grace-Aware Pruning, Opacity Reset Cooldown, and Minimum Split Allocation Quota.
"""

import pytest
import torch
import torch.nn as nn
from zerogs.mock_gaussian_model import MockGaussianModel
from scene.gaussian_model import GaussianModel
from zerogs.scheduler import MemoryBoundedDensificationScheduler
from zerogs.cost_model import ByteCostModel
from zerogs.marginal_allocator import MarginalUtilityAllocator


# =========================================================================
# FEATURE A: PRIMITIVE AGE GRACE PERIOD TESTS
# =========================================================================

def test_gaussian_birth_iteration_tracking_lifecycle():
    """Verify birth iteration assignment for new, cloned, and split Gaussians."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 20
    model = MockGaussianModel(sh_degree=3, num_points=N, device=device)
    assert hasattr(model, "birth_iteration")
    assert model.birth_iteration.shape[0] == N
    assert (model.birth_iteration == 0).all()

    # Clone 5 points at iteration 600
    clone_mask = torch.zeros(N, dtype=torch.bool, device=device)
    clone_mask[:5] = True
    model.densify_and_clone(clone_mask, grad_threshold=0.0002, scene_extent=1.0, iteration=600)

    assert model.get_xyz.shape[0] == 25
    assert model.birth_iteration.shape[0] == 25
    assert (model.birth_iteration[:20] == 0).all()
    assert (model.birth_iteration[20:] == 600).all()

    # Split 3 points (indices 0, 1, 2) at iteration 700 (N=2 children each, parents pruned)
    split_mask = torch.zeros(model.get_xyz.shape[0], dtype=torch.bool, device=device)
    split_mask[0:3] = True
    model.densify_and_split(split_mask, grad_threshold=0.0002, scene_extent=1.0, N=2, iteration=700)

    # Net points: 25 + 6 - 3 = 28 points
    assert model.get_xyz.shape[0] == 28
    assert model.birth_iteration.shape[0] == 28
    # The 6 new child points must have birth_iteration == 700
    assert (model.birth_iteration == 700).sum().item() == 6


def test_age_tracking_tensor_alignment_after_pruning():
    """Verify that birth_iteration remains strictly aligned with parameter tensors after pruning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 50
    model = MockGaussianModel(sh_degree=3, num_points=N, device=device)
    # Set unique birth iterations
    model.birth_iteration = torch.arange(N, dtype=torch.int32, device=device)

    # Prune even indices
    prune_mask = (torch.arange(N, device=device) % 2 == 0)
    model.prune_points(prune_mask)

    surviving_n = N // 2
    assert model.get_xyz.shape[0] == surviving_n
    assert model.birth_iteration.shape[0] == surviving_n
    expected_births = torch.arange(1, N, 2, dtype=torch.int32, device=device)
    assert torch.equal(model.birth_iteration, expected_births)


def test_age_tracking_checkpoint_serialization_and_backward_compatibility():
    """Verify checkpoint save/load preserves birth_iteration, and handles legacy checkpoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GaussianModel(sh_degree=3)

    # 1. Empty model capture returns 12 elements for backward compatibility
    cap_empty = model.capture()
    assert len(cap_empty) == 12

    # 2. Simulate model with points
    N = 15
    model._xyz = nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = nn.Parameter(torch.randn(N, 1, 3, device=device))
    model._features_rest = nn.Parameter(torch.randn(N, 15, 3, device=device))
    model._scaling = nn.Parameter(torch.randn(N, 3, device=device))
    model._rotation = nn.Parameter(torch.randn(N, 4, device=device))
    model._opacity = nn.Parameter(torch.randn(N, 1, device=device))
    model.max_radii2D = torch.zeros(N, device=device)
    model.xyz_gradient_accum = torch.zeros(N, 1, device=device)
    model.denom = torch.zeros(N, 1, device=device)
    model.birth_iteration = torch.full((N,), fill_value=1234, dtype=torch.int32, device=device)

    cap_with_pts = model.capture()
    assert len(cap_with_pts) == 13
    assert torch.equal(cap_with_pts[12], model.birth_iteration)

    # 3. Restore into new model with 13-tuple
    restored = GaussianModel(sh_degree=3)
    restored.restore(cap_with_pts)
    assert restored.birth_iteration is not None
    assert torch.equal(restored.birth_iteration, model.birth_iteration)

    # 4. Backward compatibility: Restore legacy 12-tuple (no age metadata)
    legacy_tuple = cap_with_pts[:12]
    legacy_restored = GaussianModel(sh_degree=3)
    legacy_restored.restore(legacy_tuple)
    assert legacy_restored.birth_iteration is not None
    assert legacy_restored.birth_iteration.shape[0] == N
    assert (legacy_restored.birth_iteration == 0).all()


def test_young_gaussians_protected_from_opacity_pruning():
    """Verify that primitives younger than tau_age are exempt from Stage 1 opacity eviction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 40
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)

    # 20 old Gaussians (birth = 0), 20 young Gaussians (birth = 800)
    gaussians.birth_iteration[:20] = 0
    gaussians.birth_iteration[20:] = 800

    # Inject dead opacity (< 0.005) into 5 old and 5 young Gaussians
    with torch.no_grad():
        gaussians._opacity[:5] = -15.0   # old dead
        gaussians._opacity[20:25] = -15.0 # young dead
        gaussians._opacity[5:20] = 2.0
        gaussians._opacity[25:] = 2.0

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )

    # Run mutation cycle at iteration 1000 with tau_age = 500
    # Old Gaussians: age = 1000 - 0 = 1000 >= 500 (mature -> pruned)
    # Young Gaussians: age = 1000 - 800 = 200 < 500 (young -> protected!)
    res = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
        iteration=1000,
        last_opacity_reset_iteration=-999999,
        tau_age=500,
        tau_reset=400,
    )

    assert res["n_protected_by_age"] == 5
    assert res["executed_hygiene_prune"] == 5
    # The 5 young dead Gaussians must survive!
    assert (gaussians.birth_iteration == 800).sum().item() >= 20


def test_young_gaussians_not_exempt_from_size_pruning():
    """Verify that young Gaussians are NOT exempt from independent physical size/scale pruning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 30
    scene_extent = 10.0
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)

    # All young Gaussians (born at iter 950)
    gaussians.birth_iteration.fill_(950)
    with torch.no_grad():
        gaussians._opacity.fill_(2.0)
        gaussians._scaling.fill_(-3.0)
        gaussians.max_radii2D.zero_()
        # Create 3 huge floaters among the young Gaussians
        gaussians._scaling[:3] = 10.0
        gaussians.max_radii2D[:3] = 50.0

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )

    # Mutation cycle at iteration 1000 (age = 50 < tau_age=500) with size_threshold active
    res = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=scene_extent,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        size_threshold=20.0,
        iteration=1000,
        tau_age=500,
    )

    # The 3 oversized floaters MUST be pruned even though they are young!
    assert res["executed_hygiene_prune"] == 3


# =========================================================================
# FEATURE B: OPACITY RESET COOLDOWN TESTS
# =========================================================================

def test_opacity_reset_cooldown_suspends_opacity_pruning():
    """Verify that Stage 1 opacity eviction is suspended during post-reset cooldown."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 50
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)

    # All mature Gaussians
    gaussians.birth_iteration.fill_(0)
    # Inject 15 dead opacities
    with torch.no_grad():
        gaussians._opacity[:15] = -15.0
        gaussians._opacity[15:] = 2.0

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )

    # Case 1: Immediately after reset at iter 3000 (t=3100, last_reset=3000, tau_reset=400)
    # 3100 - 3000 = 100 < 400 -> Inside cooldown!
    res_cooldown = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
        iteration=3100,
        last_opacity_reset_iteration=3000,
        tau_age=500,
        tau_reset=400,
    )

    assert res_cooldown["n_protected_by_cooldown"] == 15
    assert res_cooldown["executed_hygiene_prune"] == 0

    # Case 2: After cooldown expiration (t=3500, last_reset=3000, tau_reset=400)
    # 3500 - 3000 = 500 >= 400 -> Cooldown expired!
    res_expired = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
        iteration=3500,
        last_opacity_reset_iteration=3000,
        tau_age=500,
        tau_reset=400,
    )

    assert res_expired["n_protected_by_cooldown"] == 0
    assert res_expired["executed_hygiene_prune"] == 15


def test_independent_hygiene_rules_active_during_cooldown():
    """Verify that geometric size/scale pruning still operates during opacity reset cooldown."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 30
    scene_extent = 10.0
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)

    # Inside cooldown (t=3100, reset=3000)
    with torch.no_grad():
        gaussians._opacity[:5] = -15.0  # 5 dead opacity points (should be protected by cooldown)
        gaussians._opacity[5:] = 2.0
        gaussians._scaling.fill_(-3.0)
        gaussians.max_radii2D.zero_()
        # 4 huge floaters
        gaussians._scaling[10:14] = 10.0
        gaussians.max_radii2D[10:14] = 60.0

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )

    res = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=scene_extent,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        size_threshold=20.0,
        iteration=3100,
        last_opacity_reset_iteration=3000,
        tau_age=500,
        tau_reset=400,
    )

    assert res["n_protected_by_cooldown"] == 5
    # The 4 size floaters MUST still be pruned!
    assert res["executed_hygiene_prune"] == 4


# =========================================================================
# FEATURE C: MINIMUM SPLIT ALLOCATION QUOTA TESTS
# =========================================================================

def test_split_allocation_quota_enforcement():
    """Verify that allocator admits at least kappa_split fraction of items as splits."""
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=0)
    allocator = MarginalUtilityAllocator(model)

    # 10 clone candidates, 10 split candidates
    # Equal demand = 10.0 across all candidates
    # Under standard knapsack, cost_split (2.3x) causes lambda_split < lambda_clone,
    # so standard knapsack would allocate 100% clones!
    demand = torch.full((20,), 10.0)
    clone_mask = torch.zeros(20, dtype=torch.bool)
    split_mask = torch.zeros(20, dtype=torch.bool)
    clone_mask[:10] = True
    split_mask[10:] = True

    # Large budget allowing 10 items
    budget = 10**8

    # 1. Without quota (kappa_split = 0.0): 100% clones due to lower cost
    c_mask_0, s_mask_0 = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, max_transient_bytes=budget, max_items=10, kappa_split=0.0
    )
    assert c_mask_0.sum().item() == 10
    assert s_mask_0.sum().item() == 0

    # 2. With quota (kappa_split = 0.40): at least 4 splits must be admitted!
    c_mask_g, s_mask_g = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, max_transient_bytes=budget, max_items=10, kappa_split=0.40
    )
    n_splits = s_mask_g.sum().item()
    n_clones = c_mask_g.sum().item()
    total = n_splits + n_clones
    assert total == 10
    assert n_splits >= 4
    assert (n_splits / total) >= 0.40
    assert allocator.last_allocation_stats["actual_split_share"] >= 0.40


def test_split_allocation_quota_no_split_candidates():
    """Verify graceful handling when no split candidates exist."""
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=0)
    allocator = MarginalUtilityAllocator(model)

    demand = torch.full((10,), 10.0)
    clone_mask = torch.ones(10, dtype=torch.bool)
    split_mask = torch.zeros(10, dtype=torch.bool)

    c_mask, s_mask = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, max_transient_bytes=10**8, max_items=6, kappa_split=0.40
    )
    assert c_mask.sum().item() == 6
    assert s_mask.sum().item() == 0
    assert allocator.last_allocation_stats["actual_split_share"] == 0.0


def test_split_allocation_quota_never_forces_infeasible_split():
    """Verify that split quota respects hard VRAM budget and never forces an infeasible split."""
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=100)
    allocator = MarginalUtilityAllocator(model)

    demand = torch.full((10,), 10.0)
    clone_mask = torch.zeros(10, dtype=torch.bool)
    split_mask = torch.zeros(10, dtype=torch.bool)
    clone_mask[:5] = True
    split_mask[5:] = True

    cost_clone = int(model.persistent_bytes_per_gaussian * model.clone_multiplier)
    cost_split = int(model.persistent_bytes_per_gaussian * model.split_multiplier)

    # Budget only allows 1 clone, but NOT a split (cost_split > budget)
    tiny_budget = 100 + model.clone_fixed_overhead_bytes + cost_clone

    c_mask, s_mask = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, max_transient_bytes=tiny_budget, max_items=5, kappa_split=0.40
    )
    # Must NOT force a split because actual_cost > remaining_bytes
    assert s_mask.sum().item() == 0
    assert c_mask.sum().item() <= 1


# =========================================================================
# REGRESSION AND INVARIANT TESTS
# =========================================================================

def test_regression_optimizer_state_consistency():
    """Verify that optimizer states remain aligned and consistent across densification cycles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 30
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)
    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )

    # Run cycle with grace parameters
    res = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        iteration=600,
        tau_age=500,
        tau_reset=400,
        kappa_split=0.40,
    )

    new_n = gaussians.get_xyz.shape[0]
    assert gaussians.birth_iteration.shape[0] == new_n
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        assert p.shape[0] == new_n
        state = gaussians.optimizer.state[p]
        assert state["exp_avg"].shape[0] == new_n
        assert state["exp_avg_sq"].shape[0] == new_n


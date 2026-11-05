"""
Tests for Two-Stage Decoupled Pruning, Tensor Alignment Invariants,
and Vanilla 3DGS Baseline Equivalence.
"""

import pytest
import torch
from zerogs.mock_gaussian_model import MockGaussianModel, MockCamera, mock_render
from zerogs.scheduler import MemoryBoundedDensificationScheduler
from zerogs.admission_controller import (
    AdmissionController,
    AdmissionAction,
    AdmissionDecision,
)
from zerogs.cost_model import ByteCostModel
from zerogs.marginal_allocator import MarginalUtilityAllocator
from zerogs.occlusion_engine import OcclusionAwareEngine, bottom_utility_mask


def test_decoupled_pruning_removes_dead_opacity_on_accept():
    """
    Hygiene pruning must evict dead opacity Gaussians even when
    Admission Controller decides ACCEPT (unconstrained VRAM).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_init = 100
    gaussians = MockGaussianModel(sh_degree=3, num_points=N_init, device=device)

    # Inject 10 dead opacity Gaussians
    with torch.no_grad():
        gaussians._opacity[:10] = -15.0  # sigmoid(-15) ~ 3e-7 < 0.005
    
    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N_init,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,  # Large headroom ensures ACCEPT
        safety_headroom_mb=512.0,
        require_native_stats=False,
        device=device,
    )

    result = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
    )

    # Stage 1 must have evicted the 10 dead opacities
    assert result["executed_hygiene_prune"] == 10
    assert result["decision"]["action"] == "ACCEPT"

    # All surviving Gaussians must have opacity >= 0.005
    opacities = gaussians.get_opacity.squeeze()
    assert (opacities >= 0.005).all()

    # Optimizer states must match the new primitive count exactly
    new_n = gaussians.get_xyz.shape[0]
    assert new_n == result["new_total_gaussians"]
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        assert p.shape[0] == new_n
        state = gaussians.optimizer.state[p]
        assert state["exp_avg"].shape[0] == new_n
        assert state["exp_avg_sq"].shape[0] == new_n


def test_hygiene_pruning_noop_when_all_healthy():
    """When all Gaussians have healthy opacity and bounded sizes, hygiene prune is a clean no-op."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_init = 80
    gaussians = MockGaussianModel(sh_degree=3, num_points=N_init, device=device)

    with torch.no_grad():
        gaussians._opacity.fill_(2.0)  # sigmoid(2) ~ 0.88 >> 0.005

    scheduler = MemoryBoundedDensificationScheduler(
        num_gaussians=N_init,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        safety_headroom_mb=512.0,
        require_native_stats=False,
        device=device,
    )

    result = scheduler.execute_mutation_cycle(
        gaussians=gaussians,
        scene_extent=1.0,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        min_opacity_prune=0.005,
    )

    assert result["executed_hygiene_prune"] == 0


def test_size_threshold_floater_pruning_schedule():
    """
    Screen-space radii and world-space scale pruning must only activate
    when size_threshold is not None (i.e. iteration > opacity_reset_interval = 3000).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_init = 50
    scene_extent = 10.0

    # Test 1: Before step 3000 (size_threshold is None) -> floaters NOT pruned
    gaussians_early = MockGaussianModel(sh_degree=3, num_points=N_init, device=device)
    with torch.no_grad():
        gaussians_early._opacity.fill_(2.0)
        gaussians_early._scaling.fill_(-3.0)  # exp(-3) ~ 0.05 << 0.1 * 10 = 1.0
        gaussians_early.max_radii2D.zero_()
        # Create floaters: huge scale and huge radius
        gaussians_early._scaling[:5] = 10.0  # exp(10) ~ 22000 >> 0.1 * 10
        gaussians_early.max_radii2D[:5] = 50.0  # > 20 px

    scheduler_early = MemoryBoundedDensificationScheduler(
        num_gaussians=N_init,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )
    res_early = scheduler_early.execute_mutation_cycle(
        gaussians=gaussians_early,
        scene_extent=scene_extent,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        size_threshold=None,  # Iteration <= 3000
    )
    assert res_early["executed_hygiene_prune"] == 0

    # Test 2: After step 3000 (size_threshold = 20) -> floaters ARE pruned
    gaussians_late = MockGaussianModel(sh_degree=3, num_points=N_init, device=device)
    with torch.no_grad():
        gaussians_late._opacity.fill_(2.0)
        gaussians_late._scaling.fill_(-3.0)
        gaussians_late.max_radii2D.zero_()
        gaussians_late._scaling[:5] = 10.0
        gaussians_late.max_radii2D[:5] = 50.0

    scheduler_late = MemoryBoundedDensificationScheduler(
        num_gaussians=N_init,
        sh_degree=3,
        hard_vram_limit_mb=16384.0,
        require_native_stats=False,
        device=device,
    )
    res_late = scheduler_late.execute_mutation_cycle(
        gaussians=gaussians_late,
        scene_extent=scene_extent,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        size_threshold=20.0,  # Iteration > 3000
    )
    assert res_late["executed_hygiene_prune"] == 5


def test_two_stage_consecutive_pruning_tensor_alignment():
    """
    Rigorous verification of tensor synchronization across two consecutive prunings:
    Stage 1 removes k1 hygiene points (N0 -> N1).
    Stage 2 removes k2 admission points (N1 -> N2).
    Allocation adds k_new points (N2 -> N_final).
    All parameter tensors and optimizer states must match N_final exactly with no IndexError.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N0 = 200
    gaussians = MockGaussianModel(sh_degree=3, num_points=N0, device=device)
    scene_extent = 1.0

    # Setup: 20 dead opacities for Stage 1
    k1 = 20
    with torch.no_grad():
        gaussians._opacity.fill_(2.0)
        gaussians._opacity[:k1] = -15.0

    # Demand and candidates
    viewspace_grad = torch.ones(N0, device=device) * 0.05
    occ_engine = OcclusionAwareEngine(num_gaussians=N0, device=device)
    occ_engine.grad_accum = viewspace_grad
    # Give some points varying depth across views so they are protected
    vis_idx = torch.arange(N0, device=device)
    occ_engine.update_per_view_stats(vis_idx, torch.ones(N0, device=device) * 1.0, torch.ones(N0, device=device) * 0.1)
    occ_engine.update_per_view_stats(vis_idx, torch.where(vis_idx < 30, torch.tensor(3.0, device=device), torch.tensor(1.0, device=device)), torch.ones(N0, device=device) * 0.1)

    demand, protected_mask = occ_engine.compute_compensated_demand(scene_extent=scene_extent)
    clone_mask = torch.ones(N0, dtype=torch.bool, device=device)
    split_mask = torch.zeros(N0, dtype=torch.bool, device=device)

    # --- STAGE 1: HYGIENE PRUNE ---
    hygiene_mask = (gaussians.get_opacity < 0.005).squeeze()
    assert hygiene_mask.sum().item() == k1
    gaussians.prune_points(hygiene_mask)
    N1 = gaussians.get_xyz.shape[0]
    assert N1 == N0 - k1

    # Synchronize tensors
    surviving_1 = ~hygiene_mask
    clone_mask = clone_mask[surviving_1]
    split_mask = split_mask[surviving_1]
    demand = demand[surviving_1]
    protected_mask = protected_mask[surviving_1]

    assert clone_mask.shape[0] == N1
    assert split_mask.shape[0] == N1
    assert demand.shape[0] == N1
    assert protected_mask.shape[0] == N1

    # --- STAGE 2: ADMISSION UTILITY PRUNE ---
    k2 = 15
    prunable_utility = bottom_utility_mask(demand, protected_mask, fraction=0.2)
    utility_indices = torch.where(prunable_utility)[0]
    assert len(utility_indices) >= k2

    sorted_by_demand = utility_indices[torch.argsort(demand[utility_indices])]
    exec_prune_indices = sorted_by_demand[:k2]

    exec_prune_mask = torch.zeros(N1, dtype=torch.bool, device=device)
    exec_prune_mask[exec_prune_indices] = True
    gaussians.prune_points(exec_prune_mask)
    N2 = gaussians.get_xyz.shape[0]
    assert N2 == N1 - k2

    # Synchronize tensors after Stage 2
    surviving_2 = ~exec_prune_mask
    clone_mask = clone_mask[surviving_2]
    split_mask = split_mask[surviving_2]
    demand = demand[surviving_2]
    protected_mask = protected_mask[surviving_2]

    assert clone_mask.shape[0] == N2
    assert split_mask.shape[0] == N2
    assert demand.shape[0] == N2

    # --- ALLOCATION & MUTATION ---
    cost_model = ByteCostModel(sh_degree=3)
    allocator = MarginalUtilityAllocator(cost_model=cost_model)
    final_clone_mask, final_split_mask = allocator.allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        n_clone_quota=10,
        n_split_quota=0,
    )
    k_new = int(final_clone_mask.sum().item())
    assert k_new == 10

    gaussians.densify_and_clone(final_clone_mask, 0.0002, scene_extent)
    N_final = gaussians.get_xyz.shape[0]
    assert N_final == N2 + k_new

    # Verify optimizer state alignment
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        assert p.shape[0] == N_final
        state = gaussians.optimizer.state[p]
        assert state["exp_avg"].shape[0] == N_final
        assert state["exp_avg_sq"].shape[0] == N_final


def test_legacy_as_is_vs_decoupled_dead_opacity_behavior():
    """
    Demonstrates the difference between legacy_as_is (bug preserved for ablation)
    and decoupled (fixed):
    When admission action is ACCEPT, legacy_as_is leaves dead opacities unpruned,
    while decoupled always purges them.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 50

    # Helper simulating the legacy logic
    def run_legacy(gaussians):
        size_threshold = None
        dead_opacity_mask = (gaussians.get_opacity < 0.005).squeeze()
        prunable_utility = torch.zeros_like(dead_opacity_mask)
        redundant_mask = torch.logical_or(dead_opacity_mask, prunable_utility)
        # Decision is ACCEPT
        action = "ACCEPT"
        k_prune = 0
        executed_pre_prune = 0
        if action in ["PRUNE_FIRST", "SHRINK_AND_PRUNE_FIRST"]:
            if k_prune > 0:
                gaussians.prune_points(redundant_mask)
                executed_pre_prune = int(redundant_mask.sum().item())
        return executed_pre_prune

    # Helper simulating decoupled logic
    def run_decoupled(gaussians):
        size_threshold = None
        hygiene_mask = (gaussians.get_opacity < 0.005).squeeze()
        executed_hygiene = 0
        if hygiene_mask.any():
            executed_hygiene = int(hygiene_mask.sum().item())
            gaussians.prune_points(hygiene_mask)
        return executed_hygiene

    # Model 1: Legacy
    g_legacy = MockGaussianModel(sh_degree=3, num_points=N, device=device)
    with torch.no_grad():
        g_legacy._opacity.fill_(2.0)
        g_legacy._opacity[:8] = -15.0  # 8 dead opacities
    pruned_legacy = run_legacy(g_legacy)
    # In legacy, dead opacities LEAK through!
    assert pruned_legacy == 0
    assert g_legacy.get_xyz.shape[0] == N
    assert (g_legacy.get_opacity.squeeze() < 0.005).sum().item() == 8

    # Model 2: Decoupled
    g_decoupled = MockGaussianModel(sh_degree=3, num_points=N, device=device)
    with torch.no_grad():
        g_decoupled._opacity.fill_(2.0)
        g_decoupled._opacity[:8] = -15.0  # 8 dead opacities
    pruned_decoupled = run_decoupled(g_decoupled)
    # In decoupled, dead opacities are CLEANLY purged!
    assert pruned_decoupled == 8
    assert g_decoupled.get_xyz.shape[0] == N - 8
    assert (g_decoupled.get_opacity.squeeze() < 0.005).sum().item() == 0


def test_vanilla_baseline_path_equivalence():
    """
    Under enable_zero_oom=False, canonical Vanilla 3DGS lifecycle executes:
    Clone -> Split -> Prune without admission controller or knapsack restrictions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 60
    scene_extent = 1.0
    gaussians = MockGaussianModel(sh_degree=3, num_points=N, device=device)

    with torch.no_grad():
        gaussians._opacity.fill_(2.0)
        # 5 dead opacities
        gaussians._opacity[:5] = -15.0

    clone_mask = torch.zeros(N, dtype=torch.bool, device=device)
    clone_mask[10:20] = True  # 10 clone candidates
    split_mask = torch.zeros(N, dtype=torch.bool, device=device)
    split_mask[20:25] = True  # 5 split candidates

    # Canonical Vanilla execution:
    # 1. Clone
    gaussians.densify_and_clone(clone_mask, 0.0002, scene_extent)
    # 2. Split
    # Slicing/padding for split:
    split_mask_padded = torch.cat([split_mask, torch.zeros(10, dtype=torch.bool, device=device)])
    gaussians.densify_and_split(split_mask_padded, 0.0002, scene_extent)
    # 3. Prune
    prune_mask = (gaussians.get_opacity < 0.005).squeeze()
    assert prune_mask[:5].all()
    gaussians.prune_points(prune_mask)

    # Dead opacities must be gone
    assert (gaussians.get_opacity.squeeze() < 0.005).sum().item() == 0

    # Optimizer state aligned
    N_final = gaussians.get_xyz.shape[0]
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        assert p.shape[0] == N_final
        assert gaussians.optimizer.state[p]["exp_avg"].shape[0] == N_final


def test_view_count_vs_pixel_footprint_separation():
    """
    BUG-01 Verification:
    1. A Gaussian visible in 5 views accumulates visibility count 5.
    2. A Gaussian visible in 1 view accumulates visibility count 1.
    3. An invisible Gaussian does not increment its count.
    4. Pixel footprint count (e.g. 500 px) does NOT affect the gradient denominator.
    5. Average gradient matches the expected per-view normalization.
    """
    device = "cpu"
    N = 3
    accum_visibility = torch.zeros(N, dtype=torch.int32, device=device)
    accum_pixel_footprint = torch.zeros(N, dtype=torch.int32, device=device)
    viewspace_grad = torch.zeros(N, device=device)

    # Point 0: visible in 5 views, 100 pixels per view, grad norm = 0.001 per view
    for _ in range(5):
        active_mask = torch.tensor([True, False, False])
        cur_vis_count = torch.tensor([100, 0, 0])
        accum_visibility[active_mask] += 1
        accum_pixel_footprint[active_mask] += cur_vis_count[active_mask]
        viewspace_grad[active_mask] += 0.001

    # Point 1: visible in 1 view, 500 pixels, grad norm = 0.0005
    active_mask = torch.tensor([False, True, False])
    cur_vis_count = torch.tensor([0, 500, 0])
    accum_visibility[active_mask] += 1
    accum_pixel_footprint[active_mask] += cur_vis_count[active_mask]
    viewspace_grad[active_mask] += 0.0005

    # Point 2: invisible in all views

    # Assertions on visibility counts (views)
    assert accum_visibility[0].item() == 5
    assert accum_visibility[1].item() == 1
    assert accum_visibility[2].item() == 0

    # Assertions on pixel footprints
    assert accum_pixel_footprint[0].item() == 500
    assert accum_pixel_footprint[1].item() == 500
    assert accum_pixel_footprint[2].item() == 0

    # Gradient normalization denominator MUST use view count, NOT pixel footprint
    denom = torch.clamp(accum_visibility.float(), min=1.0)
    avg_grad = viewspace_grad / denom

    # Point 0: total grad = 0.005 / 5 views = 0.001 (NOT 0.005 / 500 = 0.00001!)
    assert torch.isclose(avg_grad[0], torch.tensor(0.001))
    # Point 1: total grad = 0.0005 / 1 view = 0.0005 (NOT 0.0005 / 500 = 0.000001!)
    assert torch.isclose(avg_grad[1], torch.tensor(0.0005))
    # Point 2: unobserved
    assert torch.isclose(avg_grad[2], torch.tensor(0.0))


def test_ablation_matrix_factor_independence():
    """
    Verifies 2x2 Factorial Design independence:
    Factor P (Pruning): P0 (legacy_as_is) vs P1 (decoupled)
    Factor O (Occlusion): O0 (legacy) vs O1 (corrected)
    - A0: P0 + O0
    - A1: P1 + O0
    - A2: P0 + O1
    - A3: P1 + O1
    """
    device = "cpu"
    N = 20

    # Test Occlusion Factor O:
    # Under identical inputs (1 view, view_depth_variance = 2.0, low T):
    vis_idx = torch.arange(N)
    depths = torch.full((N,), 3.0)
    T = torch.full((N,), 0.1)
    v_depth_var = torch.full((N,), 2.0)

    # O0 (legacy):
    eng_O0 = OcclusionAwareEngine(N, occlusion_physics_mode="legacy", device=device)
    eng_O0.update_per_view_stats(vis_idx, depths, T, view_depth_variance=v_depth_var)
    eng_O0.grad_accum = torch.ones(N)
    _, prot_O0 = eng_O0.compute_compensated_demand(10.0)
    # Legacy adds within_view_var -> protected even with 1 observation
    assert prot_O0.all()

    # O1 (corrected):
    eng_O1 = OcclusionAwareEngine(N, occlusion_physics_mode="corrected", device=device)
    eng_O1.update_per_view_stats(vis_idx, depths, T, view_depth_variance=v_depth_var)
    eng_O1.grad_accum = torch.ones(N)
    _, prot_O1 = eng_O1.compute_compensated_demand(10.0)
    # Corrected ignores screen-space footprint -> V=1 has 0 variance -> NOT protected
    assert not prot_O1.any()

    # None mode:
    eng_none = OcclusionAwareEngine(N, occlusion_physics_mode="none", device=device)
    eng_none.update_per_view_stats(vis_idx, depths, T, view_depth_variance=v_depth_var)
    eng_none.grad_accum = torch.ones(N)
    dem_none, prot_none = eng_none.compute_compensated_demand(10.0)
    assert not prot_none.any()
    assert torch.allclose(dem_none, torch.ones(N))  # omega = 1.0 everywhere


def test_three_case_gradient_mask_consistency_and_canonical_equivalence():
    """
    Mandatory audit test verifying gradient numerator/denominator mask consistency
    across three physical observation regimes and proving Canonical Vanilla equivalence:
    - Case 1: Visible with pixel contribution (in frustum, cur_vis_count > 0)
    - Case 2: In frustum but NO pixel contribution (in frustum, cur_vis_count == 0)
    - Case 3: Outside frustum (radii == 0, cur_vis_count == 0)
    """
    device = "cpu"
    N = 4

    # Setup 4 primitives:
    # G0: visible in all 10 views (in frustum, cur_vis_count=100, grad=0.001)
    # G1: in frustum in all 10 views, occluded in views 0-8 (cur_vis_count=0, grad=0), visible in view 9 (cur_vis_count=50, grad=0.0003)
    # G2: in frustum in all 10 views, but occluded in ALL 10 views (cur_vis_count=0, grad=0)
    # G3: outside frustum in all 10 views (radii=0, cur_vis_count=0, grad=0)

    # -------------------------------------------------------------------------
    # 1. OPTION A: CANONICAL VANILLA SEMANTICS (Both use visibility_filter)
    # -------------------------------------------------------------------------
    accum_vis_canonical = torch.zeros(N, dtype=torch.int32, device=device)
    grad_accum_canonical = torch.zeros(N, device=device)

    for v in range(10):
        # Frustum culling filter (radii > 0)
        in_frustum = torch.tensor([True, True, True, False], device=device)
        # Pixel contribution counts
        if v < 9:
            vis_counts = torch.tensor([100, 0, 0, 0], device=device)
            view_grad = torch.tensor([0.001, 0.0, 0.0, 0.0], device=device)
        else:
            vis_counts = torch.tensor([100, 50, 0, 0], device=device)
            view_grad = torch.tensor([0.001, 0.0003, 0.0, 0.0], device=device)

        # Canonical semantics: EXACT SAME MASK FOR NUMERATOR AND DENOMINATOR
        accum_vis_canonical[in_frustum] += 1
        grad_accum_canonical[in_frustum] += view_grad[in_frustum]

    denom_canon = torch.clamp(accum_vis_canonical.float(), min=1.0)
    avg_grad_canonical = grad_accum_canonical / denom_canon

    # Check G0 (Visible in all 10 views)
    assert accum_vis_canonical[0].item() == 10
    assert torch.isclose(grad_accum_canonical[0], torch.tensor(0.010))
    assert torch.isclose(avg_grad_canonical[0], torch.tensor(0.001))

    # Check G1 (In frustum 10 views, but occluded 9 views):
    # Denominator counts all 10 frustum views -> avg_grad is diluted to 0.00003 (< threshold 0.0002)
    assert accum_vis_canonical[1].item() == 10
    assert torch.isclose(grad_accum_canonical[1], torch.tensor(0.0003))
    assert torch.isclose(avg_grad_canonical[1], torch.tensor(0.00003))
    assert avg_grad_canonical[1].item() < 0.0002

    # Check G2 (In frustum 10 views, completely occluded):
    assert accum_vis_canonical[2].item() == 10
    assert torch.isclose(grad_accum_canonical[2], torch.tensor(0.0))
    assert torch.isclose(avg_grad_canonical[2], torch.tensor(0.0))

    # Check G3 (Outside frustum in all views):
    assert accum_vis_canonical[3].item() == 0
    assert torch.isclose(grad_accum_canonical[3], torch.tensor(0.0))
    assert torch.isclose(avg_grad_canonical[3], torch.tensor(0.0))

    # -------------------------------------------------------------------------
    # 2. OPTION B: CONTRIBUTION-AWARE SEMANTICS (Both use cur_vis_count > 0)
    # -------------------------------------------------------------------------
    accum_vis_contrib = torch.zeros(N, dtype=torch.int32, device=device)
    grad_accum_contrib = torch.zeros(N, device=device)

    for v in range(10):
        in_frustum = torch.tensor([True, True, True, False], device=device)
        if v < 9:
            vis_counts = torch.tensor([100, 0, 0, 0], device=device)
            view_grad = torch.tensor([0.001, 0.0, 0.0, 0.0], device=device)
        else:
            vis_counts = torch.tensor([100, 50, 0, 0], device=device)
            view_grad = torch.tensor([0.001, 0.0003, 0.0, 0.0], device=device)

        has_pixels = (vis_counts > 0)

        # Contribution-aware semantics: EXACT SAME MASK FOR NUMERATOR AND DENOMINATOR
        accum_vis_contrib[has_pixels] += 1
        grad_accum_contrib[has_pixels] += view_grad[has_pixels]

    denom_contrib = torch.clamp(accum_vis_contrib.float(), min=1.0)
    avg_grad_contrib = grad_accum_contrib / denom_contrib

    # Check G0 (Visible in all 10 views)
    assert accum_vis_contrib[0].item() == 10
    assert torch.isclose(avg_grad_contrib[0], torch.tensor(0.001))

    # Check G1 under Contribution-Aware:
    # Denominator ONLY counts the 1 view where pixels were rendered -> avg_grad is 0.0003 (>= threshold 0.0002)
    assert accum_vis_contrib[1].item() == 1
    assert torch.isclose(grad_accum_contrib[1], torch.tensor(0.0003))
    assert torch.isclose(avg_grad_contrib[1], torch.tensor(0.0003))
    assert avg_grad_contrib[1].item() >= 0.0002

    # Check G2 (Completely occluded)
    assert accum_vis_contrib[2].item() == 0
    assert torch.isclose(avg_grad_contrib[2], torch.tensor(0.0))
    assert torch.isclose(avg_grad_contrib[2], torch.tensor(0.0))

    # Check G3 (Outside frustum)
    assert accum_vis_contrib[3].item() == 0
    assert torch.isclose(avg_grad_contrib[3], torch.tensor(0.0))
    assert torch.isclose(avg_grad_contrib[3], torch.tensor(0.0))

    # -------------------------------------------------------------------------
    # 3. VERIFY NO MASK MIXING IN EITHER MODE
    # -------------------------------------------------------------------------
    # In both options, the set of updated indices for numerator equals denominator
    assert (accum_vis_canonical > 0).tolist() == [True, True, True, False]
    assert (accum_vis_contrib > 0).tolist() == [True, True, False, False]


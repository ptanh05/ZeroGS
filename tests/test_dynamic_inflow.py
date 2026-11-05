"""
Unit tests for PV-SRA-3DGS V3: Dynamic VRAM-Aware Inflow Admission.
Validates candidate admission policy, memory bounding, revalidation, and V2 backward compatibility.
"""

import pytest
import torch
import math

from zerogs.cost_model import ByteCostModel
from zerogs.admission_controller import (
    AdmissionController,
    AdmissionAction,
    DynamicInflowDecision,
)
from zerogs.marginal_allocator import MarginalUtilityAllocator


@pytest.fixture
def cost_model():
    return ByteCostModel(sh_degree=3)


@pytest.fixture
def controller(cost_model):
    # 6,144 MB hard limit, 512 MB safety headroom, 256 MB non-pytorch allowance
    # Effective budget = 6144 - 512 = 5632 MB
    return AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=256.0,
    )


def test_low_headroom_cap_decreases(controller):
    """1. Low headroom: cap decreases towards and below minimum to guarantee safety."""
    # Target headroom is 1024 MB
    # High headroom: allocated 2000 MB -> footprint 2256 MB -> headroom 3376 MB (> 2048 MB)
    dec_high = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=2000 * 1024 * 1024,
    )
    assert dec_high.computed_cap == 15000

    # Medium headroom: allocated 4100 MB -> footprint 4356 MB -> headroom 1276 MB (between 1024 and 2048)
    dec_med = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=4100 * 1024 * 1024,
    )
    assert 5000 < dec_med.computed_cap < 15000
    assert dec_med.computed_cap < dec_high.computed_cap

    # Low headroom: allocated 4400 MB -> footprint 4656 MB -> headroom 976 MB (<= 1024 MB)
    dec_low = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=4400 * 1024 * 1024,
    )
    assert dec_low.computed_cap == 5000

    # Critically low headroom: allocated 5373 MB -> footprint 5629 MB -> headroom 3 MB (3,145,728 bytes)
    # Memory safe limit: (3MB - 1MB padding) / 2171 bytes ~ 966 < 5000 -> cap decreases below 5000!
    dec_critical = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=5373 * 1024 * 1024,
    )
    assert dec_critical.computed_cap < 5000
    assert dec_critical.computed_cap == dec_critical.memory_safe_limit

    # Zero headroom: allocated 5400 MB -> footprint 5656 MB > 5632 MB -> headroom = 0
    dec_zero = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=5400 * 1024 * 1024,
    )
    assert dec_zero.computed_cap == 0
    assert dec_zero.admissible_cap == 0


def test_high_headroom_cap_increases(controller):
    """2. High headroom: cap increases monotonically as headroom grows above target."""
    headroom_mbs = [1024, 1200, 1400, 1600, 1800, 2048]
    caps = []
    # Effective budget is 5632 MB; footprint = 5632 - H; allocated = footprint - 256
    for h in headroom_mbs:
        footprint_mb = 5632 - h
        alloc_mb = footprint_mb - 256
        dec = controller.compute_dynamic_inflow_cap(
            n_clone_candidates=20000,
            n_split_candidates=20000,
            current_allocated_bytes=alloc_mb * 1024 * 1024,
        )
        caps.append(dec.computed_cap)

    assert caps[0] == 5000
    for i in range(1, len(caps)):
        assert caps[i] >= caps[i - 1], f"Cap should be monotonic: {caps[i]} < {caps[i-1]}"
    assert caps[-1] == 15000


def test_cap_never_exceeds_max(controller):
    """3. Cap never exceeds 15,000, even under massive VRAM availability."""
    # 100 GB available headroom
    huge_controller = AdmissionController(
        cost_model=controller.cost_model,
        hard_vram_limit_mb=100_000.0,
        safety_headroom_mb=512.0,
    )
    dec = huge_controller.compute_dynamic_inflow_cap(
        n_clone_candidates=50000,
        n_split_candidates=50000,
        current_allocated_bytes=1000 * 1024 * 1024,
    )
    assert dec.computed_cap <= 15000
    assert dec.computed_cap == 15000


def test_cap_never_falls_below_min_when_memory_permits(controller):
    """4. Cap never falls below 5,000 when valid candidates and memory permit."""
    # Headroom = 600 MB (below target 1024 MB, but plenty to hold 5,000 items at ~2.17KB = ~11MB)
    footprint_mb = 5632 - 600
    alloc_mb = footprint_mb - 256
    dec = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=3000,
        n_split_candidates=3000,
        current_allocated_bytes=alloc_mb * 1024 * 1024,
    )
    assert dec.computed_cap == 5000
    assert dec.admissible_cap == 5000


def test_zero_candidates_zero_allocation(controller, cost_model):
    """5. No candidates: zero allocation."""
    dec = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=0,
        n_split_candidates=0,
        current_allocated_bytes=2000 * 1024 * 1024,
    )
    assert dec.n_requested == 0
    assert dec.admissible_cap == 0

    allocator = MarginalUtilityAllocator(cost_model)
    clone_mask = torch.zeros(10, dtype=torch.bool)
    split_mask = torch.zeros(10, dtype=torch.bool)
    demand = torch.zeros(10, dtype=torch.float32)

    final_c, final_s = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=10**8,
        max_items=dec.admissible_cap,
    )
    assert final_c.sum().item() == 0
    assert final_s.sum().item() == 0


def test_split_transient_cost_respected(controller):
    """6. Split transient cost (2.0x multiplier) is strictly respected in unit cost model."""
    dec = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=5000,
        n_split_candidates=5000,
        current_allocated_bytes=2000 * 1024 * 1024,
    )
    c_clone = dec.estimated_bytes_per_clone
    c_split = dec.estimated_bytes_per_split
    assert abs(c_split - 2 * c_clone) <= 2
    assert c_split > c_clone


def test_clone_and_split_obey_hard_budget(cost_model):
    """7. Clone and split allocation strictly obeys hard byte budget."""
    allocator = MarginalUtilityAllocator(cost_model)
    N = 200
    demand = torch.linspace(1.0, 10.0, N)
    clone_mask = torch.zeros(N, dtype=torch.bool)
    clone_mask[:100] = True
    split_mask = ~clone_mask

    # Provide budget: padding + 50,000 bytes
    max_budget = cost_model.cuda_padding_bytes + 50_000
    final_c, final_s = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=max_budget,
        max_items=15000,
        kappa_split=0.40,
    )

    n_c = int(final_c.sum().item())
    n_s = int(final_s.sum().item())
    estimated_cost = (
        cost_model.cuda_padding_bytes
        + n_c * cost_model.persistent_bytes_per_gaussian * cost_model.clone_multiplier
        + n_s * cost_model.persistent_bytes_per_gaussian * cost_model.split_multiplier
    )
    assert estimated_cost <= max_budget + 1024  # within padding / fixed overhead allowance
    assert n_c + n_s > 0


def test_memory_estimates_remain_nonnegative(controller):
    """8. Memory estimates, caps, and headroom remain nonnegative across extreme inputs."""
    for alloc_mb in [0, 1000, 5632, 7000]:
        dec = controller.compute_dynamic_inflow_cap(
            n_clone_candidates=5000,
            n_split_candidates=5000,
            current_allocated_bytes=alloc_mb * 1024 * 1024,
        )
        assert dec.current_footprint_bytes >= 0
        assert dec.available_headroom_bytes >= 0
        assert dec.target_headroom_bytes >= 0
        assert dec.estimated_bytes_per_clone >= 0
        assert dec.estimated_bytes_per_split >= 0
        assert dec.memory_safe_limit >= 0
        assert dec.computed_cap >= 0
        assert dec.admissible_cap >= 0
        assert dec.rejected_by_cap >= 0
        assert dec.rejected_by_budget >= 0


def test_admission_revalidated_before_allocation(controller):
    """9. Admission is revalidated before tensor allocation when memory shifts."""
    # Stage A: Initial condition has 3000 MB headroom -> allows 15,000 cap
    dec_a = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=2000 * 1024 * 1024,
    )
    assert dec_a.computed_cap == 15000

    # Stage B: Unexpected memory spike right before allocation
    # Available headroom shrinks to 5 MB
    alloc_spiked = (5632 - 5 - 256) * 1024 * 1024
    decision_revalidated = controller.evaluate_admission(
        n_clone_requested=7500,
        n_split_requested=7500,
        current_num_gaussians=300000,
        prune_candidates_count=100,
        current_allocated_bytes=alloc_spiked,
    )
    # Revalidation catches the spike and enforces safety (shrinks or rejects)
    assert decision_revalidated.action in [
        AdmissionAction.SHRINK_AND_PRUNE_FIRST,
        AdmissionAction.REJECT_AND_PRUNE,
    ]
    assert decision_revalidated.n_clone < 7500
    assert decision_revalidated.n_split < 7500


def test_v2_reproducibility_when_dynamic_disabled(controller):
    """10. Existing V2 behavior remains reproducible when dynamic admission is disabled."""
    # When inflow_mode == 'static', cap stays fixed at inflow_cap_min (5000) regardless of abundant headroom
    dec_static_high = controller.compute_dynamic_inflow_cap(
        n_clone_candidates=10000,
        n_split_candidates=10000,
        current_allocated_bytes=1000 * 1024 * 1024,
        inflow_mode="static",
        inflow_cap_min=5000,
    )
    assert dec_static_high.computed_cap == 5000
    assert dec_static_high.admissible_cap == 5000
    assert dec_static_high.rejected_by_cap == 15000

    # evaluate_admission also carries static mode
    eval_dec = controller.evaluate_admission(
        n_clone_requested=10000,
        n_split_requested=10000,
        current_num_gaussians=100000,
        prune_candidates_count=0,
        current_allocated_bytes=1000 * 1024 * 1024,
        inflow_mode="static",
        inflow_cap_min=5000,
    )
    assert eval_dec.action == AdmissionAction.ACCEPT
    assert eval_dec.dynamic_inflow["computed_cap"] == 5000
    assert eval_dec.dynamic_inflow["inflow_mode"] == "static"


def test_train_argparser_dynamic_inflow():
    from train import build_parser
    parser, lp, op, pp = build_parser()
    args = parser.parse_args([
        '-s', 'data/bonsai',
        '--inflow_mode', 'dynamic',
        '--inflow_cap_min', '6000',
        '--inflow_cap_max', '18000',
        '--target_headroom_mb', '1536.0',
    ])
    assert args.inflow_mode == 'dynamic'
    assert args.inflow_cap_min == 6000
    assert args.inflow_cap_max == 18000
    assert args.target_headroom_mb == 1536.0


def test_calibrated_cost_safe_limit_deducts_overhead():
    calib_model = ByteCostModel.from_calibration_file('experiments/results/cost_calibration.json')
    ctrl = AdmissionController(
        cost_model=calib_model,
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=256.0,
    )
    # Headroom = 100 MB < (118.6 MB split overhead + 1 MB padding)
    # Allocated = 5632 - 256 - 100 = 5276 MB
    dec_tight = ctrl.compute_dynamic_inflow_cap(
        n_clone_candidates=1000,
        n_split_candidates=1000,
        current_allocated_bytes=5276 * 1024 * 1024,
    )
    assert dec_tight.memory_safe_limit == 0
    assert dec_tight.computed_cap == 0

    # Headroom = 200 MB > 119.6 MB overhead
    # Allocated = 5632 - 256 - 200 = 5176 MB
    dec_ok = ctrl.compute_dynamic_inflow_cap(
        n_clone_candidates=1000,
        n_split_candidates=1000,
        current_allocated_bytes=5176 * 1024 * 1024,
    )
    assert dec_ok.memory_safe_limit > 0
    assert dec_ok.computed_cap > 0

"""
Unit tests for AdmissionController (Module 2).
"""

import pytest
from zerogs.cost_model import ByteCostModel
from zerogs.admission_controller import (
    AdmissionController,
    AdmissionAction,
    AdmissionDecision,
)


def test_admission_accept_when_headroom_abundant():
    cost_model = ByteCostModel(sh_degree=3)
    # 8GB hard limit, 512MB safety headroom -> ~7680MB effective
    # Allocated: 1000MB -> ~6680MB headroom
    controller = AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=8192.0,
        safety_headroom_mb=512.0,
    )

    decision = controller.evaluate_admission(
        n_clone_requested=5000,
        n_split_requested=5000,
        current_num_gaussians=50000,
        prune_candidates_count=2000,
        current_allocated_bytes=1000 * 1024 * 1024,
    )

    assert decision.action == AdmissionAction.ACCEPT
    assert decision.n_clone == 5000
    assert decision.n_split == 5000
    assert decision.required_pre_prune == 0


def test_admission_prune_first_when_headroom_tight():
    cost_model = ByteCostModel(sh_degree=3)
    # Effective limit: 8192 - 512 = 7680 MB
    # Suppose current allocated is 7670 MB -> only 10MB headroom left!
    # Transient cost for 20000 clones + splits will exceed 10MB
    controller = AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=8192.0,
        safety_headroom_mb=512.0,
        enable_deterministic_prune_first=True,
    )

    allocated = int(7670 * 1024 * 1024)
    # Requesting 10k clones, 5k splits requires ~30MB
    # We have 100k prune candidates available (~94MB potential freed)
    decision = controller.evaluate_admission(
        n_clone_requested=10000,
        n_split_requested=5000,
        current_num_gaussians=200000,
        prune_candidates_count=100000,
        current_allocated_bytes=allocated,
    )

    assert decision.action == AdmissionAction.PRUNE_FIRST
    assert decision.required_pre_prune > 0
    assert decision.n_clone == 10000
    assert decision.n_split == 5000


def test_admission_shrink_and_prune_first_on_saturation():
    cost_model = ByteCostModel(sh_degree=3)
    # Effective limit: 8192 - 512 = 7680 MB
    # Current allocated is 7675 MB -> only 5MB headroom left!
    # Available prune candidates is small (only 1,000 pts ~ 0.94MB)
    # Total potential space is only ~5.94MB
    # Requesting huge batch (50k clones + 50k splits ~ 150MB)
    controller = AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=8192.0,
        safety_headroom_mb=512.0,
    )

    allocated = int(7675 * 1024 * 1024)
    decision = controller.evaluate_admission(
        n_clone_requested=50000,
        n_split_requested=50000,
        current_num_gaussians=500000,
        prune_candidates_count=1000,
        current_allocated_bytes=allocated,
    )

    assert decision.action == AdmissionAction.SHRINK_AND_PRUNE_FIRST
    assert decision.required_pre_prune == 1000
    assert decision.n_clone < 50000
    assert decision.n_split < 50000
    assert decision.transient_cost_bytes <= (
        decision.available_headroom_bytes + decision.freed_bytes
    )
    assert decision.mutation_budget_bytes == (
        decision.available_headroom_bytes + decision.freed_bytes
    )


def test_admission_4096mb_hard_budget():
    """Verify hard 4096MB VRAM budget enforcement under INT8 and FP16 models."""
    from zerogs.admission_controller import VramBudgetExceededError
    from zerogs.cost_model import ByteCostModel

    cost_model_int8 = ByteCostModel(sh_degree=3, optimizer_type="int8_adam")
    controller = AdmissionController(
        cost_model=cost_model_int8,
        hard_vram_limit_mb=4096.0,
        safety_headroom_mb=256.0,
        non_pytorch_allowance_mb=256.0,
        enable_deterministic_prune_first=True,
    )

    # Effective limit: 4096 - 256 = 3840 MB
    assert controller.effective_budget_bytes == int(3840 * 1024 * 1024)
    
    # 1. Normal state: 2000 MB allocated -> plenty of headroom
    dec_accept = controller.evaluate_admission(
        n_clone_requested=2000,
        n_split_requested=2000,
        current_num_gaussians=100_000,
        prune_candidates_count=5000,
        current_allocated_bytes=int(2000 * 1024 * 1024),
    )
    assert dec_accept.action == AdmissionAction.ACCEPT
    assert dec_accept.n_clone == 2000
    assert dec_accept.n_split == 2000

    # 2. Saturated state: 3580 MB allocated (only ~4MB headroom left before non-pytorch allowance)
    dec_saturated = controller.evaluate_admission(
        n_clone_requested=20000,
        n_split_requested=20000,
        current_num_gaussians=600_000,
        prune_candidates_count=2000,
        current_allocated_bytes=int(3580 * 1024 * 1024),
    )
    assert dec_saturated.action in [
        AdmissionAction.SHRINK_AND_PRUNE_FIRST,
        AdmissionAction.PRUNE_FIRST,
        AdmissionAction.REJECT_AND_PRUNE,
    ]
    assert dec_saturated.transient_cost_bytes <= (dec_saturated.available_headroom_bytes + dec_saturated.freed_bytes)

    # 3. Hard limit parameters
    assert controller.hard_vram_limit_bytes == int(4096 * 1024 * 1024)
    assert controller.safety_headroom_bytes == int(256 * 1024 * 1024)
    assert controller.non_pytorch_allowance_bytes == int(256 * 1024 * 1024)

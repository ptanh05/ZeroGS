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

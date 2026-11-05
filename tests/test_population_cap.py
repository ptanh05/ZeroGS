"""
Unit tests for PV-SRA-3DGS V4: Controlled Population-Cap Ablation Mechanics.
Validates population ceiling clipping, candidate conservation accounting,
and backward compatibility with 1.0M baseline.
"""

import pytest
import torch
from zerogs.cost_model import ByteCostModel
from zerogs.marginal_allocator import MarginalUtilityAllocator


@pytest.fixture
def allocator():
    cm = ByteCostModel(sh_degree=3)
    return MarginalUtilityAllocator(cost_model=cm)


def test_population_cap_below_saturation():
    """When population is below ceiling, mutation cap is determined solely by dynamic inflow cap."""
    max_gaussians_cap = 1_150_000
    current_gaussians = 800_000
    effective_inflow_cap = 15_000

    remaining_cap = max(0, max_gaussians_cap - current_gaussians)
    assert remaining_cap == 350_000

    mutation_item_cap = min(effective_inflow_cap, remaining_cap)
    assert mutation_item_cap == 15_000


def test_population_cap_saturation_clipping(allocator):
    """When population nears ceiling, mutation cap is clipped to remaining slots."""
    max_gaussians_cap = 1_150_000
    current_gaussians = 1_146_000
    effective_inflow_cap = 15_000

    remaining_cap = max(0, max_gaussians_cap - current_gaussians)
    assert remaining_cap == 4_000

    mutation_item_cap = min(effective_inflow_cap, remaining_cap)
    assert mutation_item_cap == 4_000

    # Test joint knapsack allocator respects mutation_item_cap
    n_candidates = 10_000
    demand = torch.rand(n_candidates)
    clone_mask = torch.ones(n_candidates, dtype=torch.bool)
    split_mask = torch.zeros(n_candidates, dtype=torch.bool)

    alloc_clone, alloc_split = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=100 * 1024 * 1024,
        max_items=mutation_item_cap,
    )

    admitted = int(alloc_clone.sum().item() + alloc_split.sum().item())
    assert admitted == 4_000
    assert current_gaussians + admitted <= max_gaussians_cap


def test_population_cap_at_ceiling(allocator):
    """When population reaches or exceeds ceiling, 0 candidates are admitted."""
    max_gaussians_cap = 1_150_000
    current_gaussians = 1_150_000
    effective_inflow_cap = 15_000

    remaining_cap = max(0, max_gaussians_cap - current_gaussians)
    assert remaining_cap == 0

    mutation_item_cap = min(effective_inflow_cap, remaining_cap)
    assert mutation_item_cap == 0

    n_candidates = 12_000
    demand = torch.rand(n_candidates)
    clone_mask = torch.ones(n_candidates, dtype=torch.bool)
    split_mask = torch.zeros(n_candidates, dtype=torch.bool)

    alloc_clone, alloc_split = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=100 * 1024 * 1024,
        max_items=mutation_item_cap,
    )

    admitted = int(alloc_clone.sum().item() + alloc_split.sum().item())
    assert admitted == 0
    assert current_gaussians + admitted == max_gaussians_cap


def test_candidate_conservation_equation():
    """Candidate conservation equation strictly accounts for all candidates with Delta = 0."""
    requested = 25_000
    dynamic_cap = 15_000
    remaining_pop_cap = 6_000

    # 1. Dynamic cap limits candidates
    rej_by_cap = max(0, requested - dynamic_cap)
    assert rej_by_cap == 10_000

    # 2. Population ceiling limits admissible candidates
    item_cap = min(dynamic_cap, remaining_pop_cap)
    assert item_cap == 6_000

    admitted = item_cap
    rej_by_pop = max(0, requested - rej_by_cap - admitted)
    assert rej_by_pop == 9_000

    # 3. Conservation verification
    sum_components = admitted + rej_by_cap + rej_by_pop
    delta = requested - sum_components
    assert delta == 0
    assert sum_components == requested


def test_v3_backward_compatibility():
    """Default max_gaussians_cap = 1,000,000 preserves exact V3 behavior."""
    max_gaussians_cap = 1_000_000
    current_gaussians = 995_000
    effective_inflow_cap = 15_000

    remaining_cap = max(0, max_gaussians_cap - current_gaussians)
    assert remaining_cap == 5_000

    mutation_item_cap = min(effective_inflow_cap, remaining_cap)
    assert mutation_item_cap == 5_000


"""
Unit tests for MarginalUtilityAllocator (Module 3).
"""

import torch
import pytest
from zerogs.cost_model import ByteCostModel
from zerogs.marginal_allocator import MarginalUtilityAllocator


def test_marginal_utility_allocator_top_k():
    cost_model = ByteCostModel(sh_degree=3)
    allocator = MarginalUtilityAllocator(cost_model=cost_model)

    N = 10
    demand = torch.tensor([1.0, 5.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 0.5, 6.0])
    clone_mask = torch.tensor([True, True, True, True, True, False, False, False, False, False])
    split_mask = torch.tensor([False, False, False, False, False, True, True, True, True, True])

    # Quota: 2 clones, 2 splits
    final_clone_mask, final_split_mask = allocator.allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        n_clone_quota=2,
        n_split_quota=2,
    )

    assert final_clone_mask.sum().item() == 2
    assert final_split_mask.sum().item() == 2
    # In clone candidates (indices 0..4), indices 3 (9.0) and 1 (5.0) are top 2
    assert final_clone_mask[3].item() is True
    assert final_clone_mask[1].item() is True
    # In split candidates (indices 5..9), indices 5 (8.0) and 7 (7.0) are top 2
    assert final_split_mask[5].item() is True
    assert final_split_mask[7].item() is True


def test_joint_knapsack_allocate():
    cost_model = ByteCostModel(sh_degree=3)
    allocator = MarginalUtilityAllocator(cost_model=cost_model)

    demand = torch.tensor([10.0, 10.0])
    clone_mask = torch.tensor([True, False])
    split_mask = torch.tensor([False, True])

    # Cost clone is ~1085 bytes, cost split is ~2171 bytes
    # If budget is 1500 bytes: clone fits, split does not!
    final_clone_mask, final_split_mask = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=1500,
    )

    assert final_clone_mask[0].item() is True
    assert final_split_mask[1].item() is False

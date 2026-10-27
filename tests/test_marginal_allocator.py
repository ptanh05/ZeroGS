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
    # After the shared CUDA padding, 1500 bytes remain: clone fits, split does not.
    final_clone_mask, final_split_mask = allocator.joint_knapsack_allocate(
        demand=demand,
        clone_mask=clone_mask,
        split_mask=split_mask,
        max_transient_bytes=cost_model.cuda_padding_bytes + 1500,
    )

    assert final_clone_mask[0].item() is True
    assert final_split_mask[1].item() is False


def test_joint_allocator_accounts_for_fixed_overhead():
    model = ByteCostModel(
        sh_degree=3,
        cuda_padding_bytes=100,
        clone_fixed_overhead_bytes=200,
        split_fixed_overhead_bytes=400,
    )
    allocator = MarginalUtilityAllocator(model)
    demand = torch.tensor([10.0, 9.0])
    clone_mask = torch.tensor([True, False])
    split_mask = torch.tensor([False, True])
    clone_unit = int(model.persistent_bytes_per_gaussian * model.clone_multiplier)

    selected_clone, selected_split = allocator.joint_knapsack_allocate(
        demand,
        clone_mask,
        split_mask,
        max_transient_bytes=100 + 200 + clone_unit,
    )
    assert selected_clone.tolist() == [True, False]
    assert selected_split.tolist() == [False, False]


def test_joint_allocator_ablation_strategies():
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=0)
    allocator = MarginalUtilityAllocator(model)
    demand = torch.tensor([10.0, 9.0, 8.0, 7.0])
    clone_mask = torch.tensor([True, True, False, False])
    split_mask = ~clone_mask
    budget = int(model.persistent_bytes_per_gaussian * model.clone_multiplier * 2)

    demand_clone, demand_split = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, budget, strategy="demand_only"
    )
    assert demand_clone.sum().item() == 2
    assert demand_split.sum().item() == 0

    generator = torch.Generator().manual_seed(123)
    random_a = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, budget, strategy="random", generator=generator
    )
    generator = torch.Generator().manual_seed(123)
    random_b = allocator.joint_knapsack_allocate(
        demand, clone_mask, split_mask, budget, strategy="random", generator=generator
    )
    assert torch.equal(random_a[0], random_b[0])
    assert torch.equal(random_a[1], random_b[1])


def test_joint_allocator_respects_item_cap():
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=0)
    allocator = MarginalUtilityAllocator(model)
    demand = torch.arange(10, dtype=torch.float32)
    clone_mask = torch.ones(10, dtype=torch.bool)
    split_mask = torch.zeros(10, dtype=torch.bool)
    selected_clone, selected_split = allocator.joint_knapsack_allocate(
        demand,
        clone_mask,
        split_mask,
        max_transient_bytes=10**9,
        max_items=3,
    )
    assert selected_clone.sum().item() == 3
    assert selected_split.sum().item() == 0

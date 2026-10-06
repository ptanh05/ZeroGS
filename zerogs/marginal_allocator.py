"""
Marginal Utility Allocation (Module 3 - Solving Gap P3).

Reformulates primitive growth from count-based quotas to byte-level marginal utility:
    lambda_i = Delta Gain_i / Delta Bytes_i

Prioritizes resource allocation to Gaussians that deliver the greatest loss reduction
per incremental Megabyte, ensuring optimal quality under tight VRAM constraints.
"""

from typing import Tuple
import torch

from .cost_model import ByteCostModel


class MarginalUtilityAllocator:
    """
    Allocates densification quotas using marginal gain per incremental byte:
    lambda = Gain / Byte
    """

    def __init__(self, cost_model: ByteCostModel):
        self.cost_model = cost_model

    def allocate(
        self,
        demand: torch.Tensor,
        clone_mask: torch.Tensor,
        split_mask: torch.Tensor,
        n_clone_quota: int,
        n_split_quota: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top-performing candidates according to marginal gain / byte.

        Args:
            demand: 1D Tensor of compensated demand values.
            clone_mask: Boolean mask of candidates qualifying for clone.
            split_mask: Boolean mask of candidates qualifying for split.
            n_clone_quota: Admitted number of clones.
            n_split_quota: Admitted number of splits.

        Returns:
            final_clone_mask: Filtered boolean mask for clones.
            final_split_mask: Filtered boolean mask for splits.
        """
        device = demand.device
        final_clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool, device=device)
        final_split_mask = torch.zeros_like(split_mask, dtype=torch.bool, device=device)

        clone_indices = torch.where(clone_mask)[0]
        split_indices = torch.where(split_mask)[0]

        # Transient byte cost per primitive type
        cost_clone = (
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.clone_multiplier
        )
        cost_split = (
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.split_multiplier
        )

        # 1. Process Clone Candidates
        if len(clone_indices) > 0 and n_clone_quota > 0:
            if len(clone_indices) <= n_clone_quota:
                final_clone_mask[clone_indices] = True
            else:
                clone_gains = demand[clone_indices] / max(1.0, cost_clone)
                top_k = min(n_clone_quota, len(clone_indices))
                top_order = torch.argsort(clone_gains, descending=True)[:top_k]
                selected_clone_idx = clone_indices[top_order]
                final_clone_mask[selected_clone_idx] = True

        # 2. Process Split Candidates
        if len(split_indices) > 0 and n_split_quota > 0:
            if len(split_indices) <= n_split_quota:
                final_split_mask[split_indices] = True
            else:
                split_gains = demand[split_indices] / max(1.0, cost_split)
                top_k = min(n_split_quota, len(split_indices))
                top_order = torch.argsort(split_gains, descending=True)[:top_k]
                selected_split_idx = split_indices[top_order]
                final_split_mask[selected_split_idx] = True

        return final_clone_mask, final_split_mask

    def joint_knapsack_allocate(
        self,
        demand: torch.Tensor,
        clone_mask: torch.Tensor,
        split_mask: torch.Tensor,
        max_transient_bytes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Jointly allocates across both Clone and Split pools using greedy Knapsack
        ordered by lambda = Gain / Byte under a combined byte ceiling.
        """
        device = demand.device
        clone_indices = torch.where(clone_mask)[0]
        split_indices = torch.where(split_mask)[0]

        cost_clone = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.clone_multiplier
        )
        cost_split = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.split_multiplier
        )

        # Compute marginal utilities
        clone_lambdas = demand[clone_indices] / max(1.0, float(cost_clone))
        split_lambdas = demand[split_indices] / max(1.0, float(cost_split))

        # Bundle items: (lambda, item_type, index, cost)
        # item_type: 0 for clone, 1 for split
        items = []
        for i, idx in enumerate(clone_indices):
            items.append((clone_lambdas[i].item(), 0, idx.item(), cost_clone))
        for j, idx in enumerate(split_indices):
            items.append((split_lambdas[j].item(), 1, idx.item(), cost_split))

        # Sort descending by lambda
        items.sort(key=lambda x: x[0], reverse=True)

        final_clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool, device=device)
        final_split_mask = torch.zeros_like(split_mask, dtype=torch.bool, device=device)

        remaining_bytes = max_transient_bytes
        for lam, itype, idx, cost in items:
            if cost <= remaining_bytes:
                remaining_bytes -= cost
                if itype == 0:
                    final_clone_mask[idx] = True
                else:
                    final_split_mask[idx] = True
            if remaining_bytes <= min(cost_clone, cost_split):
                break

        return final_clone_mask, final_split_mask

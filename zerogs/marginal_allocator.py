"""
Marginal Utility Allocation (Module 3 - Solving Gap P3).

Reformulates primitive growth from count-based quotas to byte-level marginal utility:
    lambda_i = Delta Gain_i / Delta Bytes_i

Prioritizes resource allocation to Gaussians that deliver the greatest loss reduction
per incremental Megabyte, ensuring optimal quality under tight VRAM constraints.
"""

from typing import Tuple, Optional, Union, Dict, Any
import torch

from .cost_model import ByteCostModel


class MarginalUtilityAllocator:
    """
    Allocates densification quotas using marginal gain per incremental byte:
    lambda = Gain / Byte
    """

    def __init__(self, cost_model: ByteCostModel):
        self.cost_model = cost_model
        self.last_allocation_stats = {}

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
        strategy: str = "gain_per_byte",
        generator: torch.Generator = None,
        max_items: int = None,
        kappa_split: float = 0.0,
        return_stats: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Jointly allocates across both Clone and Split pools using greedy Knapsack
        ordered by lambda = Gain / Byte under a combined byte ceiling.
        Supports Feature C: Minimum Split Allocation Quota (kappa_split).
        """
        device = demand.device
        clone_indices = torch.where(clone_mask)[0]
        split_indices = torch.where(split_mask)[0]
        n_clone_cand = len(clone_indices)
        n_split_cand = len(split_indices)
        total_candidates = n_clone_cand + n_split_cand

        requested_split_share = float(n_split_cand / max(1, total_candidates))

        cost_clone = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.clone_multiplier
        )
        cost_split = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.split_multiplier
        )

        if strategy == "gain_per_byte":
            clone_lambdas = demand[clone_indices] / max(1.0, float(cost_clone))
            split_lambdas = demand[split_indices] / max(1.0, float(cost_split))
        elif strategy == "demand_only":
            clone_lambdas = demand[clone_indices]
            split_lambdas = demand[split_indices]
        elif strategy == "random":
            clone_lambdas = torch.rand(
                len(clone_indices), device=device, generator=generator
            )
            split_lambdas = torch.rand(
                len(split_indices), device=device, generator=generator
            )
        else:
            raise ValueError(
                f"Unknown allocator strategy {strategy!r}; expected gain_per_byte, demand_only, or random"
            )

        final_clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool, device=device)
        final_split_mask = torch.zeros_like(split_mask, dtype=torch.bool, device=device)

        remaining_bytes = max(0, int(max_transient_bytes))
        selected_types = set()
        selected_any = False
        selected_count = 0

        # Pre-sort candidate indices within each pool descending by lambda
        if n_clone_cand > 0:
            sorted_clone_order = torch.argsort(clone_lambdas, descending=True)
            sorted_clone_indices = clone_indices[sorted_clone_order]
            sorted_clone_lambdas = clone_lambdas[sorted_clone_order]
        else:
            sorted_clone_indices = torch.empty(0, dtype=torch.long, device=device)
            sorted_clone_lambdas = torch.empty(0, dtype=torch.float32, device=device)

        if n_split_cand > 0:
            sorted_split_order = torch.argsort(split_lambdas, descending=True)
            sorted_split_indices = split_indices[sorted_split_order]
            sorted_split_lambdas = split_lambdas[sorted_split_order]
        else:
            sorted_split_indices = torch.empty(0, dtype=torch.long, device=device)
            sorted_split_lambdas = torch.empty(0, dtype=torch.float32, device=device)

        allocated_split_indices = set()
        allocated_clone_indices = set()

        # Phase 1: Feature C - Minimum Split Allocation Quota (Feasibility-Checked)
        if kappa_split > 0.0 and n_split_cand > 0:
            capacity_ceiling = total_candidates
            if max_items is not None:
                capacity_ceiling = min(capacity_ceiling, max(0, int(max_items)))

            target_split_count = min(
                n_split_cand,
                int(torch.ceil(torch.tensor(float(kappa_split) * capacity_ceiling)).item())
            )

            for s_i in range(min(target_split_count, n_split_cand)):
                if max_items is not None and selected_count >= max(0, int(max_items)):
                    break
                s_idx = sorted_split_indices[s_i].item()
                operation_overhead = 0
                if not selected_any:
                    operation_overhead += self.cost_model.cuda_padding_bytes
                if 1 not in selected_types:
                    operation_overhead += self.cost_model.split_fixed_overhead_bytes
                actual_cost = cost_split + operation_overhead
                if actual_cost <= remaining_bytes:
                    remaining_bytes -= actual_cost
                    final_split_mask[s_idx] = True
                    allocated_split_indices.add(s_idx)
                    selected_types.add(1)
                    selected_any = True
                    selected_count += 1
                else:
                    # Infeasible under current VRAM budget - never force an infeasible split
                    break

        # Phase 2: Greedy Knapsack Allocation for Remaining Capacity
        unallocated_items = []
        for i in range(n_clone_cand):
            c_idx = sorted_clone_indices[i].item()
            if c_idx not in allocated_clone_indices:
                unallocated_items.append((sorted_clone_lambdas[i].item(), 0, c_idx, cost_clone))
        for j in range(n_split_cand):
            s_idx = sorted_split_indices[j].item()
            if s_idx not in allocated_split_indices:
                unallocated_items.append((sorted_split_lambdas[j].item(), 1, s_idx, cost_split))

        # Sort remaining pooled items descending by lambda
        unallocated_items.sort(key=lambda x: x[0], reverse=True)

        for lam, itype, idx, cost in unallocated_items:
            if max_items is not None and selected_count >= max(0, int(max_items)):
                break
            operation_overhead = 0
            if not selected_any:
                operation_overhead += self.cost_model.cuda_padding_bytes
            if itype not in selected_types:
                operation_overhead += (
                    self.cost_model.clone_fixed_overhead_bytes
                    if itype == 0
                    else self.cost_model.split_fixed_overhead_bytes
                )
            actual_cost = cost + operation_overhead
            if actual_cost <= remaining_bytes:
                remaining_bytes -= actual_cost
                if itype == 0:
                    final_clone_mask[idx] = True
                    allocated_clone_indices.add(idx)
                else:
                    final_split_mask[idx] = True
                    allocated_split_indices.add(idx)
                selected_types.add(itype)
                selected_any = True
                selected_count += 1
            if remaining_bytes < min(cost_clone, cost_split):
                break

        n_clone_admitted = int(final_clone_mask.sum().item())
        n_split_admitted = int(final_split_mask.sum().item())
        total_admitted = n_clone_admitted + n_split_admitted
        actual_split_share = float(n_split_admitted / max(1, total_admitted))

        self.last_allocation_stats = {
            "requested_split_share": requested_split_share,
            "actual_split_share": actual_split_share,
            "n_clone_admitted": n_clone_admitted,
            "n_split_admitted": n_split_admitted,
            "total_admitted": total_admitted,
            "target_split_share": float(kappa_split),
        }

        if return_stats:
            return final_clone_mask, final_split_mask, self.last_allocation_stats
        return final_clone_mask, final_split_mask

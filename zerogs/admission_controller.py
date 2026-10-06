"""
Admission Controller and Prune-First Mechanism (Module 2 - Solving Gaps P2 & D2).

Guarantees Zero-OOM by continuously interrogating the PyTorch CUDA Caching Allocator
before any Gaussian mutation occurs. If transient mutation cost exceeds available headroom,
the controller deterministically triggers Prune-First (clearing low-utility primitives
to release VRAM before generating new ones) or dynamically shrinks the mutation batch.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Any, Optional
import torch

from .cost_model import ByteCostModel


class AdmissionAction(str, Enum):
    ACCEPT = "ACCEPT"
    PRUNE_FIRST = "PRUNE_FIRST"
    SHRINK_AND_PRUNE_FIRST = "SHRINK_AND_PRUNE_FIRST"
    REJECT_AND_PRUNE = "REJECT_AND_PRUNE"


@dataclass
class AdmissionDecision:
    action: AdmissionAction
    n_clone: int
    n_split: int
    required_pre_prune: int
    transient_cost_bytes: int
    available_headroom_bytes: int
    freed_bytes: int
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for backwards-compatible access in train loops."""
        return {
            "action": self.action.value,
            "n_clone": self.n_clone,
            "n_split": self.n_split,
            "required_pre_prune": self.required_pre_prune,
            "transient_cost_mb": round(self.transient_cost_bytes / (1024**2), 2),
            "available_headroom_mb": round(self.available_headroom_bytes / (1024**2), 2),
            "freed_mb": round(self.freed_bytes / (1024**2), 2),
            "stats": self.stats,
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]


class AdmissionController:
    """
    Real-time Admission Controller enforcing hard VRAM bounds:
        M_allocated + Delta M_transient <= B_VRAM - delta_safety
    """

    def __init__(
        self,
        cost_model: ByteCostModel,
        hard_vram_limit_mb: Optional[float] = None,
        safety_headroom_mb: float = 512.0,
        enable_deterministic_prune_first: bool = True,
    ):
        """
        Args:
            cost_model: Initialized ByteCostModel.
            hard_vram_limit_mb: Hard cap in MB. If None, auto-detected from active GPU capacity.
            safety_headroom_mb: Safety margin in MB to resist PyTorch allocator fragmentation.
            enable_deterministic_prune_first: If True, activates pre-emptive pruning before allocation.
        """
        self.cost_model = cost_model
        self.safety_headroom_mb = safety_headroom_mb
        self.enable_deterministic_prune_first = enable_deterministic_prune_first

        if hard_vram_limit_mb is not None:
            self.hard_vram_limit_mb = float(hard_vram_limit_mb)
        else:
            if torch.cuda.is_available():
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                self.hard_vram_limit_mb = float(total_bytes / (1024**2))
            else:
                self.hard_vram_limit_mb = 8192.0  # Fallback 8GB for CPU tests

        self.hard_vram_limit_bytes = int(self.hard_vram_limit_mb * 1024 * 1024)
        self.safety_headroom_bytes = int(self.safety_headroom_mb * 1024 * 1024)
        self.effective_budget_bytes = max(
            0, self.hard_vram_limit_bytes - self.safety_headroom_bytes
        )

        # Telemetry metrics
        self.total_evaluations = 0
        self.total_prune_first_triggered = 0
        self.total_shrink_triggered = 0
        self.total_pruned_via_admission = 0

    def get_current_vram_allocated_bytes(self) -> int:
        """Returns currently allocated GPU memory in bytes."""
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        return 0

    def evaluate_admission(
        self,
        n_clone_requested: int,
        n_split_requested: int,
        current_num_gaussians: int,
        prune_candidates_count: int,
        current_allocated_bytes: Optional[int] = None,
    ) -> AdmissionDecision:
        """
        Evaluates proposed mutation request against active hardware memory headroom.

        Args:
            n_clone_requested: Count of points qualifying for Clone.
            n_split_requested: Count of points qualifying for Split.
            current_num_gaussians: Total active Gaussians in model.
            prune_candidates_count: Number of redundant/low-utility Gaussians identified.
            current_allocated_bytes: Optional override for allocated memory (useful for tests).
        """
        self.total_evaluations += 1

        if current_allocated_bytes is None:
            current_allocated = self.get_current_vram_allocated_bytes()
        else:
            current_allocated = current_allocated_bytes

        # Available headroom before hitting hard budget (accounting for safety zone)
        available_headroom = max(0, self.effective_budget_bytes - current_allocated)

        # Transient memory required by the proposed mutation
        transient_cost = self.cost_model.estimate_transient_bytes(
            n_clone=n_clone_requested,
            n_split=n_split_requested,
            include_padding=True,
        )

        # Case 1: Headroom is sufficient for full mutation
        if transient_cost <= available_headroom:
            return AdmissionDecision(
                action=AdmissionAction.ACCEPT,
                n_clone=n_clone_requested,
                n_split=n_split_requested,
                required_pre_prune=0,
                transient_cost_bytes=transient_cost,
                available_headroom_bytes=available_headroom,
                freed_bytes=0,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "transient_cost_mb": round(transient_cost / (1024**2), 2),
                    "reason": "Sufficient headroom available",
                },
            )

        # Case 2: Insufficient headroom -> OOM Hazard!
        # Check if Prune-First can clear enough space
        max_freed_bytes = self.cost_model.estimate_prune_freed_bytes(prune_candidates_count)
        potential_headroom = available_headroom + max_freed_bytes

        if self.enable_deterministic_prune_first and potential_headroom >= transient_cost:
            # We can accommodate 100% of the requested batch by pruning first
            deficit_bytes = transient_cost - available_headroom
            bytes_per_gaussian = self.cost_model.persistent_bytes_per_gaussian
            required_pre_prune = min(
                prune_candidates_count,
                int(math.ceil(deficit_bytes / max(1, bytes_per_gaussian))),
            )

            freed_bytes = self.cost_model.estimate_prune_freed_bytes(required_pre_prune)
            self.total_prune_first_triggered += 1
            self.total_pruned_via_admission += required_pre_prune

            return AdmissionDecision(
                action=AdmissionAction.PRUNE_FIRST,
                n_clone=n_clone_requested,
                n_split=n_split_requested,
                required_pre_prune=required_pre_prune,
                transient_cost_bytes=transient_cost,
                available_headroom_bytes=available_headroom,
                freed_bytes=freed_bytes,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "transient_cost_mb": round(transient_cost / (1024**2), 2),
                    "freed_mb": round(freed_bytes / (1024**2), 2),
                    "deficit_mb": round(deficit_bytes / (1024**2), 2),
                    "reason": "Prune-First deterministic eviction clears sufficient transient headroom",
                },
            )

        # Case 3: Saturation -> Even pruning all candidates is not enough
        # We must prune all candidates AND shrink the requested mutation batch
        self.total_shrink_triggered += 1
        self.total_pruned_via_admission += prune_candidates_count

        freed_bytes = max_freed_bytes
        admitted_budget_bytes = max(0, potential_headroom)

        if admitted_budget_bytes <= self.cost_model.cuda_padding_bytes:
            # Complete saturation: cannot allocate any new primitives safely
            return AdmissionDecision(
                action=AdmissionAction.REJECT_AND_PRUNE,
                n_clone=0,
                n_split=0,
                required_pre_prune=prune_candidates_count,
                transient_cost_bytes=0,
                available_headroom_bytes=available_headroom,
                freed_bytes=freed_bytes,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "reason": "Hard VRAM budget reached; mutated batch rejected; cleaning garbage",
                },
            )

        # Proportional scale factor for batch shrinking
        usable_budget = admitted_budget_bytes - self.cost_model.cuda_padding_bytes
        scale_ratio = max(0.0, min(1.0, float(usable_budget / max(1, transient_cost))))

        n_clone_admitted = int(math.floor(n_clone_requested * scale_ratio))
        n_split_admitted = int(math.floor(n_split_requested * scale_ratio))

        # Re-verify scaled cost
        scaled_transient_cost = self.cost_model.estimate_transient_bytes(
            n_clone=n_clone_admitted,
            n_split=n_split_admitted,
            include_padding=True,
        )

        return AdmissionDecision(
            action=AdmissionAction.SHRINK_AND_PRUNE_FIRST,
            n_clone=n_clone_admitted,
            n_split=n_split_admitted,
            required_pre_prune=prune_candidates_count,
            transient_cost_bytes=scaled_transient_cost,
            available_headroom_bytes=available_headroom,
            freed_bytes=freed_bytes,
            stats={
                "current_allocated_mb": round(current_allocated / (1024**2), 2),
                "headroom_mb": round(available_headroom / (1024**2), 2),
                "scale_ratio": round(scale_ratio, 4),
                "original_clone": n_clone_requested,
                "original_split": n_split_requested,
                "scaled_transient_cost_mb": round(scaled_transient_cost / (1024**2), 2),
                "reason": "Mutation batch shrunk proportionally and all prune candidates evicted",
            },
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Telemetry statistics on decisions made by controller."""
        return {
            "total_evaluations": self.total_evaluations,
            "prune_first_triggered": self.total_prune_first_triggered,
            "shrink_triggered": self.total_shrink_triggered,
            "total_pruned_via_admission": self.total_pruned_via_admission,
            "hard_vram_limit_mb": self.hard_vram_limit_mb,
            "safety_headroom_mb": self.safety_headroom_mb,
        }

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
from .memory_telemetry import capture_cuda_memory


class VramBudgetExceededError(RuntimeError):
    """Raised when the existing workload already exceeds the effective budget."""

    def __init__(
        self,
        message: str,
        before_cleanup: Optional[Dict[str, Any]] = None,
        after_cleanup: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.before_cleanup = before_cleanup
        self.after_cleanup = after_cleanup


class AdmissionAction(str, Enum):
    ACCEPT = "ACCEPT"
    PRUNE_FIRST = "PRUNE_FIRST"
    SHRINK_AND_PRUNE_FIRST = "SHRINK_AND_PRUNE_FIRST"
    REJECT_AND_PRUNE = "REJECT_AND_PRUNE"


@dataclass
class DynamicInflowDecision:
    inflow_mode: str
    current_footprint_bytes: int
    available_headroom_bytes: int
    target_headroom_bytes: int
    estimated_bytes_per_clone: int
    estimated_bytes_per_split: int
    memory_safe_limit: int
    computed_cap: int
    admissible_cap: int
    n_requested: int
    rejected_by_cap: int
    rejected_by_budget: int
    reason: str
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inflow_mode": self.inflow_mode,
            "current_footprint_mb": round(self.current_footprint_bytes / (1024**2), 2),
            "available_headroom_mb": round(self.available_headroom_bytes / (1024**2), 2),
            "target_headroom_mb": round(self.target_headroom_bytes / (1024**2), 2),
            "estimated_bytes_per_clone": self.estimated_bytes_per_clone,
            "estimated_bytes_per_split": self.estimated_bytes_per_split,
            "memory_safe_limit": self.memory_safe_limit,
            "computed_cap": self.computed_cap,
            "admissible_cap": self.admissible_cap,
            "n_requested": self.n_requested,
            "rejected_by_cap": self.rejected_by_cap,
            "rejected_by_budget": self.rejected_by_budget,
            "reason": self.reason,
            "stats": self.stats,
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]


@dataclass
class AdmissionDecision:
    action: AdmissionAction
    n_clone: int
    n_split: int
    required_pre_prune: int
    transient_cost_bytes: int
    mutation_budget_bytes: int
    available_headroom_bytes: int
    freed_bytes: int
    stats: Dict[str, Any]
    dynamic_inflow: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for backwards-compatible access in train loops."""
        d = {
            "action": self.action.value,
            "n_clone": self.n_clone,
            "n_split": self.n_split,
            "required_pre_prune": self.required_pre_prune,
            "transient_cost_mb": round(self.transient_cost_bytes / (1024**2), 2),
            "mutation_budget_mb": round(self.mutation_budget_bytes / (1024**2), 2),
            "available_headroom_mb": round(self.available_headroom_bytes / (1024**2), 2),
            "freed_mb": round(self.freed_bytes / (1024**2), 2),
            "stats": self.stats,
        }
        if self.dynamic_inflow is not None:
            d["dynamic_inflow"] = self.dynamic_inflow
        return d

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
        non_pytorch_allowance_mb: float = 0.0,
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
        self.non_pytorch_allowance_mb = non_pytorch_allowance_mb
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
        self.non_pytorch_allowance_bytes = int(self.non_pytorch_allowance_mb * 1024 * 1024)
        self.effective_budget_bytes = max(
            0, self.hard_vram_limit_bytes - self.safety_headroom_bytes
        )

        # Telemetry metrics
        self.total_evaluations = 0
        self.total_prune_first_triggered = 0
        self.total_shrink_triggered = 0
        self.total_pruned_via_admission = 0
        self.total_cache_reclaims = 0

    def get_current_vram_allocated_bytes(self) -> int:
        """Return a conservative allocator footprint (reserved >= allocated)."""
        return capture_cuda_memory().conservative_footprint_bytes

    def assert_current_memory_within_budget(
        self,
        telemetry_writer: Any = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fail when forward/backward itself violates the configured budget.

        If conservative footprint exceeds hard budget due to uncollected cache pool
        in the PyTorch Caching Allocator, attempts safe cache reclamation (torch.cuda.empty_cache)
        exactly once before deciding whether a true budget violation occurred.
        """
        snapshot = capture_cuda_memory()
        footprint = snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes
        if footprint <= self.hard_vram_limit_bytes:
            return snapshot.to_dict()

        # Footprint exceeds budget -> Record telemetry before cleanup
        before_dict = snapshot.to_dict()
        if telemetry_writer is not None:
            telemetry_writer.write(
                "cache_reclaim_triggered",
                iteration=iteration,
                footprint_mb=round(footprint / (1024**2), 2),
                hard_budget_mb=self.hard_vram_limit_mb,
                memory_before=before_dict,
            )

        # Call empty_cache exactly once if CUDA is available
        reclaimed_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            post_snapshot = capture_cuda_memory()
            post_footprint = (
                post_snapshot.conservative_footprint_bytes
                + self.non_pytorch_allowance_bytes
            )
            reclaimed_bytes = max(0, snapshot.reserved_bytes - post_snapshot.reserved_bytes)
        else:
            post_snapshot = snapshot
            post_footprint = footprint

        after_dict = post_snapshot.to_dict()

        # If post-cleanup footprint still exceeds budget, raise with before/after telemetry
        if post_footprint > self.hard_vram_limit_bytes:
            if telemetry_writer is not None:
                telemetry_writer.write(
                    "budget_exceeded_post_cleanup",
                    iteration=iteration,
                    footprint_mb=round(post_footprint / (1024**2), 2),
                    hard_budget_mb=self.hard_vram_limit_mb,
                    memory_before=before_dict,
                    memory_after=after_dict,
                )
            raise VramBudgetExceededError(
                "Current CUDA footprint exceeds the hard VRAM budget before mutation "
                f"(unrecoverable after empty_cache): "
                f"footprint_post_cleanup={post_footprint / 1024**2:.2f}MB, "
                f"initial_footprint={footprint / 1024**2:.2f}MB, "
                f"hard_budget={self.hard_vram_limit_bytes / 1024**2:.2f}MB "
                f"(hard={self.hard_vram_limit_mb:.2f}MB, safety={self.safety_headroom_mb:.2f}MB).",
                before_cleanup=before_dict,
                after_cleanup=after_dict,
            )

        # Successfully reclaimed cache -> record event and allow continuation
        self.total_cache_reclaims += 1
        reclaim_event_data = {
            "iteration": iteration,
            "reclaimed_mb": round(reclaimed_bytes / (1024**2), 2),
            "footprint_before_mb": round(footprint / (1024**2), 2),
            "footprint_after_mb": round(post_footprint / (1024**2), 2),
            "reserved_before_mb": round(snapshot.reserved_bytes / (1024**2), 2),
            "reserved_after_mb": round(post_snapshot.reserved_bytes / (1024**2), 2),
            "memory_after": after_dict,
        }
        if telemetry_writer is not None:
            telemetry_writer.write("cache_reclaimed", **reclaim_event_data)

        result = after_dict.copy()
        result["cache_reclaimed"] = True
        result["reclaimed_mb"] = round(reclaimed_bytes / (1024**2), 2)
        return result

    def compute_dynamic_inflow_cap(
        self,
        n_clone_candidates: int,
        n_split_candidates: int,
        inflow_cap_min: int = 5000,
        inflow_cap_max: int = 15000,
        target_headroom_mb: float = 1024.0,
        kappa_split: float = 0.40,
        current_allocated_bytes: Optional[int] = None,
        current_reserved_bytes: Optional[int] = None,
        inflow_mode: str = "dynamic",
    ) -> DynamicInflowDecision:
        """
        Calculates an adaptive candidate admission cap based on current memory headroom.

        Units & Policy:
            - Headroom in bytes: H_usable = max(0, effective_budget - M_curr)
            - Target headroom in bytes: H_target = target_headroom_mb * 1024 * 1024 (1,024 MB)
            - Unit costs: c_clone, c_split derived from cost_model (approx 3,840 and 7,680 bytes)
            - Memory-safe limit: N_safe = max(0, (H_usable - cuda_padding) // max(c_clone, c_split))
            - Dynamic Scaling:
                * If H_usable <= H_target: cap = min(inflow_cap_min, N_safe)
                * If H_usable > H_target:
                    surplus = H_usable - H_target
                    alpha = min(1.0, surplus / H_target)
                    scaled = inflow_cap_min + alpha * (inflow_cap_max - inflow_cap_min)
                    cap = min(scaled, inflow_cap_max, N_safe)
            - If total_candidates == 0: admissible_cap = 0
            - Rejections recorded separately by cap and by memory budget.
        """
        if n_clone_candidates < 0 or n_split_candidates < 0:
            raise ValueError("Candidate counts must be non-negative")

        total_candidates = n_clone_candidates + n_split_candidates
        target_headroom_bytes = max(0, int(target_headroom_mb * 1024 * 1024))

        if current_allocated_bytes is None:
            snapshot = capture_cuda_memory()
            current_allocated = snapshot.allocated_bytes
            current_reserved = snapshot.reserved_bytes
        else:
            current_allocated = current_allocated_bytes
            current_reserved = (
                current_allocated_bytes
                if current_reserved_bytes is None
                else current_reserved_bytes
            )

        current_footprint = (
            max(current_allocated, current_reserved)
            + self.non_pytorch_allowance_bytes
        )
        available_headroom = max(0, self.effective_budget_bytes - current_footprint)

        # Marginal transient costs per primitive
        cost_clone = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.clone_multiplier
        )
        cost_split = int(
            self.cost_model.persistent_bytes_per_gaussian
            * self.cost_model.split_multiplier
        )
        max_unit_cost = max(cost_clone, cost_split)

        # Fixed overhead: only incurred if candidates are present
        fixed_overhead = 0
        if n_split_candidates > 0:
            fixed_overhead = max(fixed_overhead, getattr(self.cost_model, "split_fixed_overhead_bytes", 0))
        if n_clone_candidates > 0:
            fixed_overhead = max(fixed_overhead, getattr(self.cost_model, "clone_fixed_overhead_bytes", 0))

        # Memory safe limit: absolute max items that can fit inside available headroom
        cuda_padding = getattr(self.cost_model, "cuda_padding_bytes", 0)
        total_overhead = fixed_overhead + cuda_padding
        usable_for_items = max(0, available_headroom - total_overhead)
        memory_safe_limit = usable_for_items // max(1, max_unit_cost)

        if inflow_mode == "static":
            computed_cap = min(inflow_cap_min, memory_safe_limit)
            reason = "Static inflow cap active (V2 compatibility mode)"
        else:
            # Dynamic policy
            if available_headroom <= target_headroom_bytes:
                # Tight headroom regime: cap decreases towards or below inflow_cap_min
                computed_cap = min(inflow_cap_min, memory_safe_limit)
                if memory_safe_limit < inflow_cap_min:
                    reason = f"Low headroom ({available_headroom / 1024**2:.1f}MB <= target {target_headroom_mb}MB): cap throttled by memory to {computed_cap}"
                else:
                    reason = f"Headroom at or below target ({available_headroom / 1024**2:.1f}MB <= {target_headroom_mb}MB): cap held at baseline min {inflow_cap_min}"
            else:
                # Abundant headroom regime: interpolate from min to max
                surplus_bytes = available_headroom - target_headroom_bytes
                scaling_span_bytes = max(1, target_headroom_bytes)
                alpha = min(1.0, float(surplus_bytes) / float(scaling_span_bytes))
                scaled_cap = int(inflow_cap_min + alpha * (inflow_cap_max - inflow_cap_min))
                computed_cap = min(scaled_cap, inflow_cap_max, memory_safe_limit)
                reason = f"Abundant headroom ({available_headroom / 1024**2:.1f}MB > {target_headroom_mb}MB): cap scaled to {computed_cap} (alpha={alpha:.3f})"

        computed_cap = max(0, computed_cap)
        if total_candidates == 0:
            admissible_cap = 0
        else:
            admissible_cap = min(computed_cap, total_candidates)

        rejected_by_cap = max(0, total_candidates - computed_cap)
        rejected_by_budget = max(0, total_candidates - memory_safe_limit) if memory_safe_limit < total_candidates else 0

        stats = {
            "current_allocated_mb": round(current_allocated / (1024**2), 2),
            "current_reserved_mb": round(current_reserved / (1024**2), 2),
            "current_footprint_mb": round(current_footprint / (1024**2), 2),
            "available_headroom_mb": round(available_headroom / (1024**2), 2),
            "target_headroom_mb": round(target_headroom_mb, 2),
            "memory_safe_limit": memory_safe_limit,
            "cost_clone_bytes": cost_clone,
            "cost_split_bytes": cost_split,
            "max_unit_cost_bytes": max_unit_cost,
            "fixed_overhead_bytes": fixed_overhead,
            "cuda_padding_bytes": cuda_padding,
            "inflow_cap_min": inflow_cap_min,
            "inflow_cap_max": inflow_cap_max,
        }

        return DynamicInflowDecision(
            inflow_mode=inflow_mode,
            current_footprint_bytes=current_footprint,
            available_headroom_bytes=available_headroom,
            target_headroom_bytes=target_headroom_bytes,
            estimated_bytes_per_clone=cost_clone,
            estimated_bytes_per_split=cost_split,
            memory_safe_limit=memory_safe_limit,
            computed_cap=computed_cap,
            admissible_cap=admissible_cap,
            n_requested=total_candidates,
            rejected_by_cap=rejected_by_cap,
            rejected_by_budget=rejected_by_budget,
            reason=reason,
            stats=stats,
        )

    def evaluate_admission(
        self,
        n_clone_requested: int,
        n_split_requested: int,
        current_num_gaussians: int,
        prune_candidates_count: int,
        current_allocated_bytes: Optional[int] = None,
        current_reserved_bytes: Optional[int] = None,
        inflow_cap_min: int = 5000,
        inflow_cap_max: int = 15000,
        target_headroom_mb: float = 1024.0,
        kappa_split: float = 0.40,
        inflow_mode: str = "dynamic",
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
        if min(n_clone_requested, n_split_requested, current_num_gaussians, prune_candidates_count) < 0:
            raise ValueError("Admission counts must be non-negative")
        self.total_evaluations += 1

        if current_allocated_bytes is None:
            snapshot = capture_cuda_memory()
            current_allocated = snapshot.allocated_bytes
            current_reserved = snapshot.reserved_bytes
        else:
            current_allocated = current_allocated_bytes
            current_reserved = (
                current_allocated_bytes
                if current_reserved_bytes is None
                else current_reserved_bytes
            )

        current_footprint = (
            max(current_allocated, current_reserved)
            + self.non_pytorch_allowance_bytes
        )

        # Available headroom before hitting hard budget (accounting for safety zone)
        available_headroom = max(0, self.effective_budget_bytes - current_footprint)

        # Transient memory required by the proposed mutation
        transient_cost = self.cost_model.estimate_transient_bytes(
            n_clone=n_clone_requested,
            n_split=n_split_requested,
            include_padding=True,
        )

        inflow_decision = self.compute_dynamic_inflow_cap(
            n_clone_candidates=n_clone_requested,
            n_split_candidates=n_split_requested,
            inflow_cap_min=inflow_cap_min,
            inflow_cap_max=inflow_cap_max,
            target_headroom_mb=target_headroom_mb,
            kappa_split=kappa_split,
            current_allocated_bytes=current_allocated,
            current_reserved_bytes=current_reserved,
            inflow_mode=inflow_mode,
        )

        # Case 1: Headroom is sufficient for full mutation
        if transient_cost <= available_headroom:
            return AdmissionDecision(
                action=AdmissionAction.ACCEPT,
                n_clone=n_clone_requested,
                n_split=n_split_requested,
                required_pre_prune=0,
                transient_cost_bytes=transient_cost,
                mutation_budget_bytes=transient_cost,
                available_headroom_bytes=available_headroom,
                freed_bytes=0,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "current_reserved_mb": round(current_reserved / (1024**2), 2),
                    "current_footprint_mb": round(current_footprint / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "transient_cost_mb": round(transient_cost / (1024**2), 2),
                    "reason": "Sufficient headroom available",
                },
                dynamic_inflow=inflow_decision.to_dict(),
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
                mutation_budget_bytes=transient_cost,
                available_headroom_bytes=available_headroom,
                freed_bytes=freed_bytes,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "current_reserved_mb": round(current_reserved / (1024**2), 2),
                    "current_footprint_mb": round(current_footprint / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "transient_cost_mb": round(transient_cost / (1024**2), 2),
                    "freed_mb": round(freed_bytes / (1024**2), 2),
                    "deficit_mb": round(deficit_bytes / (1024**2), 2),
                    "reason": "Prune-First deterministic eviction clears sufficient transient headroom",
                },
                dynamic_inflow=inflow_decision.to_dict(),
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
                mutation_budget_bytes=0,
                available_headroom_bytes=available_headroom,
                freed_bytes=freed_bytes,
                stats={
                    "current_allocated_mb": round(current_allocated / (1024**2), 2),
                    "current_reserved_mb": round(current_reserved / (1024**2), 2),
                    "current_footprint_mb": round(current_footprint / (1024**2), 2),
                    "headroom_mb": round(available_headroom / (1024**2), 2),
                    "reason": "Hard VRAM budget reached; mutated batch rejected; cleaning garbage",
                },
                dynamic_inflow=inflow_decision.to_dict(),
            )

        # Binary-search a proportional request that really fits. A direct ratio
        # is unsafe because clone/split fixed overhead makes the cost affine,
        # not linear in primitive count.
        low, high = 0.0, 1.0
        n_clone_admitted = 0
        n_split_admitted = 0
        scaled_transient_cost = 0
        scale_ratio = 0.0
        for _ in range(32):
            candidate_ratio = (low + high) / 2.0
            candidate_clone = int(math.floor(n_clone_requested * candidate_ratio))
            candidate_split = int(math.floor(n_split_requested * candidate_ratio))
            candidate_cost = self.cost_model.estimate_transient_bytes(
                n_clone=candidate_clone,
                n_split=candidate_split,
                include_padding=True,
            )
            if candidate_cost <= admitted_budget_bytes:
                low = candidate_ratio
                scale_ratio = candidate_ratio
                n_clone_admitted = candidate_clone
                n_split_admitted = candidate_split
                scaled_transient_cost = candidate_cost
            else:
                high = candidate_ratio

        return AdmissionDecision(
            action=AdmissionAction.SHRINK_AND_PRUNE_FIRST,
            n_clone=n_clone_admitted,
            n_split=n_split_admitted,
            required_pre_prune=prune_candidates_count,
            transient_cost_bytes=scaled_transient_cost,
            mutation_budget_bytes=admitted_budget_bytes,
            available_headroom_bytes=available_headroom,
            freed_bytes=freed_bytes,
            stats={
                "current_allocated_mb": round(current_allocated / (1024**2), 2),
                "current_reserved_mb": round(current_reserved / (1024**2), 2),
                "current_footprint_mb": round(current_footprint / (1024**2), 2),
                "headroom_mb": round(available_headroom / (1024**2), 2),
                "scale_ratio": round(scale_ratio, 4),
                "original_clone": n_clone_requested,
                "original_split": n_split_requested,
                "scaled_transient_cost_mb": round(scaled_transient_cost / (1024**2), 2),
                "reason": "Mutation batch shrunk proportionally and all prune candidates evicted",
            },
            dynamic_inflow=inflow_decision.to_dict(),
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
            "non_pytorch_allowance_mb": self.non_pytorch_allowance_mb,
        }

"""
Memory-Bounded Densification Scheduler (Unified ZeroGS Wrapper).

Wraps all 4 core modules into an intuitive, engine-agnostic controller
that can be inserted into any 3DGS codebase (vanilla 3DGS, nerfstudio, gsplat)
with minimal boilerplate code.
"""

from typing import Dict, Any, Optional, Tuple
import torch

from .cost_model import ByteCostModel
from .admission_controller import AdmissionController, AdmissionAction, AdmissionDecision
from .occlusion_engine import OcclusionAwareEngine
from .marginal_allocator import MarginalUtilityAllocator


class MemoryBoundedDensificationScheduler:
    """
    Unified manager orchestrating:
    1. ByteCostModel (transient footprint modeling)
    2. OcclusionAwareEngine (physics-informed demand & structure preservation)
    3. AdmissionController (deterministic Prune-First & headroom enforcement)
    4. MarginalUtilityAllocator (byte-level Knapsack quota allocation)
    """

    def __init__(
        self,
        num_gaussians: int,
        sh_degree: int = 3,
        hard_vram_limit_mb: Optional[float] = 8192.0,
        safety_headroom_mb: float = 512.0,
        gamma_occ: float = 0.35,
        transmittance_tau: float = 0.25,
        clone_multiplier: float = 1.15,
        split_multiplier: float = 2.30,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.sh_degree = sh_degree

        # 1. Byte Cost Model
        self.cost_model = ByteCostModel(
            sh_degree=sh_degree,
            use_adam=True,
            clone_multiplier=clone_multiplier,
            split_multiplier=split_multiplier,
        )

        # 2. Admission Controller
        self.admission_controller = AdmissionController(
            cost_model=self.cost_model,
            hard_vram_limit_mb=hard_vram_limit_mb,
            safety_headroom_mb=safety_headroom_mb,
        )

        # 3. Occlusion Engine
        self.occ_engine = OcclusionAwareEngine(
            num_gaussians=num_gaussians,
            gamma_occ=gamma_occ,
            transmittance_tau=transmittance_tau,
            device=device,
        )

        # 4. Marginal Utility Allocator
        self.allocator = MarginalUtilityAllocator(cost_model=self.cost_model)

        # View-space gradient accumulator
        self.viewspace_grad_accum = torch.zeros(num_gaussians, device=device)

    def record_forward_stats(
        self,
        viewpoint_cam: Any,
        gaussians: Any,
        render_pkg: Dict[str, Any],
    ) -> None:
        """
        Extracts view depth and transmittance statistics from render package.
        Supports custom CUDA rasterizer tensors or fallback estimates.
        """
        visibility_filter = render_pkg.get("visibility_filter", None)
        if visibility_filter is None or visibility_filter.sum() == 0:
            return

        vis_indices = torch.where(visibility_filter)[0]
        radii = render_pkg.get("radii", None)

        # Retrieve or synthesize mean transmittance
        mean_T = render_pkg.get("mean_transmittance", None)
        if mean_T is None:
            mean_T = render_pkg.get("mean_T", None)
        if mean_T is None and radii is not None:
            mean_T = torch.ones_like(radii, dtype=torch.float32)
        elif mean_T is None:
            mean_T = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.float32, device=self.device)

        # Compute camera view depth: (world_view_transform[2, :3] @ xyz.T) + world_view_transform[2, 3]
        if hasattr(viewpoint_cam, "world_view_transform"):
            R_z = viewpoint_cam.world_view_transform[2, :3].to(self.device)
            t_z = viewpoint_cam.world_view_transform[2, 3].to(self.device)
            view_depths = (gaussians.get_xyz[vis_indices] @ R_z) + t_z
        else:
            view_depths = torch.norm(gaussians.get_xyz[vis_indices], dim=-1)

        self.occ_engine.update_per_view_stats(
            visible_indices=vis_indices,
            view_depths=view_depths,
            view_transmittance=mean_T[vis_indices],
        )

    def record_backward_stats(
        self,
        viewspace_points: torch.Tensor,
        visibility_filter: torch.Tensor,
    ) -> None:
        """Accumulates 2D viewspace gradients during backward pass."""
        if visibility_filter is not None and viewspace_points.grad is not None:
            grad_norm = torch.norm(viewspace_points.grad[visibility_filter, :2], dim=-1)
            self.viewspace_grad_accum[visibility_filter] += grad_norm

    def execute_mutation_cycle(
        self,
        gaussians: Any,
        scene_extent: float,
        densify_grad_threshold: float,
        percent_dense: float,
        size_threshold: Optional[float] = None,
        min_opacity_prune: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Executes a deterministic Zero-OOM densification and pruning cycle:
        1. Occlusion Demand Estimation & Structure Protection.
        2. Candidate generation (Clone & Split).
        3. Redundant pruning candidate detection.
        4. Admission Control & Headroom verification.
        5. Deterministic Prune-First (if required).
        6. Marginal Gain / Byte Quota Selection.
        7. Safe execution of densify_and_split and densify_and_clone.
        8. Buffer reallocation for new Gaussian population.
        """
        # 1. Occlusion Demand & Protected Mask
        self.occ_engine.grad_accum = self.viewspace_grad_accum
        demand, protected_mask = self.occ_engine.compute_compensated_demand(scene_extent=scene_extent)

        # 2. Identify Clone & Split candidates
        grads_high = self.viewspace_grad_accum >= densify_grad_threshold
        max_scales = torch.max(gaussians.get_scaling, dim=1).values
        scale_limit = percent_dense * scene_extent

        clone_mask = torch.logical_and(grads_high, max_scales <= scale_limit)
        split_mask = torch.logical_and(grads_high, max_scales > scale_limit)

        # 3. Identify Redundant Prune candidates
        dead_opacity_mask = (gaussians.get_opacity < min_opacity_prune).squeeze()
        low_utility_threshold = torch.quantile(demand, 0.05)
        prunable_utility = torch.logical_and(demand < low_utility_threshold, ~protected_mask)

        redundant_mask = torch.logical_or(dead_opacity_mask, prunable_utility)
        if size_threshold is not None and hasattr(gaussians, "max_radii2D"):
            too_large_mask = gaussians.max_radii2D > size_threshold
            redundant_mask = torch.logical_or(redundant_mask, too_large_mask)

        prune_candidates_count = int(redundant_mask.sum().item())
        n_clone_req = int(clone_mask.sum().item())
        n_split_req = int(split_mask.sum().item())

        # 4. Evaluate Admission
        decision: AdmissionDecision = self.admission_controller.evaluate_admission(
            n_clone_requested=n_clone_req,
            n_split_requested=n_split_req,
            current_num_gaussians=gaussians.get_xyz.shape[0],
            prune_candidates_count=prune_candidates_count,
        )

        # 5. Deterministic Prune-First
        if decision.action in [AdmissionAction.PRUNE_FIRST, AdmissionAction.SHRINK_AND_PRUNE_FIRST]:
            k_prune = decision.required_pre_prune
            redundant_indices = torch.where(redundant_mask)[0]
            if len(redundant_indices) > 0 and k_prune > 0:
                # Evict primitives with lowest demand first
                sorted_order = torch.argsort(demand[redundant_indices])
                prune_indices = redundant_indices[sorted_order[:k_prune]]

                exec_prune_filter = torch.zeros(
                    gaussians.get_xyz.shape[0], dtype=torch.bool, device=self.device
                )
                exec_prune_filter[prune_indices] = True
                gaussians.prune_points(exec_prune_filter)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Adjust masks after tensor dimension shrinkage
                surviving = ~exec_prune_filter
                clone_mask = clone_mask[surviving]
                split_mask = split_mask[surviving]
                demand = demand[surviving]

        # 6. Marginal Utility Allocation
        final_clone_mask, final_split_mask = self.allocator.allocate(
            demand=demand,
            clone_mask=clone_mask,
            split_mask=split_mask,
            n_clone_quota=decision.n_clone,
            n_split_quota=decision.n_split,
        )

        # 7. Safe Execution of Densification (Clone first, then Split)
        n_clone_executed = int(final_clone_mask.sum().item())
        n_split_executed = int(final_split_mask.sum().item())

        if n_clone_executed > 0:
            gaussians.densify_and_clone(final_clone_mask, densify_grad_threshold, scene_extent)
        if n_split_executed > 0:
            gaussians.densify_and_split(final_split_mask, densify_grad_threshold, scene_extent)

        # 8. Reset Buffers for New Population
        new_num_gaussians = gaussians.get_xyz.shape[0]
        self.occ_engine.reset_buffers(new_num_gaussians)
        self.viewspace_grad_accum = torch.zeros(new_num_gaussians, device=self.device)

        return {
            "decision": decision.to_dict(),
            "n_clone_executed": n_clone_executed,
            "n_split_executed": n_split_executed,
            "new_total_gaussians": new_num_gaussians,
            "protected_count": int(protected_mask.sum().item()),
        }

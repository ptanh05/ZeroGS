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
from .occlusion_engine import OcclusionAwareEngine, bottom_utility_mask
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
        require_native_stats: bool = True,
        occlusion_mode: str = "full",
        occlusion_physics_mode: str = "corrected",
        enable_structure_protection: bool = True,
        allocator_strategy: str = "gain_per_byte",
        gaussians_per_cycle_cap: int = 5000,
        max_gaussians_cap: int = 1_000_000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.sh_degree = sh_degree
        self.require_native_stats = require_native_stats
        self.allocator_strategy = allocator_strategy
        self.gaussians_per_cycle_cap = gaussians_per_cycle_cap
        self.max_gaussians_cap = max_gaussians_cap

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
            occlusion_mode=occlusion_mode,
            occlusion_physics_mode=occlusion_physics_mode,
            enable_structure_protection=enable_structure_protection,
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
        if self.require_native_stats and not bool(render_pkg.get("stats_available", False)):
            raise RuntimeError(
                "Occlusion-aware scheduling requires native rasterizer statistics. "
                "Rebuild the ZeroGS CUDA extension; synthetic fallback statistics "
                "must not be used for research runs."
            )
        visibility_filter = render_pkg.get("visibility_filter", None)
        if visibility_filter is None or visibility_filter.sum() == 0:
            return

        vis_indices = torch.where(visibility_filter)[0]
        radii = render_pkg.get("radii", None)

        # Retrieve native mean transmittance. Missing values are invalid because
        # a T=1 fallback silently disables occlusion compensation.
        mean_T = render_pkg.get("mean_transmittance", None)
        if mean_T is None:
            mean_T = render_pkg.get("mean_T", None)
        if mean_T is None and self.require_native_stats:
            raise RuntimeError("Render package declares stats_available but mean_T is missing")
        if mean_T is None:
            mean_T = torch.ones_like(radii, dtype=torch.float32)

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
            view_depth_variance=(
                render_pkg["depth_var"][vis_indices]
                if render_pkg.get("depth_var") is not None
                else None
            ),
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
        max_world_size_ratio: float = 0.1,
        iteration: int = 0,
        last_opacity_reset_iteration: int = -999999,
        tau_age: int = 0,
        tau_reset: int = 0,
        kappa_split: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executes a deterministic Zero-OOM densification and pruning cycle:
        1. Occlusion Demand Estimation & Structure Protection.
        2. Candidate generation (Clone & Split).
        3. Stage 1: Regular Hygiene Pruning (dead opacity + floaters when size_threshold active).
        4. Stage 2: Admission Control & Prune-First (bottom utility excluding protected structures).
        5. Marginal Gain / Byte Quota Selection (joint knapsack).
        6. Safe execution of densify_and_clone and densify_and_split.
        7. Buffer reallocation for new Gaussian population.
        """
        # 1. Occlusion Demand & Protected Mask
        self.occ_engine.grad_accum = self.viewspace_grad_accum
        demand, protected_mask = self.occ_engine.compute_compensated_demand(scene_extent=scene_extent)
        demand = torch.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Identify Clone & Split candidates
        grads_high = self.viewspace_grad_accum >= densify_grad_threshold
        max_scales = torch.max(gaussians.get_scaling, dim=1).values
        scale_limit = percent_dense * scene_extent

        clone_mask = torch.logical_and(grads_high, max_scales <= scale_limit)
        split_mask = torch.logical_and(grads_high, max_scales > scale_limit)

        # 3. STAGE 1: REGULAR HYGIENE PRUNING (Always executed independently of admission!)
        is_in_reset_cooldown = (iteration - last_opacity_reset_iteration) < tau_reset
        raw_dead_opacity = (gaussians.get_opacity < min_opacity_prune).squeeze()

        n_protected_by_cooldown = 0
        n_protected_by_age = 0

        if is_in_reset_cooldown:
            # During reset cooldown, suspend only opacity-based Stage 1 eviction
            n_protected_by_cooldown = int(raw_dead_opacity.sum().item())
            dead_opacity_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=self.device)
        else:
            # Exempt young Gaussians from opacity eviction
            if hasattr(gaussians, "birth_iteration") and gaussians.birth_iteration is not None and gaussians.birth_iteration.numel() > 0:
                is_young = (iteration - gaussians.birth_iteration) < tau_age
                n_protected_by_age = int((raw_dead_opacity & is_young).sum().item())
                dead_opacity_mask = raw_dead_opacity & (~is_young)
            else:
                dead_opacity_mask = raw_dead_opacity

        # Physical size/scale hygiene remains active unconditionally
        hygiene_mask = dead_opacity_mask
        if size_threshold is not None:
            if hasattr(gaussians, "max_radii2D"):
                hygiene_mask = torch.logical_or(hygiene_mask, gaussians.max_radii2D > size_threshold)
            big_ws = max_scales > (max_world_size_ratio * scene_extent)
            hygiene_mask = torch.logical_or(hygiene_mask, big_ws)

        executed_hygiene_prune = 0
        if hygiene_mask.any():
            executed_hygiene_prune = int(hygiene_mask.sum().item())
            gaussians.prune_points(hygiene_mask)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Synchronize candidate and auxiliary tensors after Stage 1 prune
            surviving_1 = ~hygiene_mask
            clone_mask = clone_mask[surviving_1]
            split_mask = split_mask[surviving_1]
            demand = demand[surviving_1]
            protected_mask = protected_mask[surviving_1]

        # 4. STAGE 2: ADMISSION CONTROL & OPTIONAL PRUNE-FIRST
        n_clone_req = int(clone_mask.sum().item())
        n_split_req = int(split_mask.sum().item())

        prunable_utility = bottom_utility_mask(demand, protected_mask, fraction=0.05)
        prune_candidates_count = int(prunable_utility.sum().item())

        decision: AdmissionDecision = self.admission_controller.evaluate_admission(
            n_clone_requested=n_clone_req,
            n_split_requested=n_split_req,
            current_num_gaussians=gaussians.get_xyz.shape[0],
            prune_candidates_count=prune_candidates_count,
        )

        executed_pre_prune = 0
        if decision.action in [AdmissionAction.PRUNE_FIRST, AdmissionAction.SHRINK_AND_PRUNE_FIRST]:
            k_prune = decision.required_pre_prune
            utility_indices = torch.where(prunable_utility)[0]
            if len(utility_indices) > 0 and k_prune > 0:
                # Evict primitives with lowest demand first
                sorted_order = torch.argsort(demand[utility_indices])
                prune_indices = utility_indices[sorted_order[:k_prune]]

                exec_prune_filter = torch.zeros(
                    gaussians.get_xyz.shape[0], dtype=torch.bool, device=self.device
                )
                exec_prune_filter[prune_indices] = True
                gaussians.prune_points(exec_prune_filter)
                executed_pre_prune = int(exec_prune_filter.sum().item())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Adjust masks after Stage 2 tensor dimension shrinkage
                surviving_2 = ~exec_prune_filter
                clone_mask = clone_mask[surviving_2]
                split_mask = split_mask[surviving_2]
                demand = demand[surviving_2]
                protected_mask = protected_mask[surviving_2]

        # 5. Marginal Utility Allocation
        final_clone_mask, final_split_mask = self.allocator.joint_knapsack_allocate(
            demand=demand,
            clone_mask=clone_mask,
            split_mask=split_mask,
            max_transient_bytes=decision.mutation_budget_bytes,
            strategy=self.allocator_strategy,
            max_items=min(
                self.gaussians_per_cycle_cap,
                max(0, self.max_gaussians_cap - int(gaussians.get_xyz.shape[0])),
            ),
            kappa_split=kappa_split,
        )

        # 6. Safe Execution of Densification (Clone first, then Split)
        n_clone_executed = int(final_clone_mask.sum().item())
        n_split_executed = int(final_split_mask.sum().item())

        if n_clone_executed > 0:
            gaussians.densify_and_clone(final_clone_mask, densify_grad_threshold, scene_extent, iteration=iteration)
        if n_split_executed > 0:
            gaussians.densify_and_split(final_split_mask, densify_grad_threshold, scene_extent, iteration=iteration)

        # 7. Reset Buffers for New Population
        new_num_gaussians = gaussians.get_xyz.shape[0]
        self.occ_engine.reset_buffers(new_num_gaussians)
        self.viewspace_grad_accum = torch.zeros(new_num_gaussians, device=self.device)

        alloc_stats = getattr(self.allocator, "last_allocation_stats", {})
        return {
            "decision": decision.to_dict(),
            "n_clone_executed": n_clone_executed,
            "n_split_executed": n_split_executed,
            "executed_hygiene_prune": executed_hygiene_prune,
            "executed_pre_prune": executed_pre_prune,
            "new_total_gaussians": new_num_gaussians,
            "protected_count": int(protected_mask.sum().item()),
            "n_protected_by_age": n_protected_by_age,
            "n_protected_by_cooldown": n_protected_by_cooldown,
            "requested_split_share": alloc_stats.get("requested_split_share", 0.0),
            "actual_split_share": alloc_stats.get("actual_split_share", 0.0),
            "tau_age": tau_age,
            "tau_reset": tau_reset,
            "kappa_split": kappa_split,
        }

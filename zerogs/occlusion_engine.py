"""
Occlusion-Aware Demand Engine (Module 4 - Solving Gap S1).

Addresses the 0.20 - 0.28 dB PSNR drop observed on complex occluded scenes
(bonsai, flowers, garden from SPARE-GS Table VII).

Normalizes raw visibility oscillations across depth discontinuities by extracting
mean transmittance (T_bar) and depth variance (sigma_z) directly from the forward pass,
producing a compensated demand score and protecting micro-structures (branches, leaves, petals)
from catastrophic over-pruning.
"""

from typing import Tuple, Optional
import torch


def bottom_utility_mask(
    demand: torch.Tensor,
    protected_mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select an exact deterministic bottom-k set while excluding protected points."""
    if demand.ndim != 1 or protected_mask.shape != demand.shape:
        raise ValueError("demand and protected_mask must be matching 1D tensors")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    result = torch.zeros_like(protected_mask, dtype=torch.bool)
    if demand.numel() == 0 or fraction == 0.0:
        return result
    target_count = min(demand.numel(), max(1, int(demand.numel() * fraction)))
    eligible_indices = torch.where(~protected_mask)[0]
    if eligible_indices.numel() == 0:
        return result
    selected_count = min(target_count, eligible_indices.numel())
    order = torch.argsort(demand[eligible_indices], stable=True)[:selected_count]
    result[eligible_indices[order]] = True
    return result


class OcclusionAwareEngine:
    """
    Stateful engine tracking per-view depth and transmittance statistics across training steps,
    producing compensated demand and structural protection masks before densification.
    """

    def __init__(
        self,
        num_gaussians: int,
        gamma_occ: float = 0.35,
        transmittance_tau: float = 0.25,
        depth_epsilon: float = 1e-4,
        occlusion_mode: str = "full",
        enable_structure_protection: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            num_gaussians: Initial number of Gaussians.
            gamma_occ: Occlusion compensation weight (default: 0.35).
            transmittance_tau: Transmittance threshold below which a point is deemed foreground/occluding (default: 0.25).
            depth_epsilon: Numerical stability epsilon.
            device: Computing device ('cuda' or 'cpu').
        """
        self.gamma_occ = gamma_occ
        self.transmittance_tau = transmittance_tau
        self.depth_epsilon = depth_epsilon
        if occlusion_mode not in {"full", "transmittance_only", "depth_only", "none"}:
            raise ValueError(f"Unsupported occlusion_mode: {occlusion_mode}")
        self.occlusion_mode = occlusion_mode
        self.enable_structure_protection = enable_structure_protection
        self.device = device

        self.grad_accum: Optional[torch.Tensor] = None
        self.reset_buffers(num_gaussians)

    def reset_buffers(self, num_gaussians: int) -> None:
        """Re-initializes accumulation buffers when primitive count changes (N -> N')."""
        self.num_gaussians = num_gaussians
        self.accum_vis = torch.zeros(num_gaussians, dtype=torch.int32, device=self.device)
        self.accum_T = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)
        self.accum_depth = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)
        self.accum_depth_sq = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)
        self.accum_depth_var = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)
        self.grad_accum = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)

    def update_per_view_stats(
        self,
        visible_indices: torch.Tensor,
        view_depths: torch.Tensor,
        view_transmittance: torch.Tensor,
        view_depth_variance: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates statistics after a camera forward pass.

        Args:
            visible_indices: 1D Tensor of indices of Gaussians visible in this camera view.
            view_depths: 1D Tensor of depth values in camera coordinate space (z > 0).
            view_transmittance: 1D Tensor of accumulated transmittances T.
            view_depth_variance: Optional native within-view depth variance per Gaussian.
        """
        if visible_indices is None or len(visible_indices) == 0:
            return

        idx = visible_indices.to(self.device).long()
        depths = view_depths.to(self.device).float()
        T = view_transmittance.to(self.device).float()
        within_view_var = (
            torch.zeros_like(depths)
            if view_depth_variance is None
            else view_depth_variance.to(self.device).float()
        )
        expected_shape = idx.shape
        for name, value in (("view_depths", depths), ("view_transmittance", T), ("view_depth_variance", within_view_var)):
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match visible indices {tuple(expected_shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if torch.any(T < 0) or torch.any(T > 1):
            raise ValueError("view_transmittance must be bounded in [0, 1]")
        if torch.any(within_view_var < 0):
            raise ValueError("view_depth_variance must be non-negative")

        self.accum_vis[idx] += 1
        self.accum_T[idx] += T
        self.accum_depth[idx] += depths
        self.accum_depth_sq[idx] += depths * depths
        self.accum_depth_var[idx] += within_view_var

    def compute_compensated_demand(
        self,
        scene_extent: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the occlusion-compensated demand and the protected structure mask.

        Returns:
            demand: 1D Tensor of compensated demand values for all Gaussians.
            protected_mask: 1D Boolean Tensor where True indicates structural elements
                            (foliage, thin branches) protected from utility pruning.
        """
        # Ensure denominator is at least 1 to avoid division by zero
        observed_counts = self.accum_vis.float()
        vis_counts = torch.clamp(observed_counts, min=1.0)
        observed = observed_counts > 0

        # Mean transmittance and mean depth
        mean_T = self.accum_T / vis_counts
        mean_depth = self.accum_depth / vis_counts
        mean_depth_sq = self.accum_depth_sq / vis_counts

        # Law of total variance: E[Var(z|view)] + Var(E[z|view]).
        between_view_var = torch.clamp(mean_depth_sq - (mean_depth**2), min=0.0)
        within_view_var = self.accum_depth_var / vis_counts
        depth_var = torch.clamp(between_view_var + within_view_var, min=0.0)
        depth_std = torch.sqrt(depth_var + self.depth_epsilon)

        # Occlusion modulation factor Omega_i:
        # High depth variance (boundary across cameras) + Low transmittance (surface/dense)
        # -> High occlusion sensitivity
        normalized_depth_fluctuation = depth_std / (float(scene_extent) + self.depth_epsilon)
        if self.occlusion_mode == "full":
            occlusion_signal = normalized_depth_fluctuation * (1.0 - mean_T)
        elif self.occlusion_mode == "transmittance_only":
            occlusion_signal = 1.0 - mean_T
        elif self.occlusion_mode == "depth_only":
            occlusion_signal = normalized_depth_fluctuation
        else:
            occlusion_signal = torch.zeros_like(mean_T)
        occlusion_factor = 1.0 + self.gamma_occ * occlusion_signal

        # Baseline demand: Viewspace gradient * Visibility count
        if self.grad_accum is None:
            raw_demand = vis_counts
        else:
            raw_demand = self.grad_accum * vis_counts

        compensated_demand = torch.where(observed, raw_demand * occlusion_factor, torch.zeros_like(raw_demand))

        # Protected structure mask:
        # Gaussians with significant depth variance (thin structures visible from distinct angles)
        # and low transmittance should NOT be eliminated by bottom-quantile pruning!
        is_high_depth_var = depth_std > (0.01 * float(scene_extent))
        is_occluding = mean_T < self.transmittance_tau
        if not self.enable_structure_protection or self.occlusion_mode == "none":
            protected_mask = torch.zeros_like(observed)
        elif self.occlusion_mode == "transmittance_only":
            protected_mask = torch.logical_and(is_occluding, observed)
        elif self.occlusion_mode == "depth_only":
            protected_mask = torch.logical_and(is_high_depth_var, observed)
        else:
            protected_mask = torch.logical_and(
                torch.logical_and(is_high_depth_var, is_occluding), observed
            )

        return compensated_demand, protected_mask


class OcclusionAwareDemandEstimator:
    """
    Stateless / functional variant for direct computation on pre-accumulated tensors.
    """

    def __init__(
        self,
        gamma_occlusion: float = 0.35,
        depth_epsilon: float = 1e-4,
        sh_degree: int = 3,
    ):
        self.gamma_occlusion = gamma_occlusion
        self.depth_epsilon = depth_epsilon
        self.sh_degree = sh_degree

    def compute_marginal_demand(
        self,
        grad_accum: torch.Tensor,
        visibility_count: torch.Tensor,
        depth_variance: torch.Tensor,
        mean_transmittance: torch.Tensor,
        radii: torch.Tensor,
        cost_clone_bytes: float = 1085.6,
        cost_split_bytes: float = 2171.2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes compensated demand along with marginal gain per byte for Clone and Split.

        Returns:
            demand: Compensated demand tensor.
            gain_clone: Marginal gain per byte for Clone (demand / cost_clone_bytes).
            gain_split: Marginal gain per byte for Split (demand / cost_split_bytes).
        """
        depth_std = torch.sqrt(torch.clamp(depth_variance, min=0.0) + self.depth_epsilon)
        depth_norm = depth_std / (torch.max(depth_std) + self.depth_epsilon)

        occlusion_weight = 1.0 + self.gamma_occlusion * depth_norm * (
            1.0 - torch.clamp(mean_transmittance, 0.0, 1.0)
        )

        demand = grad_accum * torch.clamp(visibility_count, min=1).float() * occlusion_weight

        gain_clone = demand / max(1.0, float(cost_clone_bytes))
        gain_split = demand / max(1.0, float(cost_split_bytes))

        return demand, gain_clone, gain_split

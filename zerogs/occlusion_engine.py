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
        self.grad_accum = torch.zeros(num_gaussians, dtype=torch.float32, device=self.device)

    def update_per_view_stats(
        self,
        visible_indices: torch.Tensor,
        view_depths: torch.Tensor,
        view_transmittance: torch.Tensor,
    ) -> None:
        """
        Updates statistics after a camera forward pass.

        Args:
            visible_indices: 1D Tensor of indices of Gaussians visible in this camera view.
            view_depths: 1D Tensor of depth values in camera coordinate space (z > 0).
            view_transmittance: 1D Tensor of accumulated transmittances T.
        """
        if visible_indices is None or len(visible_indices) == 0:
            return

        idx = visible_indices.to(self.device).long()
        depths = view_depths.to(self.device).float()
        T = view_transmittance.to(self.device).float()

        self.accum_vis[idx] += 1
        self.accum_T[idx] += T
        self.accum_depth[idx] += depths
        self.accum_depth_sq[idx] += depths * depths

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
        vis_counts = torch.clamp(self.accum_vis, min=1).float()

        # Mean transmittance and mean depth
        mean_T = self.accum_T / vis_counts
        mean_depth = self.accum_depth / vis_counts
        mean_depth_sq = self.accum_depth_sq / vis_counts

        # Depth variance: Var(z) = E[z^2] - (E[z])^2
        depth_var = torch.clamp(mean_depth_sq - (mean_depth**2), min=0.0)
        depth_std = torch.sqrt(depth_var + self.depth_epsilon)

        # Occlusion modulation factor Omega_i:
        # High depth variance (boundary across cameras) + Low transmittance (surface/dense)
        # -> High occlusion sensitivity
        normalized_depth_fluctuation = depth_std / (float(scene_extent) + self.depth_epsilon)
        occlusion_factor = 1.0 + self.gamma_occ * normalized_depth_fluctuation * (1.0 - mean_T)

        # Baseline demand: Viewspace gradient * Visibility count
        if self.grad_accum is None:
            raw_demand = vis_counts
        else:
            raw_demand = self.grad_accum * vis_counts

        compensated_demand = raw_demand * occlusion_factor

        # Protected structure mask:
        # Gaussians with significant depth variance (thin structures visible from distinct angles)
        # and low transmittance should NOT be eliminated by bottom-quantile pruning!
        is_high_depth_var = depth_std > (0.01 * float(scene_extent))
        is_occluding = mean_T < self.transmittance_tau
        protected_mask = torch.logical_and(is_high_depth_var, is_occluding)

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

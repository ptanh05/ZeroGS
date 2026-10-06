"""
Byte-Level Transient Memory Cost Model (Module 1 - Solving Gaps P1 & S3).

Accurately models the analytical and empirical transient VRAM footprint during
Gaussian mutation (Clone vs Split), taking into account:
- Raw Gaussian parameter tensors (xyz, rotation, scale, opacity, SH).
- Backward gradient buffers.
- Adam/FusedAdam optimizer states (first & second moments: exp_avg, exp_avg_sq).
- PyTorch Caching Allocator fragmentation and temporary concatenation buffers.
"""

from typing import Dict, Any, Optional
import torch


class ByteCostModel:
    """
    Byte-Level Transient Cost Model for 3D Gaussian Splatting.

    Solves Gap P1 (Transient Peak disparity between Clone vs Split)
    and Gap S3 (Byte-level GPU footprint modeling instead of crude primitive counts).
    """

    def __init__(
        self,
        sh_degree: int = 3,
        use_adam: bool = True,
        bytes_per_float: int = 4,
        clone_multiplier: float = 1.15,
        split_multiplier: float = 2.30,
        cuda_padding_bytes: int = 1024 * 1024,  # 1MB CUDA block padding
    ):
        """
        Args:
            sh_degree: Spherical Harmonics degree (0 to 3).
            use_adam: Whether Adam/AdamW optimizer states are tracked.
            bytes_per_float: Byte size per float (4 for float32, 2 for fp16).
            clone_multiplier: Empirical transient multiplier for clone (cat overhead + fragmentation).
            split_multiplier: Empirical transient multiplier for split (2x children + parent retention + Adam rebuild).
            cuda_padding_bytes: Safety padding for allocator block alignment (bytes).
        """
        self.sh_degree = sh_degree
        self.use_adam = use_adam
        self.bytes_per_float = bytes_per_float
        self.clone_multiplier = clone_multiplier
        self.split_multiplier = split_multiplier
        self.cuda_padding_bytes = cuda_padding_bytes

        self.recalculate_constants()

    def recalculate_constants(self) -> None:
        """Computes primitive parameter dimensions and byte footprints."""
        # 3 (xyz) + 4 (rotation) + 3 (scale) + 1 (opacity) = 11 floats
        self.base_floats_per_gaussian = 11
        # SH coefficients: 3 channels * (sh_degree + 1)^2
        self.sh_floats_per_gaussian = 3 * ((self.sh_degree + 1) ** 2)
        self.total_floats_per_gaussian = (
            self.base_floats_per_gaussian + self.sh_floats_per_gaussian
        )

        # Raw model parameter bytes per Gaussian
        self.param_bytes_per_gaussian = (
            self.total_floats_per_gaussian * self.bytes_per_float
        )

        # Gradient tensor bytes per Gaussian
        self.grad_bytes_per_gaussian = self.param_bytes_per_gaussian

        # Adam optimizer states: exp_avg (moment 1) + exp_avg_sq (moment 2)
        if self.use_adam:
            self.adam_states_bytes_per_gaussian = 2 * self.param_bytes_per_gaussian
        else:
            self.adam_states_bytes_per_gaussian = 0

        # Total persistent bytes per Gaussian on GPU during training
        # Parameters + Gradients + Optimizer states
        self.persistent_bytes_per_gaussian = (
            self.param_bytes_per_gaussian
            + self.grad_bytes_per_gaussian
            + self.adam_states_bytes_per_gaussian
        )

    def estimate_persistent_bytes(self, num_gaussians: int) -> int:
        """Total persistent memory in bytes for a given number of Gaussians."""
        return int(num_gaussians * self.persistent_bytes_per_gaussian)

    def estimate_transient_bytes(
        self,
        n_clone: int,
        n_split: int,
        include_padding: bool = True,
    ) -> int:
        """
        Computes the peak transient memory spike Delta M_transient in bytes.

        Delta M_transient = Delta M_clone(N_clone) + Delta M_split(N_split) + Padding_CUDA

        Clone footprint:
            N_clone * persistent_bytes * clone_multiplier
        Split footprint:
            N_split * persistent_bytes * split_multiplier
        """
        if n_clone < 0 or n_split < 0:
            raise ValueError(f"Negative primitive counts received: clone={n_clone}, split={n_split}")

        clone_bytes = n_clone * self.persistent_bytes_per_gaussian * self.clone_multiplier
        split_bytes = n_split * self.persistent_bytes_per_gaussian * self.split_multiplier
        total = clone_bytes + split_bytes

        if include_padding and (n_clone > 0 or n_split > 0):
            total += self.cuda_padding_bytes

        return int(round(total))

    def estimate_prune_freed_bytes(self, n_prune: int) -> int:
        """
        Computes the persistent memory freed when pruning n_prune Gaussians.
        When points are pruned, parameter tensors, gradient tensors, and optimizer
        states are shrunk.
        """
        if n_prune <= 0:
            return 0
        return int(n_prune * self.persistent_bytes_per_gaussian)

    def fit_empirical_multipliers(
        self,
        measured_clone_spike_bytes: float,
        measured_split_spike_bytes: float,
        delta_n_clone: int,
        delta_n_split: int,
    ) -> None:
        """
        Calibrates the clone and split multipliers from real hardware profiling (Experiment 0).
        """
        if delta_n_clone > 0:
            unit_bytes = delta_n_clone * self.persistent_bytes_per_gaussian
            self.clone_multiplier = max(1.0, float(measured_clone_spike_bytes / unit_bytes))
        if delta_n_split > 0:
            unit_bytes = delta_n_split * self.persistent_bytes_per_gaussian
            self.split_multiplier = max(1.5, float(measured_split_spike_bytes / unit_bytes))

    def get_breakdown(self, num_gaussians: int) -> Dict[str, Any]:
        """Returns human-readable memory breakdown for inspection/logging."""
        param_mb = (num_gaussians * self.param_bytes_per_gaussian) / (1024**2)
        grad_mb = (num_gaussians * self.grad_bytes_per_gaussian) / (1024**2)
        adam_mb = (num_gaussians * self.adam_states_bytes_per_gaussian) / (1024**2)
        total_mb = (num_gaussians * self.persistent_bytes_per_gaussian) / (1024**2)

        return {
            "num_gaussians": num_gaussians,
            "sh_degree": self.sh_degree,
            "floats_per_gaussian": self.total_floats_per_gaussian,
            "bytes_per_gaussian_params": self.param_bytes_per_gaussian,
            "bytes_per_gaussian_persistent": self.persistent_bytes_per_gaussian,
            "total_params_mb": round(param_mb, 2),
            "total_grads_mb": round(grad_mb, 2),
            "total_adam_states_mb": round(adam_mb, 2),
            "total_persistent_mb": round(total_mb, 2),
            "clone_multiplier": self.clone_multiplier,
            "split_multiplier": self.split_multiplier,
        }

    def __repr__(self) -> str:
        return (
            f"ByteCostModel(sh={self.sh_degree}, "
            f"params={self.param_bytes_per_gaussian}B/pt, "
            f"persistent={self.persistent_bytes_per_gaussian}B/pt, "
            f"clone_mul={self.clone_multiplier:.2f}, "
            f"split_mul={self.split_multiplier:.2f})"
        )

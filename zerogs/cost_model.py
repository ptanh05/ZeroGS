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
import json
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
        clone_fixed_overhead_bytes: int = 0,
        split_fixed_overhead_bytes: int = 0,
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
        self.clone_fixed_overhead_bytes = int(clone_fixed_overhead_bytes)
        self.split_fixed_overhead_bytes = int(split_fixed_overhead_bytes)

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
        if n_clone > 0:
            clone_bytes += self.clone_fixed_overhead_bytes
        if n_split > 0:
            split_bytes += self.split_fixed_overhead_bytes
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

    @staticmethod
    def _fit_affine_cost(counts, measured_bytes):
        """Least-squares fit y = intercept + slope*x without extra dependencies."""
        if len(counts) != len(measured_bytes) or len(counts) < 2:
            raise ValueError("At least two matched profiling samples are required")
        x_mean = sum(float(x) for x in counts) / len(counts)
        y_mean = sum(float(y) for y in measured_bytes) / len(measured_bytes)
        denominator = sum((float(x) - x_mean) ** 2 for x in counts)
        if denominator <= 0:
            raise ValueError("Profiling samples must use at least two mutation sizes")
        slope = sum(
            (float(x) - x_mean) * (float(y) - y_mean)
            for x, y in zip(counts, measured_bytes)
        ) / denominator
        intercept = y_mean - slope * x_mean
        return max(0.0, intercept), max(0.0, slope)

    def fit_profile_results(self, results) -> Dict[str, float]:
        """Fit per-operation fixed overhead and per-Gaussian multipliers."""
        counts = [int(row["delta_n"]) for row in results]
        clone_bytes = [
            max(0.0, float(row["clone_spike_mb"]) * 1024**2 - self.cuda_padding_bytes)
            for row in results
        ]
        split_bytes = [
            max(0.0, float(row["split_spike_mb"]) * 1024**2 - self.cuda_padding_bytes)
            for row in results
        ]
        clone_intercept, clone_slope = self._fit_affine_cost(counts, clone_bytes)
        split_intercept, split_slope = self._fit_affine_cost(counts, split_bytes)
        self.clone_fixed_overhead_bytes = int(round(clone_intercept))
        self.split_fixed_overhead_bytes = int(round(split_intercept))
        self.clone_multiplier = clone_slope / self.persistent_bytes_per_gaussian
        self.split_multiplier = split_slope / self.persistent_bytes_per_gaussian
        return {
            "clone_multiplier": self.clone_multiplier,
            "split_multiplier": self.split_multiplier,
            "clone_fixed_overhead_bytes": self.clone_fixed_overhead_bytes,
            "split_fixed_overhead_bytes": self.split_fixed_overhead_bytes,
        }

    def profile_error(self, results) -> Dict[str, float]:
        """Return mean absolute percentage error against measured profile rows."""
        clone_errors = []
        split_errors = []
        for row in results:
            count = int(row["delta_n"])
            measured_clone = float(row["clone_spike_mb"]) * 1024**2
            measured_split = float(row["split_spike_mb"]) * 1024**2
            predicted_clone = self.estimate_transient_bytes(count, 0)
            predicted_split = self.estimate_transient_bytes(0, count)
            clone_errors.append(abs(predicted_clone - measured_clone) / max(1.0, measured_clone))
            split_errors.append(abs(predicted_split - measured_split) / max(1.0, measured_split))
        return {
            "clone_mape_pct": 100.0 * sum(clone_errors) / max(1, len(clone_errors)),
            "split_mape_pct": 100.0 * sum(split_errors) / max(1, len(split_errors)),
            "combined_mape_pct": 100.0 * (
                sum(clone_errors) + sum(split_errors)
            ) / max(1, len(clone_errors) + len(split_errors)),
        }

    def calibration_dict(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "sh_degree": self.sh_degree,
            "bytes_per_float": self.bytes_per_float,
            "use_adam": self.use_adam,
            "clone_multiplier": self.clone_multiplier,
            "split_multiplier": self.split_multiplier,
            "clone_fixed_overhead_bytes": self.clone_fixed_overhead_bytes,
            "split_fixed_overhead_bytes": self.split_fixed_overhead_bytes,
            "cuda_padding_bytes": self.cuda_padding_bytes,
            "metadata": metadata or {},
        }

    @classmethod
    def from_calibration_file(cls, path: str) -> "ByteCostModel":
        with open(path, encoding="utf-8") as source:
            data = json.load(source)
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported cost calibration schema")
        return cls(
            sh_degree=int(data["sh_degree"]),
            bytes_per_float=int(data["bytes_per_float"]),
            use_adam=bool(data["use_adam"]),
            clone_multiplier=float(data["clone_multiplier"]),
            split_multiplier=float(data["split_multiplier"]),
            clone_fixed_overhead_bytes=int(data["clone_fixed_overhead_bytes"]),
            split_fixed_overhead_bytes=int(data["split_fixed_overhead_bytes"]),
            cuda_padding_bytes=int(data["cuda_padding_bytes"]),
        )

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
            "clone_fixed_overhead_mb": round(self.clone_fixed_overhead_bytes / (1024**2), 3),
            "split_fixed_overhead_mb": round(self.split_fixed_overhead_bytes / (1024**2), 3),
        }

    def __repr__(self) -> str:
        return (
            f"ByteCostModel(sh={self.sh_degree}, "
            f"params={self.param_bytes_per_gaussian}B/pt, "
            f"persistent={self.persistent_bytes_per_gaussian}B/pt, "
            f"clone_mul={self.clone_multiplier:.2f}, "
            f"split_mul={self.split_multiplier:.2f})"
        )

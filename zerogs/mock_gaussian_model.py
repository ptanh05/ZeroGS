"""
Mock Gaussian Model and Synthetic Scene for Testing and Isolated Experiments.

Implements the exact interface and parameter shapes of 3DGS (Kerbl et al., SIGGRAPH 2023),
including Adam optimizer states, densify_and_clone, densify_and_split, and prune_points.
Allows rigorous profiling of transient memory spikes on actual CUDA hardware
without external dataset or rasterizer dependencies.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
from torch import nn


class MockGaussianModel:
    """
    Simulates the exact tensor layout and mutation behaviors of official 3DGS GaussianModel.
    """

    def __init__(self, sh_degree: int = 3, num_points: int = 10000, device: str = "cuda"):
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Parameter dimensions
        self.num_points = num_points
        self.sh_floats = 3 * ((sh_degree + 1) ** 2)

        # 3DGS Parameter Tensors
        self._xyz = torch.randn(num_points, 3, device=self.device, requires_grad=True)
        self._rotation = torch.randn(num_points, 4, device=self.device, requires_grad=True)
        self._scaling = torch.randn(num_points, 3, device=self.device, requires_grad=True)
        self._opacity = torch.randn(num_points, 1, device=self.device, requires_grad=True)
        self._features_dc = torch.randn(num_points, 1, 3, device=self.device, requires_grad=True)
        self._features_rest = torch.randn(
            num_points, (self.sh_floats // 3) - 1, 3, device=self.device, requires_grad=True
        )

        # 2D Screen Radii and Gradient Tracking
        self.max_radii2D = torch.zeros(num_points, device=self.device)
        self.xyz_gradient_accum = torch.zeros(num_points, 1, device=self.device)
        self.denom = torch.ones(num_points, 1, device=self.device)

        # Setup Optimizer with Adam states
        self.setup_optimizer()

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def setup_optimizer(self) -> None:
        """Initializes Adam optimizer and populates first and second moments."""
        params = [
            {"params": [self._xyz], "name": "xyz"},
            {"params": [self._rotation], "name": "rotation"},
            {"params": [self._scaling], "name": "scaling"},
            {"params": [self._opacity], "name": "opacity"},
            {"params": [self._features_dc], "name": "features_dc"},
            {"params": [self._features_rest], "name": "features_rest"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=1e-3, eps=1e-15)

        # Populate optimizer states for all parameters so memory footprint matches real training
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                state["step"] = torch.tensor(1.0, device=self.device)
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def replace_tensor_to_optimizer(
        self, tensor: torch.Tensor, name: str
    ) -> Dict[str, torch.Tensor]:
        """Official 3DGS interface: replaces parameter tensor and transfers/resizes its Adam states."""
        optimizable_tensors: Dict[str, torch.Tensor] = {}
        if self.optimizer is None:
            group_param = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[name] = group_param
            return optimizable_tensors

        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = {
                        "step": torch.tensor(1.0, device=self.device),
                        "exp_avg": torch.zeros_like(tensor),
                        "exp_avg_sq": torch.zeros_like(tensor),
                    }
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Replaces a parameter tensor and transfers/resizes its Adam states."""
        optimizable_tensors = self.replace_tensor_to_optimizer(tensor, name)
        return optimizable_tensors[name]

    def prune_points(self, mask: torch.Tensor) -> None:
        """Prunes points where mask is True, reducing tensor sizes in optimizer."""
        valid_points_mask = ~mask
        current_n = self._xyz.shape[0]
        retained_n = int(valid_points_mask.sum().item())

        self._xyz = self._xyz[valid_points_mask].detach().requires_grad_(True)
        self._rotation = self._rotation[valid_points_mask].detach().requires_grad_(True)
        self._scaling = self._scaling[valid_points_mask].detach().requires_grad_(True)
        self._opacity = self._opacity[valid_points_mask].detach().requires_grad_(True)
        self._features_dc = self._features_dc[valid_points_mask].detach().requires_grad_(True)
        self._features_rest = self._features_rest[valid_points_mask].detach().requires_grad_(True)

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        # Rebuild optimizer states for surviving points
        for group in self.optimizer.param_groups:
            old_p = group["params"][0]
            old_state = self.optimizer.state.get(old_p, {})
            name = group["name"]

            # Map to newly sliced tensor
            new_tensor = getattr(self, f"_{name}")
            new_p = nn.Parameter(new_tensor.detach().requires_grad_(True))
            group["params"][0] = new_p

            del self.optimizer.state[old_p]
            new_state = {"step": old_state.get("step", torch.tensor(1.0, device=self.device))}
            if "exp_avg" in old_state:
                new_state["exp_avg"] = old_state["exp_avg"][valid_points_mask]
            else:
                new_state["exp_avg"] = torch.zeros_like(new_p)
            if "exp_avg_sq" in old_state:
                new_state["exp_avg_sq"] = old_state["exp_avg_sq"][valid_points_mask]
            else:
                new_state["exp_avg_sq"] = torch.zeros_like(new_p)
            self.optimizer.state[new_p] = new_state

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
    ) -> None:
        """
        Duplicates points matching clone mask.
        In vanilla 3DGS, this concatenates newly cloned points onto existing tensors.
        """
        if isinstance(grads, torch.Tensor) and grads.dtype == torch.bool:
            if grads.shape[0] < self._xyz.shape[0]:
                pad = torch.zeros(self._xyz.shape[0] - grads.shape[0], dtype=torch.bool, device=self.device)
                selected_pts_mask = torch.cat([grads, pad])
            else:
                selected_pts_mask = grads
        else:
            selected_pts_mask = grads >= grad_threshold

        n_new = int(selected_pts_mask.sum().item())
        if n_new == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        # Concatenate parameters
        self._xyz = torch.cat([self._xyz, new_xyz], dim=0).detach().requires_grad_(True)
        self._rotation = torch.cat([self._rotation, new_rotation], dim=0).detach().requires_grad_(True)
        self._scaling = torch.cat([self._scaling, new_scaling], dim=0).detach().requires_grad_(True)
        self._opacity = torch.cat([self._opacity, new_opacity], dim=0).detach().requires_grad_(True)
        self._features_dc = torch.cat([self._features_dc, new_features_dc], dim=0).detach().requires_grad_(True)
        self._features_rest = torch.cat([self._features_rest, new_features_rest], dim=0).detach().requires_grad_(True)

        # Concatenate auxiliary tracking
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(n_new, device=self.device)])
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros(n_new, 1, device=self.device)]
        )
        self.denom = torch.cat([self.denom, torch.ones(n_new, 1, device=self.device)])

        # Extend optimizer states with zeros for newly cloned points
        for group in self.optimizer.param_groups:
            old_p = group["params"][0]
            name = group["name"]
            new_p = nn.Parameter(getattr(self, f"_{name}").detach().requires_grad_(True))
            group["params"][0] = new_p

            old_state = self.optimizer.state.get(old_p, {})
            del self.optimizer.state[old_p]

            new_state = {"step": old_state.get("step", torch.tensor(1.0, device=self.device))}
            added_pts = getattr(self, f"_{name}")[-n_new:]
            if "exp_avg" in old_state:
                new_state["exp_avg"] = torch.cat(
                    [old_state["exp_avg"], torch.zeros_like(added_pts)], dim=0
                )
            else:
                new_state["exp_avg"] = torch.zeros_like(new_p)

            if "exp_avg_sq" in old_state:
                new_state["exp_avg_sq"] = torch.cat(
                    [old_state["exp_avg_sq"], torch.zeros_like(added_pts)], dim=0
                )
            else:
                new_state["exp_avg_sq"] = torch.zeros_like(new_p)

            self.optimizer.state[new_p] = new_state

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
    ) -> None:
        """
        Splits high-gradient points into N child points (default 2), then prunes the parent points.
        Crucially replicates the transient peak spike of 3DGS!
        """
        if isinstance(grads, torch.Tensor) and grads.dtype == torch.bool:
            if grads.shape[0] < self._xyz.shape[0]:
                pad = torch.zeros(self._xyz.shape[0] - grads.shape[0], dtype=torch.bool, device=self.device)
                selected_pts_mask = torch.cat([grads, pad])
            else:
                selected_pts_mask = grads
        else:
            selected_pts_mask = grads >= grad_threshold

        n_split = int(selected_pts_mask.sum().item())
        if n_split == 0:
            return

        # 1. Generate 2 child points per split candidate
        stds = torch.exp(self._scaling[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = self._rotation[selected_pts_mask].repeat(N, 1)

        new_xyz = torch.bmm(
            torch.eye(3, device=self.device).repeat(samples.size(0), 1, 1),
            samples.unsqueeze(-1),
        ).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)

        # Rescale children by 1.6
        new_scaling = torch.log(torch.exp(self._scaling[selected_pts_mask]).repeat(N, 1) / 1.6)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)

        # 2. Concatenate children onto the model (TRANSIENT PEAK: holds parents + 2x children)
        self._xyz = torch.cat([self._xyz, new_xyz], dim=0)
        self._rotation = torch.cat([self._rotation, new_rotation], dim=0)
        self._scaling = torch.cat([self._scaling, new_scaling], dim=0)
        self._opacity = torch.cat([self._opacity, new_opacity], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_features_rest], dim=0)

        n_children = N * n_split
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros(n_children, device=self.device)]
        )
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros(n_children, 1, device=self.device)]
        )
        self.denom = torch.cat([self.denom, torch.ones(n_children, 1, device=self.device)])

        # Extend optimizer states for all children
        for group in self.optimizer.param_groups:
            old_p = group["params"][0]
            name = group["name"]
            new_p = nn.Parameter(getattr(self, f"_{name}").detach().requires_grad_(True))
            group["params"][0] = new_p

            old_state = self.optimizer.state.get(old_p, {})
            del self.optimizer.state[old_p]

            new_state = {"step": old_state.get("step", torch.tensor(1.0, device=self.device))}
            added_pts = getattr(self, f"_{name}")[-n_children:]
            if "exp_avg" in old_state:
                new_state["exp_avg"] = torch.cat(
                    [old_state["exp_avg"], torch.zeros_like(added_pts)], dim=0
                )
            else:
                new_state["exp_avg"] = torch.zeros_like(new_p)

            if "exp_avg_sq" in old_state:
                new_state["exp_avg_sq"] = torch.cat(
                    [old_state["exp_avg_sq"], torch.zeros_like(added_pts)], dim=0
                )
            else:
                new_state["exp_avg_sq"] = torch.zeros_like(new_p)

            self.optimizer.state[new_p] = new_state

        # 3. Prune original parent points (now at indices selected_pts_mask)
        prune_filter = torch.cat(
            [selected_pts_mask, torch.zeros(n_children, dtype=torch.bool, device=self.device)]
        )
        self.prune_points(prune_filter)

    def add_densification_stats(
        self, viewspace_points: torch.Tensor, visibility_filter: torch.Tensor
    ) -> None:
        if visibility_filter is not None and viewspace_points.grad is not None:
            self.xyz_gradient_accum[visibility_filter] += torch.norm(
                viewspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True
            )
            self.denom[visibility_filter] += 1

    def update_learning_rate(self, iteration: int) -> None:
        pass

    def oneupSHdegree(self) -> None:
        if self.sh_degree < self.max_sh_degree:
            self.sh_degree += 1

    def reset_opacity(self) -> None:
        with torch.no_grad():
            self._opacity.fill_(-2.0)  # sigmoid(-2.0) ~ 0.119

    def capture(self) -> Dict[str, Any]:
        return {
            "xyz": self._xyz.data,
            "scaling": self._scaling.data,
            "rotation": self._rotation.data,
            "opacity": self._opacity.data,
            "features_dc": self._features_dc.data,
            "features_rest": self._features_rest.data,
        }

    def restore(self, model_params: Dict[str, Any], opt: Any) -> None:
        self._xyz.data = model_params["xyz"]
        self._scaling.data = model_params["scaling"]
        self._rotation.data = model_params["rotation"]
        self._opacity.data = model_params["opacity"]
        self._features_dc.data = model_params["features_dc"]
        self._features_rest.data = model_params["features_rest"]


class MockCamera:
    """Simulates a camera view with world-view transform matrix."""

    def __init__(self, camera_id: int = 0, device: str = "cuda"):
        self.camera_id = camera_id
        dev = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        # 4x4 world_view_transform
        W = torch.eye(4, device=dev)
        W[2, 3] = 2.0  # translate depth
        self.world_view_transform = W
        self.image_width = 800
        self.image_height = 800
        self.original_image = torch.zeros(3, 800, 800, device=dev)


def mock_render(
    camera: Any,
    gaussians: Any,
    pipe: Any = None,
    background: Any = None,
) -> Dict[str, Any]:
    """
    Simulates the forward render pass, returning dummy image and physical stats:
    - viewspace_points
    - visibility_filter
    - radii
    - mean_T (accumulated transmittance)
    - depth_var (depth variance)
    - vis_count
    """
    dev = getattr(gaussians, "device", None)
    if dev is None:
        try:
            dev = gaussians.get_xyz.device
        except Exception:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = gaussians.get_xyz.shape[0]

    # Random visibility filter (~80% visible)
    visibility_filter = torch.rand(N, device=dev) > 0.2
    vis_count = visibility_filter.int()

    radii = torch.full((N,), 5.0, device=dev)
    mean_T = torch.rand(N, device=dev) * 0.8 + 0.1  # in [0.1, 0.9]
    depth_var = torch.rand(N, device=dev) * 0.05

    # Viewspace points tensor requiring grad
    viewspace_points = torch.zeros(N, 3, device=dev, requires_grad=True)
    # Give it dummy gradient
    viewspace_points.grad = torch.randn(N, 3, device=dev) * 0.001

    # Connect render_image to parameters so loss.backward() flows gradients
    dummy_loss_link = (gaussians.get_xyz[:1].sum() * 0.0)
    render_image = dummy_loss_link + torch.zeros(3, camera.image_height, camera.image_width, device=dev)

    return {
        "render": render_image,
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "mean_T": mean_T,
        "mean_transmittance": mean_T,
        "depth_var": depth_var,
        "depth_variance": depth_var,
        "vis_count": vis_count,
    }

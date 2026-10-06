"""
Learned Opacity Field Network (Section 3 of spec).

Replaces the static per-Gaussian opacity scalar with a view-dependent opacity
predicted by a small MLP:

    opacity = fθ(features, view_direction, depth)

Input (per Gaussian):
    - Gaussian embedding (xyz, scale, base_opacity) → 7-dim
    - View direction (4-dim, quaternion of direction vector)
    - Distance to camera (1-dim, normalised depth)

Output:
    - opacity ∈ [0, 1] via Sigmoid

Integration point:
    Called from gaussian_renderer/__init__.py::render() when the
    ``use_opacity_network`` flag is set.  Returns a modified opacity tensor
    that replaces ``pc.get_opacity`` in the rasterization call.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OpacityFieldNetwork(nn.Module):
    """View-dependent opacity prediction network.

    Architecture: 12 → 64 → 32 → 1 (Sigmoid)

    Input (12-dim):
        3 (xyz) + 3 (scale) + 1 (base_opacity) + 4 (view_dir) + 1 (depth)

    See :meth:`forward` for details.
    """

    def __init__(
        self,
        feat_dim: int = 7,       # xyz + scale + base_opacity
        view_dim: int = 4,       # view direction
        depth_dim: int = 1,      # distance to camera
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
    ):
        super().__init__()
        input_dim = feat_dim + view_dim + depth_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_2, 1),
        )

        # Small initialisation so output starts near the base opacity
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        base_opacity: torch.Tensor,
        view_dirs: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Predict view-dependent opacity for all Gaussians.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape (N, 3) Gaussian positions.
        scales : torch.Tensor
            Shape (N, 3) scaling factors (already activated, i.e. exp(_scaling)).
        base_opacity : torch.Tensor
            Shape (N, 1) base opacity (already sigmoid-activated).
        view_dirs : torch.Tensor
            Shape (N, 4) view direction unit vectors [x, y, z, w] (normalised).
        depths : torch.Tensor
            Shape (N, 1) depth (distance from camera center in world units,
            normalised to [0, 1] range).

        Returns
        -------
        torch.Tensor
            Shape (N, 1) modified opacity values ∈ [0, 1].
        """
        # Feature encoding: xyz + scales + base_opacity
        gauss_feat = torch.cat(
            [xyz, scales, base_opacity], dim=-1
        )  # (N, 3+3+1=7)

        # Concatenate everything
        inp = torch.cat(
            [gauss_feat, view_dirs, depths], dim=-1
        )  # (N, 7+4+1=12)

        delta = self.net(inp)  # (N, 1) small additive adjustment
        # Constrain to reasonable range so opacity stays [0, 1]
        modified_opacity = torch.sigmoid(delta + 2.0 * (base_opacity - 0.5))
        return modified_opacity


# --------------------------------------------------------------------------- #
#  Direction / depth helpers
# --------------------------------------------------------------------------- #


def compute_view_directions(
    means3D: torch.Tensor, camera_center: torch.Tensor
) -> torch.Tensor:
    """Compute normalised 4D view direction vectors.

    The direction from each Gaussian to the camera center is encoded as a
    unit quaternion-like 4-vector: [dx, dy, dz, 1.0] normalised.  We keep
    the 4D form for compatibility with the MLP input size.

    Parameters
    ----------
    means3D : torch.Tensor
        Shape (N, 3) Gaussian positions.
    camera_center : torch.Tensor
        Shape (3,) camera position.

    Returns
    -------
    torch.Tensor
        Shape (N, 4) normalised direction vectors.
    """
    dirs = camera_center[None, :] - means3D  # (N, 3)
    norm = dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dirs = dirs / norm
    # Pad with 1.0 as homogeneous
    return torch.cat([dirs, torch.ones_like(dirs[:, :1])], dim=-1)


def compute_depths(
    means3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> torch.Tensor:
    """Compute normalised depth in camera space.

    Transforms Gaussian positions to camera coordinates and normalises
    the z-coordinate to [0, 1] using znear/zfar.

    Parameters
    ----------
    means3D : torch.Tensor
        Shape (N, 3) Gaussian positions.
    viewmatrix : torch.Tensor
        Shape (4, 4) world-to-camera matrix.
    znear : float
        Near clipping plane.
    zfar : float
        Far clipping plane.

    Returns
    -------
    torch.Tensor
        Shape (N, 1) normalised depths in [0, 1].
    """
    # Transform to camera space: we need z_cam
    ones = torch.ones_like(means3D[:, :1])  # (N, 1)
    means_h = torch.cat([means3D, ones], dim=-1)  # (N, 4)
    # viewmatrix is column-major (world_view_transform), so multiply as
    #   cam = viewmatrix^T @ means_h
    # or equivalently cam_coords = means_h @ viewmatrix
    # (both are the same due to transpose in Camera.__init__)
    cam_coords = means_h @ viewmatrix  # (N, 4)
    z = cam_coords[:, 2:3]  # (N, 1)
    # Clamp and normalise
    z = z.clamp(min=znear, max=zfar)
    depth_norm = (z - znear) / (zfar - znear)
    return depth_norm


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    """Compute opacity regularisation loss (encourage sparsity).

    Penalises intermediate opacities — pushes opacity toward 0 or 1.

    Parameters
    ----------
    opacity : torch.Tensor
        Shape (N, 1) opacity values.

    Returns
    -------
    torch.Tensor
        Scalar regularisation loss.
    """
    return (opacity * (1 - opacity)).mean()

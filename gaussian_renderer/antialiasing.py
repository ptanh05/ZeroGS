"""
Differentiable Anti-Aliasing Renderer (Section 4 of spec).

Implements a Python-level approximation of pixel-integrated Gaussian footprint
(mip-aware Gaussian splatting) that runs on top of the existing CUDA rasterizer
without modifying CUDA code.

Core idea from "Mip-Splatting / Anti-aliased 3D Gaussian Splatting":
    Instead of   pixel += Σ G(x)
    Approximate: pixel += ∫_pixel Gaussian(x) dx

Here we implement the **footprint scaling** approach:
    1. Compute the screen-space elliptical footprint of each Gaussian.
    2. If the footprint is smaller than ~1 pixel, the Gaussian is
       "undersampled" — scale its opacity proportionally to avoid aliasing.
    3. If the footprint is much larger than 1 pixel, blend normally.

This is done entirely at the Python tensor level before the CUDA rasterizer call.
"""

from __future__ import annotations

import math
import torch


def compute_mip_level(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tanfovx: float,
    tanfovy: float,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
) -> tuple:
    """Compute screen-space Gaussian footprint and mip levels.

    For each Gaussian, estimate the number of pixels it covers in screen space
    by projecting its 3D covariance and computing the ellipse area.

    Parameters
    ----------
    means3D : torch.Tensor
        Shape (N, 3) Gaussian positions.
    scales : torch.Tensor
        Shape (N, 3) Gaussian scales (activated, i.e. exp(_scaling)).
    rotations : torch.Tensor
        Shape (N, 4) Gaussian rotations (quaternions, normalised).
    viewmatrix : torch.Tensor
        Shape (4, 4) world-to-camera matrix.
    projmatrix : torch.Tensor
        Shape (4, 4) projection matrix.
    tanfovx, tanfovy : float
        Half FoV tangents.
    image_width, image_height : int
        Image dimensions.
    fx, fy : float
        Focal lengths in pixels.

    Returns
    -------
    radii : torch.Tensor
        Shape (N,) screen-space radius in pixels.
    mip_levels : torch.Tensor
        Shape (N,) continuous mip level (0 = full res, >0 = downsampled).
    pixel_coverages : torch.Tensor
        Shape (N,) fraction of pixel covered by each Gaussian.
    """
    device = means3D.device
    N = means3D.shape[0]

    # --- Build 3D covariance from scales + rotations ---
    # Rotation matrix from quaternion
    R = _quat_to_rotmat(rotations)  # (N, 3, 3)
    # Scale matrix
    S = torch.zeros(N, 3, 3, device=device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    # Cov = R @ S^2 @ R^T (no need to square S since we'll use sqrt later)
    # Actually Gaussians use the full 3D covariance matrix = R @ S @ S @ R^T
    cov3D = R @ S.pow(2) @ R.transpose(1, 2)  # (N, 3, 3)

    # --- Project to screen space ---
    # Camera-space position: x_cam = viewmatrix[:3] @ [xyz, 1]
    ones = torch.ones(N, 1, device=device)
    means_h = torch.cat([means3D, ones], dim=-1)  # (N, 4)
    cam_coords = (viewmatrix @ means_h.unsqueeze(-1)).squeeze(-1)  # (N, 4)

    # Jacobian of perspective projection at the Gaussian center
    # d(u,v)/d(x_cam) for projection
    z = cam_coords[:, 2].clamp(min=1e-4)  # (N,)
    x = cam_coords[:, 0] / z
    y = cam_coords[:, 1] / z

    # Screen-space pixel coordinates
    # For a pinhole camera: u = fx * x / z + cx, v = fy * y / z + cy
    # The Jacobian d(u,v)/d(x_cam, y_cam, z_cam) at x=y=0:
    #   [fx/z, 0,  -fx*x/z^2]
    #   [0,   fy/z, -fy*y/z^2]
    # But x/z^2 and y/z^2 vanish at Gaussian center (we use x=0,y=0 approx)
    # Full Jacobian for the affine approximation:
    # Actually, using the full GS Jacobian:
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * cam_coords[:, 0] / (z ** 2)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * cam_coords[:, 1] / (z ** 2)

    # Screen-space covariance: J @ cov3D @ J^T
    cov2D = J @ cov3D @ J.transpose(1, 2)  # (N, 2, 2)

    # --- Compute ellipse area in pixels ---
    # Area of ellipse = π * sqrt(det(cov2D))
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    det = det.clamp(min=1e-8)
    ellipse_area = math.pi * torch.sqrt(det)  # (N,)

    # Major axis radius (approximate as sqrt of largest eigenvalue)
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    disc = torch.sqrt((trace ** 2 - 4 * det).clamp(min=0.0))
    lambda_max = (trace + disc) / 2
    radii = torch.sqrt(lambda_max)  # screen-space radius in pixels

    # Pixel coverage: fraction of a pixel covered (capped at 1.0)
    pixel_coverage = torch.clamp(ellipse_area / 1.0, max=1.0)

    # Mip level: log2 of the radius (0 = one pixel, 1 = two pixels, etc.)
    mip_levels = torch.log2(radii.clamp(min=1.0))

    return radii, mip_levels, pixel_coverage


def apply_antialiasing(
    opacities: torch.Tensor,
    radii_2d: torch.Tensor,
    pixel_coverages: torch.Tensor,
    scaling_modifier: float = 1.0,
    strength: float = 1.0,
) -> tuple:
    """Apply mip-aware anti-aliasing by scaling under-sampled Gaussians.

    For Gaussians whose screen-space radius < 1 pixel, their opacity is scaled
    down proportionally to their pixel coverage.  This prevents aliasing
    (sparkling/strobing) on thin structures and distant Gaussians.

    Parameters
    ----------
    opacities : torch.Tensor
        Shape (N, 1) opacity values.
    radii_2d : torch.Tensor
        Shape (N,) screen-space radius in pixels.
    pixel_coverages : torch.Tensor
        Shape (N,) fraction of pixel covered.
    scaling_modifier : float
        Global scale modifier (from render settings).
    strength : float
        Anti-aliasing strength (1.0 = full, 0.0 = disabled).

    Returns
    -------
    torch.Tensor
        Shape (N, 1) modified opacities.
    float
        Effective scale modifier (adjusted for mip).
    """
    # Compute the sub-pixel attenuation factor
    # For radii < 1 pixel, we attenuate smoothly:
    # attenuation = min(1, pixel_coverage * (1 + radius))
    sub_pixel_mask = (radii_2d < 1.0).float()  # (N,)
    # Smooth attenuation: at radius=0.5, coverage=0.25, attenuate to ~0.5
    attenuation = torch.where(
        radii_2d < 1.0,
        (pixel_coverages + radii_2d * 0.5).clamp(max=1.0),
        torch.ones_like(radii_2d),
    )

    # Apply weighted by strength parameter
    if strength < 1.0:
        attenuation = 1.0 - strength * (1.0 - attenuation)

    modified_opacities = opacities * attenuation.unsqueeze(-1)

    # Also adjust scale_modifier: for very small Gaussians, prevent them
    # from vanishing entirely by keeping scale_modifier at 1.0
    effective_scale_modifier = scaling_modifier

    return modified_opacities, effective_scale_modifier


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to 3×3 rotation matrices.

    Parameters
    ----------
    quats : torch.Tensor
        Shape (N, 4) normalised quaternions.

    Returns
    -------
    torch.Tensor
        Shape (N, 3, 3) rotation matrices.
    """
    N = quats.shape[0]
    device = quats.device

    # Normalise
    norm = torch.sqrt(
        quats[:, 0] ** 2 + quats[:, 1] ** 2 + quats[:, 2] ** 2 + quats[:, 3] ** 2
    ).unsqueeze(-1).clamp(min=1e-8)
    q = quats / norm  # (N, 4)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(N, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def anti_aliasing_filter(
    opacities: torch.Tensor,
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    means2D: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tanfovx: float,
    tanfovy: float,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    scaling_modifier: float = 1.0,
    strength: float = 1.0,
) -> tuple:
    """Full anti-aliasing pipeline: compute footprint → filter opacities.

    Convenience wrapper that runs :func:`compute_mip_level` and
    :func:`apply_antialiasing` in sequence.

    Parameters
    ----------
    opacities : torch.Tensor
        Shape (N, 1) opacity values.
    means3D : torch.Tensor
        Shape (N, 3) Gaussian positions.
    scales : torch.Tensor
        Shape (N, 3) Gaussian scales (activated).
    rotations : torch.Tensor
        Shape (N, 4) Gaussian rotations (quaternions).
    means2D : torch.Tensor
        Shape (N, 2) projected screen-space means (for frustum-culled only).
    viewmatrix : torch.Tensor
        Shape (4, 4) world-to-camera matrix.
    projmatrix : torch.Tensor
        Shape (4, 4) projection matrix.
    tanfovx, tanfovy : float
        Half FoV tangents.
    image_width, image_height : int
        Image dimensions.
    fx, fy : float
        Focal lengths in pixels.
    scaling_modifier : float
        Global scale modifier.
    strength : float
        Anti-aliasing strength from config.

    Returns
    -------
    torch.Tensor
        Shape (N, 1) anti-aliased opacities.
    float
        Effective scale modifier (adjusted).
    """
    radii_2d, mip_levels, pixel_coverages = compute_mip_level(
        means3D, scales, rotations,
        viewmatrix, projmatrix,
        tanfovx, tanfovy,
        image_width, image_height,
        fx, fy,
    )
    return apply_antialiasing(
        opacities, radii_2d, pixel_coverages,
        scaling_modifier, strength,
    )

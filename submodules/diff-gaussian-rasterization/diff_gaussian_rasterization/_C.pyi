"""Type stubs for compiled CUDA rasterizer module _C."""
from typing import Tuple
import torch

def rasterize_gaussians(
    background: torch.Tensor,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    antialiasing: bool,
    debug: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

def rasterize_gaussians_backward(
    background: torch.Tensor,
    means3D: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    dL_dout_color: torch.Tensor,
    dL_dout_invdepth: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    geomBuffer: torch.Tensor,
    R: int,
    binningBuffer: torch.Tensor,
    imageBuffer: torch.Tensor,
    antialiasing: bool,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

def mark_visible(
    means3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
) -> torch.Tensor: ...

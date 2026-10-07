#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, use_opacity_network=False, opacity_network=None, use_antialiasing=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=use_antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # NOTE: Must be initialized BEFORE neural feature blocks, which reference scales/rotations.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ── Learned Opacity Field Network ──────────────────────────────────
    if use_opacity_network and opacity_network is not None:
        from scene.opacity_network import (
            OpacityFieldNetwork,
            compute_view_directions,
            compute_depths,
        )

        # Compute view directions: from each Gaussian toward camera
        campos = raster_settings.campos
        view_dirs = compute_view_directions(means3D, campos)

        # Compute camera-space depths
        depths = compute_depths(
            means3D,
            raster_settings.viewmatrix,
            znear=0.01,
            zfar=100.0,
        )

        # Predict view-dependent opacity
        opacity = opacity_network(
            means3D,
            scales if scales is not None else pc.get_scaling,
            opacity,  # base_opacity
            view_dirs,
            depths,
        )

    # ── Anti-Aliasing (mip-aware footprint) ────────────────────────────
    if use_antialiasing:
        from gaussian_renderer.antialiasing import anti_aliasing_filter
        from utils.graphics_utils import fov2focal

        image_w = raster_settings.image_width
        image_h = raster_settings.image_height
        # Focal length from FOV
        fx = fov2focal(viewpoint_camera.FoVx, image_w)
        fy = fov2focal(viewpoint_camera.FoVy, image_h)

        # We need the (non-None) scales and rotations for footprint computation
        aa_scales = scales if scales is not None else pc.get_scaling
        aa_rots = rotations if rotations is not None else pc.get_rotation

        opacity, effective_sm = anti_aliasing_filter(
            opacity,
            means3D,
            aa_scales,
            aa_rots,
            means2D,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            image_w,
            image_h,
            fx, fy,
            scaling_modifier,
            strength=1.0,
        )
        scaling_modifier = effective_sm

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Handle different return formats (2, 3, or 6 values)
    mean_T = None
    depth_var = None
    vis_count = None
    if len(rasterizer_output) == 2:
        rendered_image, radii = rasterizer_output
        depth_image = None
    elif len(rasterizer_output) == 3:
        rendered_image, radii, depth_image = rasterizer_output
    else:
        rendered_image, radii, depth_image, mean_T, depth_var, vis_count = rasterizer_output[:6]
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "mean_T": mean_T,
        "depth_var": depth_var,
        "vis_count": vis_count,
    }
    
    return out

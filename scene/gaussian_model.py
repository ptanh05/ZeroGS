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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore
except (ImportError, AttributeError):
    SparseGaussianAdam = None

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scene.policy_network import SplitPolicyNetwork

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense: float = 0.0
        self.spatial_lr_scale: float = 0.0
        self.tmp_radii = None
        self._exposure = torch.empty(0)
        self.exposure_mapping = {}
        self.pretrained_exposures = None
        self.exposure_optimizer = None
        self.xyz_scheduler_args = None
        self.exposure_scheduler_args = None
        # Neural-Controlled Adaptive GS:
        # Policy network (set externally by train.py when use_policy_network=True)
        self.policy_network: Optional["SplitPolicyNetwork"] = None
        self.policy_optimizer: Optional[torch.optim.Optimizer] = None
        self.use_policy_network: bool = False
        self.gaussians_per_iter_cap: int = 5000
        self.max_gaussians_cap: int = 1_000_000
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is not None and image_name in self.pretrained_exposures:
            return self.pretrained_exposures[image_name]
        if (
            hasattr(self, "exposure_mapping")
            and self.exposure_mapping is not None
            and image_name in self.exposure_mapping
            and hasattr(self, "_exposure")
            and self._exposure is not None
            and isinstance(self._exposure, torch.Tensor)
            and self._exposure.shape[0] > self.exposure_mapping[image_name]
        ):
            return self._exposure[self.exposure_mapping[image_name]]
        device = (
            self.get_xyz.device
            if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0)
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        return torch.eye(3, 4, device=device)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos = None, spatial_lr_scale : float = 0.0):
        if isinstance(cam_infos, (int, float)) and spatial_lr_scale == 0.0:
            spatial_lr_scale = float(cam_infos)
            cam_infos = None
        self.spatial_lr_scale = spatial_lr_scale
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float, device=device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float, device=device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.tmp_radii = None
        self.pretrained_exposures = None

        if cam_infos and hasattr(cam_infos, "__iter__"):
            self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos) if hasattr(cam_info, "image_name")}
            exposure = torch.eye(3, 4, device=device)[None].repeat(len(cam_infos), 1, 1)
            self._exposure = nn.Parameter(exposure.requires_grad_(True))
        else:
            self.exposure_mapping = {}
            self._exposure = nn.Parameter(torch.empty(0, 3, 4, device=device).requires_grad_(True))

    def training_setup(self, training_args):
        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        self.percent_dense = getattr(training_args, "percent_dense", 0.01)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        l = [
            {'params': [self._xyz], 'lr': getattr(training_args, "position_lr_init", 0.00016) * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': getattr(training_args, "feature_lr", 0.0025), "name": "f_dc"},
            {'params': [self._features_rest], 'lr': getattr(training_args, "feature_lr", 0.0025) / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': getattr(training_args, "opacity_lr", 0.025), "name": "opacity"},
            {'params': [self._scaling], 'lr': getattr(training_args, "scaling_lr", 0.005), "name": "scaling"},
            {'params': [self._rotation], 'lr': getattr(training_args, "rotation_lr", 0.001), "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            if SparseGaussianAdam is not None:
                try:
                    self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
                except Exception:
                    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            else:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        if hasattr(self, '_exposure') and isinstance(self._exposure, nn.Parameter) and self._exposure.numel() > 0:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        else:
            self.exposure_optimizer = None

        if hasattr(training_args, "position_lr_init") and hasattr(training_args, "position_lr_final"):
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        else:
            self.xyz_scheduler_args = None
        
        if hasattr(training_args, "exposure_lr_init") and hasattr(training_args, "exposure_lr_final"):
            self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                            lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                            lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                            max_steps=training_args.iterations)
        else:
            self.exposure_scheduler_args = None

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if (
            self.pretrained_exposures is None
            and hasattr(self, "exposure_optimizer")
            and self.exposure_optimizer is not None
            and hasattr(self, "exposure_scheduler_args")
            and self.exposure_scheduler_args is not None
        ):
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and hasattr(self, "xyz_scheduler_args")
            and self.xyz_scheduler_args is not None
        ):
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).to(device) for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None
        else:
            self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.tmp_radii = None
        if not hasattr(self, "exposure_mapping") or self.exposure_mapping is None:
            self.exposure_mapping = {}
        if not hasattr(self, "_exposure") or self._exposure is None:
            self._exposure = nn.Parameter(torch.empty(0, 3, 4, device=device).requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        if self.optimizer is None:
            group_param = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[name] = group_param
            return optimizable_tensors

        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                    if "exp_avg_sq" in stored_state:
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        if self.optimizer is None:
            if hasattr(self, "_xyz") and self._xyz.numel() > 0:
                self._xyz = nn.Parameter(self._xyz[mask].requires_grad_(True))
            if hasattr(self, "_features_dc") and self._features_dc.numel() > 0:
                self._features_dc = nn.Parameter(self._features_dc[mask].requires_grad_(True))
            if hasattr(self, "_features_rest") and self._features_rest.numel() > 0:
                self._features_rest = nn.Parameter(self._features_rest[mask].requires_grad_(True))
            if hasattr(self, "_opacity") and self._opacity.numel() > 0:
                self._opacity = nn.Parameter(self._opacity[mask].requires_grad_(True))
            if hasattr(self, "_scaling") and self._scaling.numel() > 0:
                self._scaling = nn.Parameter(self._scaling[mask].requires_grad_(True))
            if hasattr(self, "_rotation") and self._rotation.numel() > 0:
                self._rotation = nn.Parameter(self._rotation[mask].requires_grad_(True))
            return {
                "xyz": self._xyz,
                "f_dc": self._features_dc,
                "f_rest": self._features_rest,
                "opacity": self._opacity,
                "scaling": self._scaling,
                "rotation": self._rotation,
            }

        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if hasattr(self, "tmp_radii") and self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        if self.optimizer is None:
            for k, v in tensors_dict.items():
                old_val = getattr(self, f"_{k}" if hasattr(self, f"_{k}") else k, None)
                if old_val is not None:
                    cat_tensor = torch.cat((old_val, v), dim=0)
                    param = nn.Parameter(cat_tensor.requires_grad_(True))
                    optimizable_tensors[k] = param
                else:
                    optimizable_tensors[k] = nn.Parameter(v.requires_grad_(True))
            return optimizable_tensors

        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if hasattr(self, "tmp_radii") and self.tmp_radii is not None and new_tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        else:
            self.tmp_radii = None

        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        if grads.dtype == torch.bool:
            if grads.shape[0] < n_init_points:
                selected_pts_mask = torch.cat((grads, torch.zeros(n_init_points - grads.shape[0], device=device, dtype=torch.bool)))
            else:
                selected_pts_mask = grads[:n_init_points]
        else:
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device=device)
            padded_grad[:min(grads.shape[0], n_init_points)] = grads.squeeze()[:min(grads.shape[0], n_init_points)]
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if not selected_pts_mask.any():
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = (
            self.tmp_radii[selected_pts_mask].repeat(N)
            if (hasattr(self, "tmp_radii") and self.tmp_radii is not None)
            else None
        )

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * int(selected_pts_mask.sum().item()), device=device, dtype=torch.bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        if grads.dtype == torch.bool:
            if grads.shape[0] < n_init_points:
                selected_pts_mask = torch.cat((grads, torch.zeros(n_init_points - grads.shape[0], device=device, dtype=torch.bool)))
            else:
                selected_pts_mask = grads[:n_init_points]
        else:
            # Extract points that satisfy the gradient condition
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        if not selected_pts_mask.any():
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = (
            self.tmp_radii[selected_pts_mask]
            if (hasattr(self, "tmp_radii") and self.tmp_radii is not None)
            else None
        )

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii=None, max_world_size_ratio=0.1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        if self.use_policy_network and self.policy_network is not None:
            # ── Learned densification via policy network ──────────────────
            self._densify_via_policy(grads, extent)
        else:
            # ── Original rule-based densification (backward compatible) ──
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > max_world_size_ratio * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None

        # Enforce Gaussian count cap after pruning
        if 0 < self.max_gaussians_cap < self.get_xyz.shape[0]:
            self.prune_to_cap(self.max_gaussians_cap)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _densify_via_policy(self, grads, extent):
        """Apply learned policy network to decide SPLIT / MERGE / KEEP per Gaussian."""
        if self.policy_network is None:
            return

        from scene.policy_network import compute_policy_features

        # 1) Compute features for all Gaussians
        radii = (
            self.tmp_radii
            if (hasattr(self, "tmp_radii") and self.tmp_radii is not None)
            else torch.zeros(self.get_xyz.shape[0], device=self.get_xyz.device)
        )
        features = compute_policy_features(self, grads, radii, extent)

        # 2) Get action probabilities from policy network
        with torch.no_grad():
            probs = self.policy_network(features)
            actions = self.policy_network.sample_actions(probs)

        # 3) Apply actions
        split_mask = (actions == 1)   # SPLIT
        merge_mask = (actions == 2)   # MERGE

        # Cap the number of Gaussians we create this iteration
        max_new = max(1, self.gaussians_per_iter_cap)

        split_count = split_mask.sum().item()
        if split_count > max_new // 2:
            # Subsample: keep only candidates with highest probability
            split_probs = probs[split_mask, 1]  # P(SPLIT) for candidates
            k = min(split_count, max_new // 2)
            top_split_rel_indices = split_probs.topk(k).indices
            split_indices = split_mask.nonzero(as_tuple=False).squeeze(-1)
            selected_indices = split_indices[top_split_rel_indices]
            split_mask = torch.zeros_like(split_mask, dtype=torch.bool)
            split_mask[selected_indices] = True

        # Apply actions: MERGE first, then remap split_mask for surviving Gaussians and apply SPLIT
        if merge_mask.any():
            self.merge_gaussians(merge_mask)
            # Pruning merge candidates removed them from the population.
            # Since actions (KEEP=0, SPLIT=1, MERGE=2) are mutually exclusive,
            # update split_mask to reflect surviving Gaussians:
            split_mask = split_mask[~merge_mask]

        # Apply SPLIT
        if split_mask.any():
            self._split_selected(split_mask, extent)

        # ── Safety check: Gaussian attribute tensors must agree on N ─────
        n = self.get_xyz.shape[0]
        attr_tensors = [
            ("_xyz",              self._xyz),
            ("_features_dc",      self._features_dc),
            ("_features_rest",    self._features_rest),
            ("_opacity",          self._opacity),
            ("_scaling",          self._scaling),
            ("_rotation",         self._rotation),
            ("xyz_gradient_accum", self.xyz_gradient_accum),
            ("denom",             self.denom),
            ("max_radii2D",       self.max_radii2D),
        ]
        for name, t in attr_tensors:
            assert t.shape[0] == n, (
                f"Tensor mismatch after policy densification: "
                f"{name}.shape[0]={t.shape[0]} != _xyz.shape[0]={n}"
            )

    def _split_selected(self, split_mask, extent, N=2):
        """Split selected Gaussians (same logic as densify_and_split but with explicit mask)."""
        if not split_mask.any():
            return
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        if split_mask.shape[0] < n_init_points:
            split_mask = torch.cat((split_mask, torch.zeros(n_init_points - split_mask.shape[0], device=device, dtype=torch.bool)))
        elif split_mask.shape[0] > n_init_points:
            split_mask = split_mask[:n_init_points]

        stds = self.get_scaling[split_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[split_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[split_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[split_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[split_mask].repeat(N, 1)
        new_features_dc = self._features_dc[split_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[split_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[split_mask].repeat(N, 1)
        new_tmp_radii = (
            self.tmp_radii[split_mask].repeat(N)
            if (hasattr(self, "tmp_radii") and self.tmp_radii is not None)
            else None
        )

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation, new_tmp_radii,
        )

        # Prune the original Gaussians that were split (replaced by children)
        prune_filter = torch.cat(
            (split_mask, torch.zeros(N * int(split_mask.sum().item()), device=device, dtype=torch.bool))
        )
        self.prune_points(prune_filter)

    def merge_gaussians(self, merge_mask):
        """Merge selected Gaussians with their nearest neighbours.

        For each Gaussian selected for MERGE, find the nearest unselected
        neighbour and combine attributes (average position, sum opacity,
        average other params).  The merged Gaussian replaces the neighbour;
        the merge-selected Gaussian is pruned.
        """
        if not merge_mask.any():
            return

        N = self.get_xyz.shape[0]
        if merge_mask.sum() >= N:
            return

        device = self.get_xyz.device

        # Candidates to merge: indices where merge_mask is True
        merge_indices = merge_mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
        M = merge_indices.shape[0]
        if M == 0:
            return

        # For each merge candidate, find nearest neighbour (NOT in merge set)
        xyz = self._xyz  # (N, 3)
        xyz_merge = xyz[merge_indices]  # (M, 3)

        # Compute pairwise distances: (M, N)
        dists = torch.cdist(xyz_merge, xyz)  # (M, N)
        # Mask out self and other merge candidates
        mask_self = torch.zeros(M, N, device=device, dtype=torch.bool)
        for i, idx in enumerate(merge_indices):
            mask_self[i, idx] = True
            mask_self[i, merge_mask] = True

        dists[mask_self] = float("inf")

        # Nearest neighbour for each merge candidate
        nn_indices = dists.argmin(dim=-1)  # (M,)

        # Average attributes: merge_i → nn_i
        for i, merge_idx in enumerate(merge_indices):
            nn_idx = nn_indices[i]
            if torch.isinf(dists[i, nn_idx]):
                continue

            # Weighted average by distance (closer = more weight)
            dist_val = dists[i, nn_idx].clamp(min=1e-8)
            w_merge = 1.0 / (dist_val + 1.0)
            w_nn = 1.0
            w_sum = w_merge + w_nn

            # Position: average
            self._xyz.data[nn_idx] = (
                self._xyz[merge_idx] * w_merge + self._xyz[nn_idx] * w_nn
            ) / w_sum

            # Features (dc): average
            self._features_dc.data[nn_idx] = (
                self._features_dc[merge_idx] + self._features_dc[nn_idx]
            ) / 2.0

            # Features (rest): average
            self._features_rest.data[nn_idx] = (
                self._features_rest[merge_idx] + self._features_rest[nn_idx]
            ) / 2.0

            # Opacity: max (keep the more visible one)
            op_merge = self._opacity[merge_idx]
            op_nn = self._opacity[nn_idx]
            self._opacity.data[nn_idx] = torch.where(
                op_merge > op_nn, op_merge, op_nn
            )

            # Scale: weighted average via log-space
            self._scaling.data[nn_idx] = (
                self._scaling[merge_idx] * w_merge + self._scaling[nn_idx] * w_nn
            ) / w_sum

            # Rotation: spherical linear interpolation would be ideal,
            # but simple average + renormalise is sufficient here
            rot_avg = (
                self._rotation[merge_idx] * w_merge + self._rotation[nn_idx] * w_nn
            ) / w_sum
            self._rotation.data[nn_idx] = torch.nn.functional.normalize(
                rot_avg.unsqueeze(0), dim=-1
            ).squeeze(0)

            # Max radii: keep the larger one
            if hasattr(self, "max_radii2D") and self.max_radii2D.numel() > max(int(merge_idx.item()), int(nn_idx.item())):
                self.max_radii2D[nn_idx] = torch.max(
                    self.max_radii2D[merge_idx], self.max_radii2D[nn_idx]
                )

        # Mark merge indices for pruning (keep their neighbours)
        self.prune_points(merge_mask)

    _merge_selected = merge_gaussians

    def prune_to_cap(self, max_count):
        """Prune lowest-opacity Gaussians until count ≤ max_count."""
        if self.get_xyz.shape[0] <= max_count:
            return
        n_prune = self.get_xyz.shape[0] - max_count
        # Order by opacity (ascending)
        opacity = self.get_opacity.squeeze(-1)
        _, indices = opacity.sort()
        device = self.get_xyz.device if (hasattr(self, "_xyz") and self._xyz is not None and self._xyz.numel() > 0) else ("cuda" if torch.cuda.is_available() else "cpu")
        prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=device)
        prune_mask[indices[:n_prune]] = True
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if update_filter is None or viewspace_point_tensor is None or getattr(viewspace_point_tensor, "grad", None) is None:
            return
        if hasattr(update_filter, "numel") and update_filter.numel() == 0:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

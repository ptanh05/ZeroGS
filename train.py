"""
ZeroGS: Byte-Level Transient Memory Modeling and Occlusion-Aware Admission Control
for Budget-Constrained 3D Gaussian Splatting.

Complete integration into the official 3DGS training loop (train.py).
Replaces unconstrained/count-based densification with deterministic, byte-aware
Zero-OOM admission control and occlusion-compensated marginal utility scheduling.
"""

import os
import sys
import json
import platform
import subprocess
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional, Dict, Any

import torch
from tqdm import tqdm

# =========================================================================
# 1. OFFICIAL 3DGS MODULE IMPORTS
# =========================================================================
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

# =========================================================================
# 2. ZERO-OOM & OCCLUSION-AWARE SYSTEM RUNTIME MODULES
# =========================================================================
from zerogs.cost_model import ByteCostModel
from zerogs.admission_controller import (
    AdmissionController,
    AdmissionAction,
    VramBudgetExceededError,
)
from zerogs.occlusion_engine import (
    OcclusionAwareEngine,
    OcclusionAwareDemandEstimator,
    bottom_utility_mask,
)
from zerogs.marginal_allocator import MarginalUtilityAllocator
from zerogs.memory_telemetry import JsonlTelemetryWriter, capture_cuda_memory

# Standalone simulation fallback model for running tests without COLMAP datasets
from zerogs.mock_gaussian_model import (
    MockGaussianModel,
    MockCamera,
    mock_render,
)


def resolve_simulation_mode(source_path: str, simulation_mode: bool) -> bool:
    """Require an explicit opt-in before using synthetic training data."""
    src = source_path.strip() if source_path else ""
    has_valid_scene = bool(src) and (
        os.path.exists(os.path.join(src, "sparse"))
        or os.path.exists(os.path.join(src, "transforms_train.json"))
    )
    if has_valid_scene:
        return False
    if simulation_mode:
        return True
    raise FileNotFoundError(
        "No valid COLMAP/Blender scene was found. Pass -s/--source_path to a "
        "directory containing sparse/ or transforms_train.json. Use "
        "--simulation_mode only for smoke tests; simulation results are not "
        "valid research measurements."
    )


def compute_photometric_loss(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    lambda_dssim: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official 3DGS photometric objective with validated DSSIM weighting."""
    if not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError(f"lambda_dssim must be in [0, 1], got {lambda_dssim}")
    l1_value = l1_loss(image, ground_truth)
    ssim_value = ssim(image, ground_truth)
    loss = (1.0 - lambda_dssim) * l1_value + lambda_dssim * (1.0 - ssim_value)
    return loss, l1_value, ssim_value


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: list,
    saving_iterations: list,
    checkpoint_iterations: list,
    checkpoint: Optional[str] = None,
    debug_from: int = -1,
):
    first_iter = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check if a valid COLMAP or Blender dataset exists in source_path
    is_simulation = resolve_simulation_mode(
        dataset.source_path,
        bool(getattr(opt, "simulation_mode", False)),
    )

    # Initialize Gaussian Model & Scene
    if not is_simulation:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        with open(os.path.join(scene.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(dataset))))
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        scene_extent = scene.cameras_extent
    else:
        print("[ZeroGS] Explicit simulation mode enabled. Results are for smoke testing only.")
        num_init_pts = getattr(opt, "num_init_points", 5000)
        gaussians = MockGaussianModel(sh_degree=dataset.sh_degree, num_points=num_init_pts, device=device)
        scene = None
        background = torch.zeros(3, device=device)
        scene_extent = 3.0

    iter_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    iter_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="ZeroGS Training")
    first_iter += 1

    # =========================================================================
    # 3. INITIALIZE ZERO-OOM SYSTEM RUNTIME CONTROLLERS
    # =========================================================================
    hard_vram_limit_mb = getattr(opt, "hard_vram_limit_mb", 8192.0)
    safety_headroom_mb = getattr(opt, "safety_headroom_mb", 512.0)
    gamma_occ = getattr(opt, "gamma_occ", 0.35)
    transmittance_tau = getattr(opt, "transmittance_tau", 0.25)
    occlusion_mode = getattr(opt, "occlusion_mode", "full")
    enable_structure_protection = getattr(opt, "enable_structure_protection", True)
    allocator_strategy = getattr(opt, "allocator_strategy", "gain_per_byte")
    gaussians_per_iter_cap = int(getattr(opt, "gaussians_per_iter_cap", 5000))
    max_gaussians_cap = int(getattr(opt, "max_gaussians_cap", 1_000_000))
    enable_zero_oom = getattr(opt, "enable_zero_oom", True)
    require_native_stats = getattr(opt, "require_native_stats", True)
    non_pytorch_allowance_mb = getattr(opt, "non_pytorch_allowance_mb", 256.0)

    calibration_path = getattr(opt, "cost_calibration", None)
    if calibration_path:
        cost_model = ByteCostModel.from_calibration_file(calibration_path)
        if cost_model.sh_degree != dataset.sh_degree:
            raise ValueError(
                f"Calibration SH degree {cost_model.sh_degree} does not match dataset "
                f"SH degree {dataset.sh_degree}"
            )
    else:
        cost_model = ByteCostModel(
            sh_degree=dataset.sh_degree,
            use_adam=True,
            clone_multiplier=1.15,
            split_multiplier=2.30,
        )
        if not is_simulation:
            print(
                "[ZeroGS] WARNING: no --cost_calibration supplied; hard-budget "
                "claims must be treated as uncalibrated."
            )
    admission_ctrl = AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=hard_vram_limit_mb,
        safety_headroom_mb=safety_headroom_mb,
        non_pytorch_allowance_mb=non_pytorch_allowance_mb,
        enable_deterministic_prune_first=True,
    )
    occ_engine = OcclusionAwareEngine(
        num_gaussians=gaussians.get_xyz.shape[0],
        gamma_occ=gamma_occ,
        transmittance_tau=transmittance_tau,
        occlusion_mode=occlusion_mode,
        occlusion_physics_mode=getattr(opt, "occlusion_physics_mode", "corrected"),
        enable_structure_protection=enable_structure_protection,
        device=device,
    )
    allocator = MarginalUtilityAllocator(cost_model=cost_model)

    # Accumulation buffers between densification intervals
    num_pts = gaussians.get_xyz.shape[0]
    accum_visibility = torch.zeros(num_pts, dtype=torch.int32, device=device)
    accum_pixel_footprint = torch.zeros(num_pts, dtype=torch.int32, device=device)
    accum_transmittance = torch.zeros(num_pts, dtype=torch.float32, device=device)
    accum_depth_var = torch.zeros(num_pts, dtype=torch.float32, device=device)
    viewspace_point_tensor_grad = torch.zeros(num_pts, device=device)
    telemetry_path = getattr(opt, "telemetry_jsonl", None)
    if telemetry_path is None and dataset.model_path:
        telemetry_path = os.path.join(dataset.model_path, "training_telemetry.jsonl")
    telemetry_writer = JsonlTelemetryWriter(telemetry_path, truncate=True) if telemetry_path else None
    if telemetry_writer is not None:
        telemetry_writer.write(
            "training_start",
            run_id=getattr(opt, "run_id", None),
            simulation_mode=is_simulation,
            initial_gaussians=int(num_pts),
            hard_vram_limit_mb=hard_vram_limit_mb,
            safety_headroom_mb=safety_headroom_mb,
            non_pytorch_allowance_mb=non_pytorch_allowance_mb,
            occlusion_mode=occlusion_mode,
            enable_structure_protection=enable_structure_protection,
            allocator_strategy=allocator_strategy,
        )
    print(
        f"[ZeroGS] Configured: Hard Budget={hard_vram_limit_mb}MB, "
        f"Safety Headroom={safety_headroom_mb}MB, "
        f"Non-PyTorch Allowance={non_pytorch_allowance_mb}MB, gamma_occ={gamma_occ}"
    )

    # Neural-Controlled Adaptive GS extensions
    use_opacity_network = getattr(opt, "use_opacity_network", False)
    use_antialiasing = getattr(pipe, "antialiasing", False) or getattr(opt, "use_antialiasing", False)
    use_policy_network = getattr(opt, "use_policy_network", False)

    opacity_network = None
    opacity_optimizer = None
    if use_opacity_network and not is_simulation:
        from scene.opacity_network import OpacityFieldNetwork
        opacity_network = OpacityFieldNetwork().to(device)
        opacity_optimizer = torch.optim.Adam(
            opacity_network.parameters(), lr=getattr(opt, "opacity_network_lr", 1e-4)
        )

    if use_policy_network:
        from scene.policy_network import SplitPolicyNetwork
        policy_network = SplitPolicyNetwork().to(device)
        policy_optimizer = torch.optim.Adam(
            policy_network.parameters(), lr=getattr(opt, "policy_lr", 1e-4)
        )
        if hasattr(gaussians, "policy_network"):
            gaussians.policy_network = policy_network
            gaussians.policy_optimizer = policy_optimizer
            gaussians.use_policy_network = True

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    last_opacity_reset_iteration = -999999
    if first_iter > 0:
        last_opacity_reset_iteration = (first_iter // opt.opacity_reset_interval) * opt.opacity_reset_interval
        if getattr(dataset, "white_background", False) and first_iter >= opt.densify_from_iter and last_opacity_reset_iteration < opt.densify_from_iter:
            last_opacity_reset_iteration = opt.densify_from_iter

    for iteration in range(first_iter, opt.iterations + 1):
        if not is_simulation and network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if not keep_alive:
                    break
            except Exception:
                network_gui.conn = None

        if iter_start is not None:
            iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Camera selection
        if not is_simulation:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            viewpoint_cam = MockCamera(camera_id=iteration % 10, device=device)

        bg = torch.rand((3), device=device) if getattr(opt, "random_background", False) else background

        # ---------------------------------------------------------------------
        # FORWARD PASS: RENDER KÈM TRÍCH XUẤT TRANSMITTANCE & DEPTH VARIANCE
        # ---------------------------------------------------------------------
        if not is_simulation:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, bg,
                use_opacity_network=use_opacity_network,
                opacity_network=opacity_network,
                use_antialiasing=use_antialiasing,
            )
        else:
            render_pkg = mock_render(viewpoint_cam, gaussians, pipe, bg)

        if (
            not is_simulation
            and enable_zero_oom
            and gamma_occ > 0.0
            and occlusion_mode != "none"
            and require_native_stats
            and not bool(render_pkg.get("stats_available", False))
        ):
            raise RuntimeError(
                "The active CUDA rasterizer does not provide native mean_T/depth_var/vis_count. "
                "Rebuild the ZeroGS rasterizer before running occlusion-aware research. "
                "Use --allow_missing_native_stats only for a non-occlusion baseline."
            )

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        cur_mean_T = render_pkg.get("mean_T", render_pkg.get("mean_transmittance", None))
        cur_depth_var = render_pkg.get("depth_var", render_pkg.get("depth_variance", None))
        cur_vis_count = render_pkg.get("vis_count", None)

        in_frustum_mask = None
        has_pixels_mask = None
        gradient_mask_mode = getattr(opt, "gradient_mask_mode", "canonical")
        if not enable_zero_oom:
            gradient_mask_mode = "canonical"

        if iteration < opt.densify_until_iter:
            with torch.no_grad():
                in_frustum_mask = (radii > 0) if radii is not None else visibility_filter
                if in_frustum_mask is not None and in_frustum_mask.dim() > 1:
                    in_frustum_mask = in_frustum_mask.squeeze()

                has_pixels_mask = (cur_vis_count > 0) if cur_vis_count is not None else in_frustum_mask
                if has_pixels_mask is not None and has_pixels_mask.dim() > 1:
                    has_pixels_mask = has_pixels_mask.squeeze()

                # Denominator visibility accumulation:
                # - Canonical: increments for all primitives in camera frustum (radii > 0)
                # - Contribution-aware: increments only for primitives with >= 1 contributing pixel
                dens_denom_mask = in_frustum_mask if gradient_mask_mode == "canonical" else has_pixels_mask
                if dens_denom_mask is not None and dens_denom_mask.numel() > 0:
                    accum_visibility[dens_denom_mask] += 1

                if has_pixels_mask is not None and has_pixels_mask.numel() > 0:
                    if cur_vis_count is not None:
                        accum_pixel_footprint[has_pixels_mask] += cur_vis_count[has_pixels_mask]
                    if cur_mean_T is not None:
                        accum_transmittance[has_pixels_mask] += cur_mean_T[has_pixels_mask]
                    if cur_depth_var is not None:
                        accum_depth_var[has_pixels_mask] += cur_depth_var[has_pixels_mask]

                    # Update physics stats in Occlusion Engine
                    if hasattr(viewpoint_cam, "world_view_transform"):
                        R_z = viewpoint_cam.world_view_transform[2, :3].to(device)
                        t_z = viewpoint_cam.world_view_transform[2, 3].to(device)
                        view_depths = (gaussians.get_xyz[has_pixels_mask].detach() @ R_z) + t_z
                    else:
                        view_depths = torch.norm(gaussians.get_xyz[has_pixels_mask].detach(), dim=-1)

                    T_vals = cur_mean_T[has_pixels_mask] if cur_mean_T is not None else torch.ones_like(view_depths)
                    vis_indices = torch.where(has_pixels_mask)[0]
                    occ_engine.update_per_view_stats(
                        visible_indices=vis_indices,
                        view_depths=view_depths,
                        view_transmittance=T_vals,
                        view_depth_variance=(
                            cur_depth_var[has_pixels_mask] if cur_depth_var is not None else None
                        ),
                    )

        # ---------------------------------------------------------------------
        # LOSS & BACKWARD PASS
        # ---------------------------------------------------------------------
        gt_image = getattr(viewpoint_cam, "original_image", torch.zeros_like(image)).to(device)
        lambda_dssim = float(getattr(opt, "lambda_dssim", 0.2))
        loss, Ll1, ssim_value = compute_photometric_loss(image, gt_image, lambda_dssim)
        loss.backward()

        if enable_zero_oom and not is_simulation:
            try:
                admission_ctrl.assert_current_memory_within_budget(
                    telemetry_writer=telemetry_writer,
                    iteration=iteration,
                )
            except VramBudgetExceededError:
                if telemetry_writer is not None:
                    telemetry_writer.write(
                        "budget_exceeded",
                        iteration=iteration,
                        memory=capture_cuda_memory().to_dict(),
                    )
                raise

        if iter_end is not None:
            iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                vram_mb = (
                    torch.cuda.memory_allocated() / (1024**2)
                    if torch.cuda.is_available()
                    else cost_model.estimate_persistent_bytes(gaussians.get_xyz.shape[0]) / (1024**2)
                )
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.4f}",
                    "Pts": f"{gaussians.get_xyz.shape[0]:,}",
                    "VRAM": f"{vram_mb:.0f}MB",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Viewspace gradient norm accumulation
            if viewspace_point_tensor.grad is not None:
                dens_num_mask = in_frustum_mask if gradient_mask_mode == "canonical" else has_pixels_mask
                if dens_num_mask is not None and dens_num_mask.numel() > 0:
                    viewspace_point_tensor_grad[dens_num_mask] += torch.norm(
                        viewspace_point_tensor.grad[dens_num_mask, :2], dim=-1
                    )

            # -----------------------------------------------------------------
            # ZERO-OOM ADAPTIVE DENSIFICATION & PRUNING CYCLE
            # -----------------------------------------------------------------
            if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
                if hasattr(gaussians, "max_radii2D") and visibility_filter is not None:
                    vis_mask = visibility_filter.squeeze() if visibility_filter.dim() > 1 else visibility_filter
                    gaussians.max_radii2D[vis_mask] = torch.max(
                        gaussians.max_radii2D[vis_mask], radii[vis_mask]
                    )

                if iteration % opt.densification_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    mutation_memory_before = capture_cuda_memory()
                    pruning_mode = getattr(opt, "pruning_mode", "decoupled")
                    tau_age = getattr(opt, "tau_age", 500)
                    tau_reset = getattr(opt, "tau_reset", 400)
                    kappa_split = getattr(opt, "kappa_split", 0.40)
                    inflow_mode = getattr(opt, "inflow_mode", "dynamic")
                    inflow_cap_min = getattr(opt, "inflow_cap_min", 5000)
                    inflow_cap_max = getattr(opt, "inflow_cap_max", 15000)
                    target_headroom_mb = getattr(opt, "target_headroom_mb", 1024.0)

                    if not enable_zero_oom:
                        # -------------------------------------------------------------
                        # VANILLA 3DGS BASELINE PATH (V_ref: Clone -> Split -> Prune)
                        # No Admission Controller, No Allocator Quotas, No Population Caps
                        # -------------------------------------------------------------
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        denom = torch.clamp(accum_visibility.float(), min=1.0)
                        avg_grad = viewspace_point_tensor_grad / denom
                        grads_high = avg_grad >= opt.densify_grad_threshold
                        scale_limit = opt.percent_dense * scene_extent

                        clone_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values <= scale_limit,
                        )
                        split_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values > scale_limit,
                        )

                        n_clone_executed = int(clone_mask.sum().item())
                        n_split_executed = int(split_mask.sum().item())
                        if n_clone_executed > 0:
                            gaussians.densify_and_clone(clone_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)
                        if n_split_executed > 0:
                            gaussians.densify_and_split(split_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)

                        min_opacity_prune = getattr(opt, "min_opacity_prune", 0.005)
                        prune_mask = (gaussians.get_opacity < min_opacity_prune).squeeze()
                        if size_threshold is not None:
                            if hasattr(gaussians, "max_radii2D"):
                                big_points_vs = gaussians.max_radii2D > size_threshold
                                prune_mask = torch.logical_or(prune_mask, big_points_vs)
                            max_world_size_ratio = getattr(opt, "max_world_size_ratio", 0.1)
                            big_points_ws = torch.max(gaussians.get_scaling, dim=1).values > (max_world_size_ratio * scene_extent)
                            prune_mask = torch.logical_or(prune_mask, big_points_ws)

                        executed_prune = int(prune_mask.sum().item())
                        if executed_prune > 0:
                            gaussians.prune_points(prune_mask)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        mutation_memory_after = capture_cuda_memory()
                        if telemetry_writer is not None:
                            telemetry_writer.write(
                                "mutation_cycle",
                                iteration=iteration,
                                decision={"action": "VANILLA_BASELINE", "n_clone": n_clone_executed, "n_split": n_split_executed, "required_pre_prune": 0},
                                executed_pre_prune=0,
                                executed_hygiene_prune=executed_prune,
                                n_clone_executed=n_clone_executed,
                                n_split_executed=n_split_executed,
                                memory_before=mutation_memory_before.to_dict(),
                                memory_after=mutation_memory_after.to_dict(),
                                raster_stats_available=bool(render_pkg.get("stats_available", False)),
                                allocator_strategy="vanilla",
                                occlusion_mode="none",
                                structure_protection=False,
                                mutation_item_cap=None,
                            )

                    elif pruning_mode == "legacy_as_is":
                        # -------------------------------------------------------------
                        # LEGACY ZERO-GS ABLATION PATH (A0: Coupling bug preserved)
                        # -------------------------------------------------------------
                        occ_engine.grad_accum = viewspace_point_tensor_grad
                        demand, protected_mask = occ_engine.compute_compensated_demand(scene_extent=scene_extent)
                        demand = torch.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)

                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        denom = torch.clamp(accum_visibility.float(), min=1.0)
                        avg_grad = viewspace_point_tensor_grad / denom
                        grads_high = avg_grad >= opt.densify_grad_threshold
                        scale_limit = opt.percent_dense * scene_extent

                        clone_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values <= scale_limit,
                        )
                        split_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values > scale_limit,
                        )

                        dead_opacity_mask = (gaussians.get_opacity < 0.005).squeeze()
                        prunable_utility = bottom_utility_mask(demand, protected_mask, fraction=0.05)

                        redundant_mask = torch.logical_or(dead_opacity_mask, prunable_utility)
                        if size_threshold and hasattr(gaussians, "max_radii2D"):
                            too_large_mask = gaussians.max_radii2D > size_threshold
                            redundant_mask = torch.logical_or(redundant_mask, too_large_mask)

                        prune_candidates_count = int(redundant_mask.sum().item())
                        n_clone_req = int(clone_mask.sum().item())
                        n_split_req = int(split_mask.sum().item())

                        inflow_cap_min = getattr(opt, "inflow_cap_min", 5000)
                        inflow_cap_max = getattr(opt, "inflow_cap_max", 15000)
                        target_headroom_mb = getattr(opt, "target_headroom_mb", 1024.0)
                        inflow_mode = getattr(opt, "inflow_mode", "dynamic")

                        decision = admission_ctrl.evaluate_admission(
                            n_clone_requested=n_clone_req,
                            n_split_requested=n_split_req,
                            current_num_gaussians=gaussians.get_xyz.shape[0],
                            prune_candidates_count=prune_candidates_count,
                            inflow_cap_min=inflow_cap_min,
                            inflow_cap_max=inflow_cap_max,
                            target_headroom_mb=target_headroom_mb,
                            kappa_split=kappa_split,
                            inflow_mode=inflow_mode,
                        )

                        dec_action = decision.action if hasattr(decision, "action") else decision["action"]
                        dec_k_prune = decision.required_pre_prune if hasattr(decision, "required_pre_prune") else decision["required_pre_prune"]
                        dec_n_clone = decision.n_clone if hasattr(decision, "n_clone") else decision["n_clone"]
                        dec_n_split = decision.n_split if hasattr(decision, "n_split") else decision["n_split"]
                        allocation_decision = decision
                        executed_pre_prune = 0

                        if dec_action in [AdmissionAction.PRUNE_FIRST, AdmissionAction.SHRINK_AND_PRUNE_FIRST, "PRUNE_FIRST", "SHRINK_AND_PRUNE_FIRST"]:
                            k_prune = dec_k_prune
                            redundant_indices = torch.where(redundant_mask)[0]
                            if len(redundant_indices) > 0 and k_prune > 0:
                                sorted_by_demand = redundant_indices[torch.argsort(demand[redundant_indices])]
                                exec_prune_indices = sorted_by_demand[:k_prune]

                                exec_prune_mask = torch.zeros(
                                    gaussians.get_xyz.shape[0], dtype=torch.bool, device=device
                                )
                                exec_prune_mask[exec_prune_indices] = True
                                gaussians.prune_points(exec_prune_mask)
                                executed_pre_prune = int(exec_prune_mask.sum().item())

                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()

                                surviving = ~exec_prune_mask
                                clone_mask = clone_mask[surviving]
                                split_mask = split_mask[surviving]
                                demand = demand[surviving]

                                post_prune_decision = admission_ctrl.evaluate_admission(
                                    n_clone_requested=dec_n_clone,
                                    n_split_requested=dec_n_split,
                                    current_num_gaussians=gaussians.get_xyz.shape[0],
                                    prune_candidates_count=0,
                                    inflow_cap_min=getattr(opt, "inflow_cap_min", 5000),
                                    inflow_cap_max=getattr(opt, "inflow_cap_max", 15000),
                                    target_headroom_mb=getattr(opt, "target_headroom_mb", 1024.0),
                                    kappa_split=getattr(opt, "kappa_split", 0.40),
                                    inflow_mode=getattr(opt, "inflow_mode", "dynamic"),
                                )
                                dec_n_clone = post_prune_decision.n_clone
                                dec_n_split = post_prune_decision.n_split
                                allocation_decision = post_prune_decision

                        remaining_population_cap = max(
                            0, max_gaussians_cap - int(gaussians.get_xyz.shape[0])
                        )
                        inflow_mode = getattr(opt, "inflow_mode", "dynamic")
                        if inflow_mode == "dynamic" and hasattr(allocation_decision, "dynamic_inflow") and allocation_decision.dynamic_inflow:
                            effective_inflow_cap = allocation_decision.dynamic_inflow.get("computed_cap", gaussians_per_iter_cap)
                        else:
                            effective_inflow_cap = gaussians_per_iter_cap

                        mutation_item_cap = min(
                            effective_inflow_cap,
                            remaining_population_cap,
                        )
                        kappa_split = getattr(opt, "kappa_split", 0.40)
                        if hasattr(allocation_decision, "transient_cost_bytes"):
                            final_clone_mask, final_split_mask = allocator.joint_knapsack_allocate(
                                demand=demand,
                                clone_mask=clone_mask,
                                split_mask=split_mask,
                                max_transient_bytes=allocation_decision.mutation_budget_bytes,
                                strategy=allocator_strategy,
                                max_items=mutation_item_cap,
                                kappa_split=kappa_split,
                            )
                        else:
                            final_clone_mask, final_split_mask = allocator.allocate(
                                demand=demand,
                                clone_mask=clone_mask,
                                split_mask=split_mask,
                                n_clone_quota=dec_n_clone,
                                n_split_quota=dec_n_split,
                            )

                        if final_clone_mask.sum() > 0:
                            gaussians.densify_and_clone(final_clone_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)
                        if final_split_mask.sum() > 0:
                            gaussians.densify_and_split(final_split_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)

                        mutation_memory_after = capture_cuda_memory()
                        if telemetry_writer is not None:
                            telemetry_writer.write(
                                "mutation_cycle",
                                iteration=iteration,
                                decision=decision.to_dict() if hasattr(decision, "to_dict") else decision,
                                executed_pre_prune=executed_pre_prune,
                                n_clone_executed=int(final_clone_mask.sum().item()),
                                n_split_executed=int(final_split_mask.sum().item()),
                                memory_before=mutation_memory_before.to_dict(),
                                memory_after=mutation_memory_after.to_dict(),
                                raster_stats_available=bool(render_pkg.get("stats_available", False)),
                                allocator_strategy=allocator_strategy,
                                occlusion_mode=occlusion_mode,
                                structure_protection=enable_structure_protection,
                                mutation_item_cap=mutation_item_cap,
                            )

                    else:
                        # -------------------------------------------------------------
                        # DECOUPLED TWO-STAGE PRUNING PATH (Default ZeroGS)
                        # Stage 1: Regular Hygiene Pruning (Always executed)
                        # Stage 2: Admission Utility Pruning (Only if budget requires)
                        # -------------------------------------------------------------
                        # a. Compute Occlusion-Aware Demand & Protected Structure Mask
                        occ_engine.grad_accum = viewspace_point_tensor_grad
                        demand, protected_mask = occ_engine.compute_compensated_demand(scene_extent=scene_extent)
                        demand = torch.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)

                        # b. Identify Candidates for Clone & Split
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        denom = torch.clamp(accum_visibility.float(), min=1.0)
                        avg_grad = viewspace_point_tensor_grad / denom
                        grads_high = avg_grad >= opt.densify_grad_threshold
                        scale_limit = opt.percent_dense * scene_extent

                        clone_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values <= scale_limit,
                        )
                        split_mask = torch.logical_and(
                            grads_high,
                            torch.max(gaussians.get_scaling, dim=1).values > scale_limit,
                        )

                        # c. STAGE 1: REGULAR HYGIENE PRUNING (Always executed independently of admission!)
                        min_opacity_prune = getattr(opt, "min_opacity_prune", 0.005)
                        tau_age = getattr(opt, "tau_age", 500)
                        tau_reset = getattr(opt, "tau_reset", 400)

                        is_in_reset_cooldown = (iteration - last_opacity_reset_iteration) < tau_reset
                        raw_dead_opacity = (gaussians.get_opacity < min_opacity_prune).squeeze()

                        n_protected_by_cooldown = 0
                        n_protected_by_age = 0

                        if is_in_reset_cooldown:
                            # Feature B: Opacity reset cooldown suspends only opacity-based Stage 1 eviction
                            n_protected_by_cooldown = int(raw_dead_opacity.sum().item())
                            dead_opacity_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
                        else:
                            # Feature A: Primitive age grace exempts young Gaussians from opacity eviction
                            if hasattr(gaussians, "birth_iteration") and gaussians.birth_iteration is not None and gaussians.birth_iteration.numel() > 0:
                                is_young = (iteration - gaussians.birth_iteration) < tau_age
                                n_protected_by_age = int((raw_dead_opacity & is_young).sum().item())
                                dead_opacity_mask = raw_dead_opacity & (~is_young)
                            else:
                                dead_opacity_mask = raw_dead_opacity

                        # Physical size/scale hygiene remains unconditionally operational
                        hygiene_mask = dead_opacity_mask
                        if size_threshold is not None:
                            if hasattr(gaussians, "max_radii2D"):
                                hygiene_mask = torch.logical_or(hygiene_mask, gaussians.max_radii2D > size_threshold)
                            max_world_size_ratio = getattr(opt, "max_world_size_ratio", 0.1)
                            big_ws = torch.max(gaussians.get_scaling, dim=1).values > (max_world_size_ratio * scene_extent)
                            hygiene_mask = torch.logical_or(hygiene_mask, big_ws)

                        executed_hygiene_prune = 0
                        if hygiene_mask.any():
                            executed_hygiene_prune = int(hygiene_mask.sum().item())
                            gaussians.prune_points(hygiene_mask)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()

                            # Synchronize all auxiliary and candidate tensors to surviving primitives
                            surviving_1 = ~hygiene_mask
                            clone_mask = clone_mask[surviving_1]
                            split_mask = split_mask[surviving_1]
                            demand = demand[surviving_1]
                            protected_mask = protected_mask[surviving_1]

                        # d. STAGE 2: ADMISSION CONTROL & OPTIONAL PRUNE-FIRST
                        n_clone_req = int(clone_mask.sum().item())
                        n_split_req = int(split_mask.sum().item())

                        prunable_utility = bottom_utility_mask(demand, protected_mask, fraction=0.05)
                        prune_candidates_count = int(prunable_utility.sum().item())

                        decision = admission_ctrl.evaluate_admission(
                            n_clone_requested=n_clone_req,
                            n_split_requested=n_split_req,
                            current_num_gaussians=gaussians.get_xyz.shape[0],
                            prune_candidates_count=prune_candidates_count,
                            inflow_cap_min=inflow_cap_min,
                            inflow_cap_max=inflow_cap_max,
                            target_headroom_mb=target_headroom_mb,
                            kappa_split=kappa_split,
                            inflow_mode=inflow_mode,
                        )

                        dec_action = decision.action if hasattr(decision, "action") else decision["action"]
                        dec_k_prune = decision.required_pre_prune if hasattr(decision, "required_pre_prune") else decision["required_pre_prune"]
                        dec_n_clone = decision.n_clone if hasattr(decision, "n_clone") else decision["n_clone"]
                        dec_n_split = decision.n_split if hasattr(decision, "n_split") else decision["n_split"]
                        allocation_decision = decision
                        executed_pre_prune = 0

                        if dec_action in [AdmissionAction.PRUNE_FIRST, AdmissionAction.SHRINK_AND_PRUNE_FIRST, "PRUNE_FIRST", "SHRINK_AND_PRUNE_FIRST"]:
                            k_prune = dec_k_prune
                            utility_indices = torch.where(prunable_utility)[0]
                            if len(utility_indices) > 0 and k_prune > 0:
                                sorted_by_demand = utility_indices[torch.argsort(demand[utility_indices])]
                                exec_prune_indices = sorted_by_demand[:k_prune]

                                exec_prune_mask = torch.zeros(
                                    gaussians.get_xyz.shape[0], dtype=torch.bool, device=device
                                )
                                exec_prune_mask[exec_prune_indices] = True
                                gaussians.prune_points(exec_prune_mask)
                                executed_pre_prune = int(exec_prune_mask.sum().item())

                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()

                                surviving_2 = ~exec_prune_mask
                                clone_mask = clone_mask[surviving_2]
                                split_mask = split_mask[surviving_2]
                                demand = demand[surviving_2]
                                protected_mask = protected_mask[surviving_2]

                                post_prune_decision = admission_ctrl.evaluate_admission(
                                    n_clone_requested=dec_n_clone,
                                    n_split_requested=dec_n_split,
                                    current_num_gaussians=gaussians.get_xyz.shape[0],
                                    prune_candidates_count=0,
                                    inflow_cap_min=inflow_cap_min,
                                    inflow_cap_max=inflow_cap_max,
                                    target_headroom_mb=target_headroom_mb,
                                    kappa_split=kappa_split,
                                    inflow_mode=inflow_mode,
                                )
                                dec_n_clone = post_prune_decision.n_clone
                                dec_n_split = post_prune_decision.n_split
                                allocation_decision = post_prune_decision

                        # e. ALLOCATION (Joint Knapsack or Quota)
                        remaining_population_cap = max(
                            0, max_gaussians_cap - int(gaussians.get_xyz.shape[0])
                        )
                        if inflow_mode == "dynamic" and hasattr(allocation_decision, "dynamic_inflow") and allocation_decision.dynamic_inflow:
                            effective_inflow_cap = allocation_decision.dynamic_inflow.get("computed_cap", gaussians_per_iter_cap)
                        else:
                            effective_inflow_cap = gaussians_per_iter_cap

                        mutation_item_cap = min(
                            effective_inflow_cap,
                            remaining_population_cap,
                        )
                        if hasattr(allocation_decision, "transient_cost_bytes"):
                            final_clone_mask, final_split_mask = allocator.joint_knapsack_allocate(
                                demand=demand,
                                clone_mask=clone_mask,
                                split_mask=split_mask,
                                max_transient_bytes=allocation_decision.mutation_budget_bytes,
                                strategy=allocator_strategy,
                                max_items=mutation_item_cap,
                                kappa_split=kappa_split,
                            )
                        else:
                            final_clone_mask, final_split_mask = allocator.allocate(
                                demand=demand,
                                clone_mask=clone_mask,
                                split_mask=split_mask,
                                n_clone_quota=dec_n_clone,
                                n_split_quota=dec_n_split,
                            )

                        # f. EXECUTE DENSIFICATION (Clone first, then Split)
                        if final_clone_mask.sum() > 0:
                            gaussians.densify_and_clone(final_clone_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)
                        if final_split_mask.sum() > 0:
                            gaussians.densify_and_split(final_split_mask, opt.densify_grad_threshold, scene_extent, iteration=iteration)

                        mutation_memory_after = capture_cuda_memory()
                        alloc_stats = getattr(allocator, "last_allocation_stats", {})
                        pop_before = int(mutation_memory_before.num_gaussians) if hasattr(mutation_memory_before, "num_gaussians") else int(mutation_memory_before.get("num_gaussians", 0)) if isinstance(mutation_memory_before, dict) else int(gaussians.get_xyz.shape[0])
                        pop_after = int(gaussians.get_xyz.shape[0])
                        n_newborn = int(final_clone_mask.sum().item()) + int(final_split_mask.sum().item())
                        if telemetry_writer is not None:
                            inflow_data = getattr(decision, "dynamic_inflow", None) or {}
                            if not inflow_data and hasattr(allocation_decision, "dynamic_inflow"):
                                inflow_data = getattr(allocation_decision, "dynamic_inflow", None) or {}
                            telemetry_writer.write(
                                "mutation_cycle",
                                iteration=iteration,
                                population_before=pop_before,
                                population_after=pop_after,
                                n_newborn=n_newborn,
                                n_protected_by_age=n_protected_by_age,
                                n_protected_by_cooldown=n_protected_by_cooldown,
                                decision=decision.to_dict() if hasattr(decision, "to_dict") else decision,
                                executed_pre_prune=executed_pre_prune,
                                executed_hygiene_prune=executed_hygiene_prune,
                                n_clone_executed=int(final_clone_mask.sum().item()),
                                n_split_executed=int(final_split_mask.sum().item()),
                                requested_split_share=alloc_stats.get("requested_split_share", 0.0),
                                actual_split_share=alloc_stats.get("actual_split_share", 0.0),
                                tau_age=tau_age,
                                tau_reset=tau_reset,
                                kappa_split=kappa_split,
                                memory_before=mutation_memory_before.to_dict(),
                                memory_after=mutation_memory_after.to_dict(),
                                raster_stats_available=bool(render_pkg.get("stats_available", False)),
                                allocator_strategy=allocator_strategy,
                                occlusion_mode=occlusion_mode,
                                structure_protection=enable_structure_protection,
                                mutation_item_cap=mutation_item_cap,
                                inflow_mode=inflow_mode,
                                current_memory_footprint_mb=inflow_data.get("current_footprint_mb", round(mutation_memory_before.conservative_footprint_bytes / (1024**2), 2)),
                                available_headroom_mb=inflow_data.get("available_headroom_mb", 0.0),
                                estimated_bytes_per_clone=inflow_data.get("estimated_bytes_per_clone", 0),
                                estimated_bytes_per_split=inflow_data.get("estimated_bytes_per_split", 0),
                                computed_dynamic_cap=int(effective_inflow_cap),
                                requested_candidates=int(n_clone_req + n_split_req),
                                accepted_candidates=int(final_clone_mask.sum().item() + final_split_mask.sum().item()),
                                rejected_by_cap=max(0, int(n_clone_req + n_split_req) - int(effective_inflow_cap)),
                                rejected_by_memory_budget=inflow_data.get("rejected_by_budget", 0),
                                controller_budget_violations=0,
                            )

                    # h. Reset Accumulators for New Population
                    new_pts = gaussians.get_xyz.shape[0]
                    occ_engine.reset_buffers(new_pts)
                    accum_visibility = torch.zeros(new_pts, dtype=torch.int32, device=device)
                    accum_pixel_footprint = torch.zeros(new_pts, dtype=torch.int32, device=device)
                    accum_transmittance = torch.zeros(new_pts, dtype=torch.float32, device=device)
                    accum_depth_var = torch.zeros(new_pts, dtype=torch.float32, device=device)
                    viewspace_point_tensor_grad = torch.zeros(new_pts, device=device)

                # Periodic opacity reset
                if iteration % opt.opacity_reset_interval == 0 or (
                    getattr(dataset, "white_background", False) and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
                    last_opacity_reset_iteration = iteration

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if opacity_optimizer is not None:
                opacity_optimizer.step()
                opacity_optimizer.zero_grad(set_to_none=True)

            # Free per-iteration tensors
            render_pkg = None
            image = None
            loss = None
            Ll1 = None
            ssim_value = None
            gt_image = None

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                if scene is not None:
                    scene.save(iteration)
                elif hasattr(gaussians, "save_ply") and dataset.model_path:
                    pc_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{iteration}")
                    os.makedirs(pc_path, exist_ok=True)
                    gaussians.save_ply(os.path.join(pc_path, "point_cloud.ply"))
                else:
                    print(f"[ITER {iteration}] Checkpoint saved with {gaussians.get_xyz.shape[0]:,} Gaussians.")

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_dir = scene.model_path if scene is not None else dataset.model_path
                if save_dir and hasattr(gaussians, "capture"):
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save((gaussians.capture(), iteration), os.path.join(save_dir, f"chkpnt{iteration}.pth"))

    if opt.iterations not in saving_iterations:
        print(f"\n[ZeroGS] Saving final model at iteration {opt.iterations}")
        if scene is not None:
            scene.save(opt.iterations)
        elif hasattr(gaussians, "save_ply") and dataset.model_path:
            pc_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{opt.iterations}")
            os.makedirs(pc_path, exist_ok=True)
            gaussians.save_ply(os.path.join(pc_path, "point_cloud.ply"))

    print("\n[ZeroGS] Training completed successfully.")
    print(f"[ZeroGS] Telemetry: {admission_ctrl.get_telemetry()}")
    if telemetry_writer is not None:
        telemetry_writer.write(
            "training_end",
            final_gaussians=int(gaussians.get_xyz.shape[0]),
            admission=admission_ctrl.get_telemetry(),
            memory=capture_cuda_memory().to_dict(),
        )


def build_parser():
    parser = ArgumentParser(description="ZeroGS Training Engine")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # ZeroGS Budget & Occlusion Runtime Parameters
    parser.add_argument("--hard_vram_limit_mb", type=float, default=8192.0)
    parser.add_argument("--safety_headroom_mb", type=float, default=512.0)
    parser.add_argument("--non_pytorch_allowance_mb", type=float, default=256.0)
    parser.add_argument("--gamma_occ", type=float, default=0.35)
    parser.add_argument("--transmittance_tau", type=float, default=0.25)
    parser.add_argument(
        "--occlusion_mode",
        choices=["full", "transmittance_only", "depth_only", "none"],
        default="full",
    )
    parser.add_argument("--disable_structure_protection", action="store_true", default=False)
    parser.add_argument(
        "--occlusion_physics_mode",
        choices=["corrected", "legacy", "none"],
        default="corrected",
        help="Occlusion depth variance physics mode: 'corrected' (dimensionless multi-view depth variance), 'legacy' (exact pre-remediation within+between sum for ablation), or 'none'",
    )
    parser.add_argument(
        "--allocator_strategy",
        choices=["gain_per_byte", "demand_only", "random"],
        default="gain_per_byte",
    )
    parser.add_argument(
        "--pruning_mode",
        choices=["decoupled", "legacy_as_is"],
        default="decoupled",
        help="Pruning lifecycle mode: 'decoupled' (Stage 1 Hygiene + Stage 2 Admission) or 'legacy_as_is' (A0 ablation baseline)",
    )
    parser.add_argument(
        "--gradient_mask_mode",
        choices=["canonical", "contribution_aware"],
        default="canonical",
        help="Gradient densification accumulator mask mode: 'canonical' (Vanilla 3DGS visibility_filter/radii>0 for both numerator and denominator) or 'contribution_aware' (cur_vis_count>0 for both)",
    )
    parser.add_argument("--enable_zero_oom", action="store_true", default=True)
    parser.add_argument("--disable_zero_oom", action="store_false", dest="enable_zero_oom")
    parser.add_argument("--allow_missing_native_stats", action="store_true", default=False)
    parser.add_argument("--simulation_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--telemetry_jsonl", type=str, default=None)
    parser.add_argument("--cost_calibration", type=str, default=None)
    parser.add_argument("--experiment_method", type=str, default="ZeroGS")
    parser.add_argument("--experiment_scene", type=str, default=None)
    parser.add_argument("--num_init_points", type=int, default=5000)

    return parser, lp, op, pp


if __name__ == "__main__":
    parser, lp, op, pp = build_parser()
    args = parser.parse_args()

    safe_state(args.quiet, seed=args.seed)

    # Extract grouped arguments
    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    # Attach custom ZeroGS arguments to opt namespace
    opt_args.hard_vram_limit_mb = args.hard_vram_limit_mb
    opt_args.safety_headroom_mb = args.safety_headroom_mb
    opt_args.non_pytorch_allowance_mb = args.non_pytorch_allowance_mb
    opt_args.gamma_occ = args.gamma_occ
    opt_args.transmittance_tau = args.transmittance_tau
    opt_args.occlusion_mode = args.occlusion_mode
    opt_args.occlusion_physics_mode = args.occlusion_physics_mode
    opt_args.enable_structure_protection = not args.disable_structure_protection
    opt_args.allocator_strategy = args.allocator_strategy
    opt_args.enable_zero_oom = args.enable_zero_oom
    opt_args.pruning_mode = args.pruning_mode
    opt_args.gradient_mask_mode = args.gradient_mask_mode
    opt_args.require_native_stats = not args.allow_missing_native_stats
    opt_args.simulation_mode = args.simulation_mode
    opt_args.num_init_points = args.num_init_points
    opt_args.telemetry_jsonl = args.telemetry_jsonl
    opt_args.cost_calibration = args.cost_calibration
    opt_args.inflow_mode = args.inflow_mode
    opt_args.inflow_cap_min = args.inflow_cap_min
    opt_args.inflow_cap_max = args.inflow_cap_max
    opt_args.target_headroom_mb = args.target_headroom_mb
    run_id = uuid.uuid4().hex
    opt_args.run_id = run_id

    git_commit = None
    git_dirty = False
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = bool(git_status_out)
    except Exception:
        pass

    calibration_hash = None
    if getattr(args, "cost_calibration", None) and os.path.exists(args.cost_calibration):
        import hashlib
        try:
            with open(args.cost_calibration, "rb") as f:
                calibration_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "calibration_hash": calibration_hash,
        "evidence_type": "simulation" if args.simulation_mode else "measured",
        "method": args.experiment_method,
        "scene": args.experiment_scene or (
            os.path.basename(os.path.normpath(dataset_args.source_path))
            if dataset_args.source_path else "simulation"
        ),
        "hard_vram_limit_mb": args.hard_vram_limit_mb,
        "seed": args.seed,
        "simulation_mode": args.simulation_mode,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arguments": vars(args),
    }
    status_path = None
    if dataset_args.model_path:
        os.makedirs(dataset_args.model_path, exist_ok=True)
        with open(os.path.join(dataset_args.model_path, "run_manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(run_manifest, manifest_file, indent=2, default=str)
        status_path = os.path.join(dataset_args.model_path, "run_status.json")
        with open(status_path, "w", encoding="utf-8") as status_file:
            json.dump({"run_id": run_id, "status": "running", "oom": False}, status_file, indent=2)

    try:
        training(
            dataset=dataset_args,
            opt=opt_args,
            pipe=pipe_args,
            testing_iterations=args.test_iterations,
            saving_iterations=args.save_iterations,
            checkpoint_iterations=args.checkpoint_iterations,
            checkpoint=args.start_checkpoint,
            debug_from=args.debug_from,
        )
    except torch.cuda.OutOfMemoryError as exc:
        if status_path:
            with open(status_path, "w", encoding="utf-8") as status_file:
                json.dump({"run_id": run_id, "status": "failed", "oom": True, "error": str(exc)}, status_file, indent=2)
        raise
    except VramBudgetExceededError as exc:
        if status_path:
            with open(status_path, "w", encoding="utf-8") as status_file:
                json.dump(
                    {"run_id": run_id, "status": "budget_exceeded", "oom": False, "budget_exceeded": True, "error": str(exc)},
                    status_file,
                    indent=2,
                )
        raise
    except Exception as exc:
        if status_path:
            with open(status_path, "w", encoding="utf-8") as status_file:
                json.dump({"run_id": run_id, "status": "failed", "oom": False, "error": repr(exc)}, status_file, indent=2)
        raise
    else:
        if status_path:
            with open(status_path, "w", encoding="utf-8") as status_file:
                json.dump({"run_id": run_id, "status": "completed", "oom": False}, status_file, indent=2)

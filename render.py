"""
ZeroGS: Rendering Pipeline for Evaluation & Benchmark (render.py).

Renders test and/or train camera views from a trained 3DGS / ZeroGS model
and exports rendered images and ground truth images for metric computation.
"""

import os
import sys
from argparse import ArgumentParser
from typing import List, Optional, Union, Any
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from zerogs.mock_gaussian_model import MockGaussianModel, MockCamera, mock_render


def render_set(
    model_path: str,
    name: str,
    iteration: Optional[int],
    views: list,
    gaussians: Union[GaussianModel, MockGaussianModel],
    pipeline: Any,
    background: torch.Tensor,
    is_simulation: bool = False,
):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    desc = f"Rendering {name} views (iter {iteration})"
    for idx, view in enumerate(tqdm(views, desc=desc)):
        if not is_simulation and isinstance(gaussians, GaussianModel):
            rendering = render(view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :].to(rendering.device)
        else:
            pkg = mock_render(view, gaussians, pipeline, background)
            rendering = pkg["render"]
            gt = getattr(view, "original_image", rendering).to(rendering.device)

        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        file_name = f"{idx:05d}.png"
        torchvision.utils.save_image(rendering, os.path.join(render_path, file_name))
        torchvision.utils.save_image(gt, os.path.join(gts_path, file_name))


def render_sets(
    dataset: Any,
    iteration: int,
    pipeline: Any,
    skip_train: bool,
    skip_test: bool,
    simulation_mode: bool = False,
):
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src = dataset.source_path.strip() if dataset.source_path else ""
        has_valid_scene = bool(src) and (
            os.path.exists(os.path.join(src, "sparse"))
            or os.path.exists(os.path.join(src, "transforms_train.json"))
        )
        is_simulation = simulation_mode or (not has_valid_scene)

        sh_deg = getattr(dataset, "sh_degree", None) or 3
        if not is_simulation:
            gaussians = GaussianModel(sh_deg)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device=device)

            if not skip_train:
                render_set(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    scene.getTrainCameras(),
                    gaussians,
                    pipeline,
                    background,
                    is_simulation=False,
                )

            if not skip_test:
                render_set(
                    dataset.model_path,
                    "test",
                    scene.loaded_iter,
                    scene.getTestCameras(),
                    gaussians,
                    pipeline,
                    background,
                    is_simulation=False,
                )
        else:
            print("[ZeroGS Render] Running in self-contained simulation mode.")
            gaussians = MockGaussianModel(sh_degree=sh_deg, num_points=5000, device=device)
            background = torch.zeros(3, device=device)
            test_views = [MockCamera(camera_id=i, device=device) for i in range(5)]
            train_views = [MockCamera(camera_id=i, device=device) for i in range(5)]

            model_path = dataset.model_path if dataset.model_path else "output/simulation"
            loaded_iter = iteration if iteration > 0 else 30_000

            if not skip_train:
                render_set(
                    model_path,
                    "train",
                    loaded_iter,
                    train_views,
                    gaussians,
                    pipeline,
                    background,
                    is_simulation=True,
                )
            if not skip_test:
                render_set(
                    model_path,
                    "test",
                    loaded_iter,
                    test_views,
                    gaussians,
                    pipeline,
                    background,
                    is_simulation=True,
                )


if __name__ == "__main__":
    parser = ArgumentParser(description="ZeroGS View Rendering Pipeline")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration checkpoint to render")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering train views")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test views")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--simulation_mode", action="store_true", default=False)

    args = get_combined_args(parser)
    print(f"[ZeroGS Render] Rendering target: {args.model_path}")

    dataset_args = model.extract(args)
    pipeline_args = pipeline.extract(args)

    render_sets(
        dataset=dataset_args,
        iteration=args.iteration,
        pipeline=pipeline_args,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        simulation_mode=args.simulation_mode,
    )
    print("[ZeroGS Render] Rendering complete!")

"""
Integration tests for gaussian_renderer.render across diverse execution modes:
- Standard rendering with in-situ statistics extraction
- separate_sh=True (decoupled DC and rest SH features)
- override_color (custom color replacement)
- Anti-aliasing integration
- Viewspace point gradient flow
"""

import pytest
import torch
from argparse import ArgumentParser

from arguments import PipelineParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel


class MockViewpointCamera:
    def __init__(self, H=64, W=64, device="cuda"):
        self.image_height = H
        self.image_width = W
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.world_view_transform = torch.eye(4, device=device)
        self.full_proj_transform = torch.eye(4, device=device)
        self.camera_center = torch.zeros(3, device=device)
        self.image_name = "test_view_001.jpg"


@pytest.fixture
def test_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for rasterizer tests.")

    device = "cuda"
    cam = MockViewpointCamera(H=64, W=64, device=device)
    parser = ArgumentParser()
    pipe = PipelineParams(parser).extract(parser.parse_args([]))
    bg = torch.zeros(3, device=device)

    gm = GaussianModel(sh_degree=3)
    N = 30
    gm._xyz = torch.nn.Parameter(
        torch.randn(N, 3, device=device) * 0.3 + torch.tensor([0.0, 0.0, 2.5], device=device)
    )
    gm._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device=device))
    gm._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device=device))
    gm._opacity = torch.nn.Parameter(torch.ones(N, 1, device=device) * 1.5)
    gm._scaling = torch.nn.Parameter(torch.ones(N, 3, device=device) * (-2.5))
    gm._rotation = torch.nn.Parameter(torch.zeros(N, 4, device=device))
    gm._rotation.data[:, 0] = 1.0

    return cam, gm, pipe, bg, N


def test_standard_render_output_structure(test_setup):
    cam, gm, pipe, bg, N = test_setup

    out = render(cam, gm, pipe, bg)

    required_keys = [
        "render",
        "viewspace_points",
        "visibility_filter",
        "radii",
        "depth",
        "mean_T",
        "depth_var",
        "vis_count",
    ]
    for k in required_keys:
        assert k in out, f"Missing key '{k}' in render output"

    assert out["render"].shape == (3, 64, 64)
    assert out["radii"].shape == (N,)
    assert out["mean_T"] is not None
    assert out["mean_T"].shape == (N,)
    assert out["depth_var"] is not None
    assert out["depth_var"].shape == (N,)
    assert out["vis_count"] is not None
    assert out["vis_count"].shape == (N,)

    # Gradient backward flow
    loss = out["render"].sum()
    loss.backward()
    assert out["viewspace_points"].grad is not None
    assert gm._xyz.grad is not None


def test_separate_sh_rendering(test_setup):
    cam, gm, pipe, bg, N = test_setup

    # Must succeed without TypeError or UnboundLocalError
    out = render(cam, gm, pipe, bg, separate_sh=True)
    assert out["render"].shape == (3, 64, 64)
    assert out["radii"].shape == (N,)


def test_override_color_rendering(test_setup):
    cam, gm, pipe, bg, N = test_setup
    device = "cuda"

    custom_color = torch.rand(N, 3, device=device)
    out = render(cam, gm, pipe, bg, override_color=custom_color)
    assert out["render"].shape == (3, 64, 64)

    # Test separate_sh with override_color simultaneously
    out_sep_override = render(cam, gm, pipe, bg, separate_sh=True, override_color=custom_color)
    assert out_sep_override["render"].shape == (3, 64, 64)


def test_antialiasing_rendering(test_setup):
    cam, gm, pipe, bg, N = test_setup

    out = render(cam, gm, pipe, bg, use_antialiasing=True)
    assert out["render"].shape == (3, 64, 64)

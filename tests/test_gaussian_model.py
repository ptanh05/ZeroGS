import pytest
import torch
from argparse import ArgumentParser
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel


def test_gaussian_model_capture_and_restore_without_optimizer():
    """Verify capture() and restore() do not throw AttributeError when self.optimizer is None."""
    model = GaussianModel(sh_degree=3)
    assert model.optimizer is None

    # capture() should succeed and store None for optimizer state
    cap = model.capture()
    assert len(cap) == 12
    assert cap[10] is None

    # restore() should succeed without requiring training_args or an existing optimizer
    restored = GaussianModel(sh_degree=3)
    restored.restore(cap)
    assert restored.optimizer is None


def test_gaussian_model_attributes_initialized():
    """Ensure exposure and tracking attributes are safely initialized on construction."""
    model = GaussianModel(sh_degree=3)
    assert hasattr(model, "_exposure")
    assert hasattr(model, "exposure_mapping")
    assert hasattr(model, "pretrained_exposures")
    assert hasattr(model, "tmp_radii")
    assert model.tmp_radii is None
    assert model.exposure_optimizer is None


def test_gaussian_model_training_setup_and_reset_opacity():
    """Verify training_setup and reset_opacity work even before optimizer steps."""
    parser = ArgumentParser()
    opt = OptimizationParams(parser)
    args = opt.extract(parser.parse_args([]))

    model = GaussianModel(sh_degree=3)
    N = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device=device))
    model._opacity = torch.nn.Parameter(torch.randn(N, 1, device=device))
    model._scaling = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._rotation = torch.nn.Parameter(torch.randn(N, 4, device=device))
    model.max_radii2D = torch.zeros(N, device=device)

    # training_setup should not crash even if _exposure is empty
    model.training_setup(args)
    assert model.optimizer is not None

    # reset_opacity before any optimizer step must not crash with TypeError
    model.reset_opacity()
    assert model._opacity.shape[0] == N


def test_gaussian_model_prune_points_without_tmp_radii():
    """Verify prune_points handles None tmp_radii and empty optimizer state gracefully."""
    model = GaussianModel(sh_degree=3)
    N = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device=device))
    model._opacity = torch.nn.Parameter(torch.randn(N, 1, device=device))
    model._scaling = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._rotation = torch.nn.Parameter(torch.randn(N, 4, device=device))
    model.max_radii2D = torch.zeros(N, device=device)
    model.xyz_gradient_accum = torch.zeros((N, 1), device=device)
    model.denom = torch.zeros((N, 1), device=device)

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[:5] = True
    model.prune_points(mask)

    assert model.get_xyz.shape[0] == 25
    assert model.tmp_radii is None


def test_gaussian_model_densify_clone_and_split():
    """Verify densify_and_clone and densify_and_split execute safely without tmp_radii."""
    parser = ArgumentParser()
    opt = OptimizationParams(parser)
    args = opt.extract(parser.parse_args([]))

    model = GaussianModel(sh_degree=3)
    N = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device=device))
    model._opacity = torch.nn.Parameter(torch.randn(N, 1, device=device))
    model._scaling = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._rotation = torch.nn.Parameter(torch.randn(N, 4, device=device))
    model.max_radii2D = torch.zeros(N, device=device)
    model.training_setup(args)

    # Boolean mask clone
    clone_mask = torch.zeros(N, dtype=torch.bool, device=device)
    clone_mask[:5] = True
    model.densify_and_clone(clone_mask, 0.0002, 1.0)
    assert model.get_xyz.shape[0] == 45

    # Boolean mask split (split 2 Gaussians into 2 each, removing 2)
    split_mask = torch.zeros(model.get_xyz.shape[0], dtype=torch.bool, device=device)
    split_mask[:2] = True
    prev_count = model.get_xyz.shape[0]
    model.densify_and_split(split_mask, 0.0002, 1.0, N=2)
    # prev_count + 4 children - 2 parents = prev_count + 2
    assert model.get_xyz.shape[0] == prev_count + 2


def test_gaussian_model_get_exposure_fallback():
    """Verify get_exposure_from_name safely falls back to identity matrix on unknown cameras."""
    model = GaussianModel(sh_degree=3)
    exp = model.get_exposure_from_name("unknown_camera_001.jpg")
    assert exp.shape == (3, 4)
    expected_eye = torch.eye(3, 4, device=exp.device)
    assert torch.allclose(exp, expected_eye)


def test_gaussian_model_spatial_lr_scale_type():
    """Verify spatial_lr_scale and percent_dense are floats and accept float assignments."""
    model = GaussianModel(sh_degree=3)
    assert isinstance(model.spatial_lr_scale, float)
    assert isinstance(model.percent_dense, float)
    model.spatial_lr_scale = 2.5
    assert model.spatial_lr_scale == 2.5


def test_gaussian_model_densify_via_policy_safe_when_none():
    """Verify _densify_via_policy does not crash when policy_network is None."""
    model = GaussianModel(sh_degree=3)
    assert model.policy_network is None
    # Calling _densify_via_policy with policy_network as None should return safely
    grads = torch.zeros((10, 1))
    model._densify_via_policy(grads, extent=1.0)


def test_gaussian_model_densify_via_policy_execution():
    """Verify _densify_via_policy executes both SPLIT and MERGE safely with tensor consistency."""
    from scene.policy_network import SplitPolicyNetwork

    parser = ArgumentParser()
    opt = OptimizationParams(parser)
    args = opt.extract(parser.parse_args([]))

    model = GaussianModel(sh_degree=3)
    N = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device=device))
    model._opacity = torch.nn.Parameter(torch.randn(N, 1, device=device))
    model._scaling = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._rotation = torch.nn.Parameter(torch.randn(N, 4, device=device))
    model.max_radii2D = torch.zeros(N, device=device)
    model.xyz_gradient_accum = torch.zeros((N, 1), device=device)
    model.denom = torch.zeros((N, 1), device=device)
    model.training_setup(args)

    net = SplitPolicyNetwork().to(device)
    model.policy_network = net
    model.use_policy_network = True

    grads = torch.rand((N, 1), device=device) * 0.01
    model._densify_via_policy(grads, extent=1.0)

    n_final = model.get_xyz.shape[0]
    assert model._xyz.shape[0] == n_final
    assert model._opacity.shape[0] == n_final
    assert model._scaling.shape[0] == n_final
    assert model.max_radii2D.shape[0] == n_final
    assert model.xyz_gradient_accum.shape[0] == n_final
    assert model.denom.shape[0] == n_final




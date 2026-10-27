import pytest
import torch

from arguments import parse_namespace_config
from train import compute_photometric_loss, resolve_simulation_mode


def test_photometric_loss_uses_l1_and_dssim():
    image = torch.zeros(1, 3, 16, 16, requires_grad=True)
    ground_truth = torch.ones_like(image)

    loss_l1, l1_value, ssim_value = compute_photometric_loss(image, ground_truth, 0.0)
    loss_dssim, _, _ = compute_photometric_loss(image, ground_truth, 1.0)
    mixed, _, _ = compute_photometric_loss(image, ground_truth, 0.2)

    assert torch.allclose(loss_l1, l1_value)
    assert torch.allclose(loss_dssim, 1.0 - ssim_value)
    assert torch.allclose(mixed, 0.8 * l1_value + 0.2 * (1.0 - ssim_value))
    mixed.backward()
    assert image.grad is not None


def test_photometric_loss_rejects_invalid_weight():
    image = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError):
        compute_photometric_loss(image, image, 1.1)


def test_simulation_requires_explicit_opt_in(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_simulation_mode(str(tmp_path), simulation_mode=False)
    assert resolve_simulation_mode(str(tmp_path), simulation_mode=True) is True


def test_real_scene_disables_simulation(tmp_path):
    (tmp_path / "sparse").mkdir()
    assert resolve_simulation_mode(str(tmp_path), simulation_mode=True) is False


def test_namespace_config_parser_rejects_code_execution():
    parsed = parse_namespace_config("Namespace(sh_degree=3, eval=True, values=[1, 2])")
    assert parsed.sh_degree == 3
    assert parsed.eval is True
    assert parsed.values == [1, 2]

    with pytest.raises(ValueError):
        parse_namespace_config("__import__('os').system('echo unsafe')")

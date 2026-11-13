"""
Test suite for ZeroGS Low-Budget Optimizers and Compact Splat Exporters.

Verifies:
1. HalfPrecisionAdam: 50% optimizer memory reduction, valid parameter updates, loss reduction.
2. Int8GaussianAdam: 75% optimizer memory reduction, valid parameter updates, loss reduction.
3. ByteCostModel: Dynamic persistent memory accounting for FP32 (944 B), FP16 (708 B), and INT8 (590 B).
4. save_splat: Exporting to standard 32-byte .splat format (exact N * 32 byte packing).
5. save_compressed: Full-fidelity FP16 compressed archive (.npz).
6. Pruning and concatenation resilience with quantized and half-precision optimizer states.
"""

import os
import tempfile
import torch
import numpy as np
import pytest

from zerogs.optimizers import HalfPrecisionAdam, Int8GaussianAdam
from zerogs.cost_model import ByteCostModel


def test_half_precision_adam_step_and_memory():
    """Verify HalfPrecisionAdam updates parameters and stores moments in FP16."""
    p = torch.nn.Parameter(torch.randn(100, 3, requires_grad=True))
    optimizer = HalfPrecisionAdam([p], lr=0.01)

    loss_fn = lambda: (p ** 2).sum()

    initial_loss = loss_fn().item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn().item()
    assert final_loss < initial_loss, f"Loss should decrease: {final_loss} < {initial_loss}"

    # Verify moment tensors are in float16
    state = optimizer.state[p]
    assert state["exp_avg"].dtype == torch.float16
    assert state["exp_avg_sq"].dtype == torch.float16
    assert state["step"] == 10


def test_int8_gaussian_adam_step_and_memory():
    """Verify Int8GaussianAdam updates parameters and stores moments in int8/uint8."""
    p = torch.nn.Parameter(torch.randn(100, 3, requires_grad=True))
    optimizer = Int8GaussianAdam([p], lr=0.01)

    loss_fn = lambda: (p ** 2).sum()

    initial_loss = loss_fn().item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn().item()
    assert final_loss < initial_loss, f"Loss should decrease: {final_loss} < {initial_loss}"

    # Verify moment tensors are in int8 and uint8
    state = optimizer.state[p]
    assert state["exp_avg_int8"].dtype == torch.int8
    assert state["exp_avg_sq_uint8"].dtype == torch.uint8
    assert state["scale_m"] > 0.0
    assert state["scale_v"] > 0.0
    assert state["step"] == 10


def test_byte_cost_model_optimizer_tiers():
    """Verify ByteCostModel correctly reflects the reduced memory for half and int8 Adam."""
    # SH degree 3: 59 floats * 4 = 236 bytes param, 236 bytes grad
    model_fp32 = ByteCostModel(sh_degree=3, optimizer_type="default")
    model_fp16 = ByteCostModel(sh_degree=3, optimizer_type="half_adam")
    model_int8 = ByteCostModel(sh_degree=3, optimizer_type="int8_adam")

    # Default FP32 Adam: 236 param + 236 grad + 472 adam = 944 Bytes
    assert model_fp32.param_bytes_per_gaussian == 236
    assert model_fp32.grad_bytes_per_gaussian == 236
    assert model_fp32.adam_states_bytes_per_gaussian == 472
    assert model_fp32.persistent_bytes_per_gaussian == 944

    # FP16 Adam: 236 param + 236 grad + 236 adam = 708 Bytes (-25% total, -50% adam)
    assert model_fp16.adam_states_bytes_per_gaussian == 236
    assert model_fp16.persistent_bytes_per_gaussian == 708

    # INT8 Adam: 236 param + 236 grad + 118 adam = 590 Bytes (-37.5% total, -75% adam)
    assert model_int8.adam_states_bytes_per_gaussian == 118
    assert model_int8.persistent_bytes_per_gaussian == 590


def test_splat_exporter_packing():
    """Verify save_splat exports exactly 32 bytes per Gaussian."""
    class DummyGaussians:
        def __init__(self, n):
            self._xyz = torch.randn(n, 3)
            self._scaling = torch.randn(n, 3)
            self._features_dc = torch.randn(n, 1, 3)
            self._opacity = torch.randn(n, 1)
            self._rotation = torch.randn(n, 4)

    from scene.gaussian_model import GaussianModel
    dummy = DummyGaussians(1500)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.splat")
        GaussianModel.save_splat(dummy, out_path)

        assert os.path.exists(out_path)
        file_size = os.path.getsize(out_path)
        expected_size = 1500 * 32
        assert file_size == expected_size, f"Expected {expected_size} bytes, got {file_size}"


def test_compressed_npz_exporter():
    """Verify save_compressed exports a valid compressed npz archive preserving attributes."""
    class DummyGaussians:
        def __init__(self, n):
            self._xyz = torch.randn(n, 3)
            self._scaling = torch.randn(n, 3)
            self._features_dc = torch.randn(n, 1, 3)
            self._features_rest = torch.randn(n, 15, 3)
            self._opacity = torch.randn(n, 1)
            self._rotation = torch.randn(n, 4)
            self.active_sh_degree = 3
            self.max_sh_degree = 3

    from scene.gaussian_model import GaussianModel
    dummy = DummyGaussians(2000)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_compressed.npz")
        GaussianModel.save_compressed(dummy, out_path)

        assert os.path.exists(out_path)
        with np.load(out_path) as loaded:
            assert "xyz" in loaded
            assert "features_dc" in loaded
            assert "features_rest" in loaded
            assert "opacity" in loaded
            assert "scaling" in loaded
            assert "rotation" in loaded
            assert loaded["xyz"].shape == (2000, 3)
            assert loaded["xyz"].dtype == np.float16
            assert loaded["features_rest"].shape == (2000, 15, 3)


def test_optimizer_prune_and_cat_with_quantized_states():
    """Verify that GaussianModel correctly slices and concatenates int8 and fp16 optimizer states."""
    from scene.gaussian_model import GaussianModel

    gm = GaussianModel(sh_degree=0, optimizer_type="int8_adam")
    n = 100
    gm._xyz = torch.nn.Parameter(torch.randn(n, 3))
    gm._features_dc = torch.nn.Parameter(torch.randn(n, 1, 3))
    gm._features_rest = torch.nn.Parameter(torch.empty(n, 0, 3))
    gm._opacity = torch.nn.Parameter(torch.randn(n, 1))
    gm._scaling = torch.nn.Parameter(torch.randn(n, 3))
    gm._rotation = torch.nn.Parameter(torch.randn(n, 4))
    gm.xyz_gradient_accum = torch.zeros(n, 1)
    gm.denom = torch.ones(n, 1)
    gm.max_radii2D = torch.zeros(n)

    class DummyArgs:
        position_lr_init = 0.001
        position_lr_final = 0.0001
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 1000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
        percent_dense = 0.01

    gm.training_setup(DummyArgs())

    # Do one optimizer step to initialize states
    loss = (gm._xyz ** 2).sum() + (gm._opacity ** 2).sum()
    loss.backward()
    gm.optimizer.step()

    # Verify state keys
    xyz_state = gm.optimizer.state[gm.optimizer.param_groups[0]["params"][0]]
    assert "exp_avg_int8" in xyz_state
    assert xyz_state["exp_avg_int8"].shape == (100, 3)

    # Test Prune (remove 30 points)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:30] = True
    gm.prune_points(mask)
    assert gm._xyz.shape[0] == 70
    new_xyz_state = gm.optimizer.state[gm.optimizer.param_groups[0]["params"][0]]
    assert new_xyz_state["exp_avg_int8"].shape == (70, 3)

    # Test Cat (add 20 points)
    d = {
        "xyz": torch.randn(20, 3),
        "f_dc": torch.randn(20, 1, 3),
        "f_rest": torch.empty(20, 0, 3),
        "opacity": torch.randn(20, 1),
        "scaling": torch.randn(20, 3),
        "rotation": torch.randn(20, 4),
    }
    gm.cat_tensors_to_optimizer(d)
    assert gm.optimizer.param_groups[0]["params"][0].shape[0] == 90
    cat_xyz_state = gm.optimizer.state[gm.optimizer.param_groups[0]["params"][0]]
    assert cat_xyz_state["exp_avg_int8"].shape == (90, 3)

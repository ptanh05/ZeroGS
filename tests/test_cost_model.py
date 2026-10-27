"""
Unit tests for ByteCostModel (Module 1).
"""

import pytest
import json
from zerogs.cost_model import ByteCostModel


def test_byte_cost_model_constants_sh3():
    # SH degree 3: 3*16 = 48 SH floats + 11 base floats = 59 floats
    # 59 * 4 bytes = 236 bytes/gaussian
    model = ByteCostModel(sh_degree=3, use_adam=True)
    assert model.total_floats_per_gaussian == 59
    assert model.param_bytes_per_gaussian == 236
    assert model.grad_bytes_per_gaussian == 236
    assert model.adam_states_bytes_per_gaussian == 472
    # Persistent: 236 + 236 + 472 = 944 bytes/gaussian
    assert model.persistent_bytes_per_gaussian == 944


def test_byte_cost_model_constants_sh0():
    # SH degree 0: 3*1 = 3 floats + 11 base = 14 floats
    # 14 * 4 = 56 bytes/gaussian
    model = ByteCostModel(sh_degree=0, use_adam=True)
    assert model.total_floats_per_gaussian == 14
    assert model.param_bytes_per_gaussian == 56
    assert model.persistent_bytes_per_gaussian == 56 * 4  # param + grad + 2*adam = 224


def test_transient_spike_split_higher_than_clone():
    # Crucial validation of Gap P1: For the exact same count of primitives,
    # Split transient spike must be significantly higher than Clone
    model = ByteCostModel(sh_degree=3, clone_multiplier=1.15, split_multiplier=2.30)
    delta_n = 10000

    clone_transient = model.estimate_transient_bytes(n_clone=delta_n, n_split=0, include_padding=False)
    split_transient = model.estimate_transient_bytes(n_clone=0, n_split=delta_n, include_padding=False)

    assert split_transient > clone_transient
    ratio = split_transient / clone_transient
    assert pytest.approx(ratio, 0.01) == 2.30 / 1.15  # ~2.0x higher spike!


def test_prune_freed_bytes():
    model = ByteCostModel(sh_degree=3)
    n_prune = 5000
    freed = model.estimate_prune_freed_bytes(n_prune)
    assert freed == 5000 * 944


def test_affine_calibration_fits_fixed_and_variable_cost(tmp_path):
    model = ByteCostModel(sh_degree=3, cuda_padding_bytes=0)
    persistent = model.persistent_bytes_per_gaussian
    rows = []
    for count in (10_000, 20_000, 40_000):
        rows.append({
            "delta_n": count,
            "clone_spike_mb": (8 * 1024**2 + count * persistent * 1.25) / 1024**2,
            "split_spike_mb": (16 * 1024**2 + count * persistent * 2.5) / 1024**2,
        })
    fitted = model.fit_profile_results(rows)
    assert fitted["clone_multiplier"] == pytest.approx(1.25)
    assert fitted["split_multiplier"] == pytest.approx(2.5)
    assert fitted["clone_fixed_overhead_bytes"] == pytest.approx(8 * 1024**2, abs=2)
    assert fitted["split_fixed_overhead_bytes"] == pytest.approx(16 * 1024**2, abs=2)
    assert model.profile_error(rows)["combined_mape_pct"] == pytest.approx(0.0, abs=1e-5)

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(model.calibration_dict()), encoding="utf-8")
    restored = ByteCostModel.from_calibration_file(str(path))
    assert restored.estimate_transient_bytes(10_000, 0, False) == model.estimate_transient_bytes(10_000, 0, False)

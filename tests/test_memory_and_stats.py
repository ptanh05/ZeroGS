import json

import pytest
import torch

from gaussian_renderer import validate_physical_stats
from zerogs.admission_controller import AdmissionAction, AdmissionController, VramBudgetExceededError
from zerogs.cost_model import ByteCostModel
from zerogs.memory_telemetry import JsonlTelemetryWriter, capture_cuda_memory


def test_reserved_memory_controls_admission_headroom():
    controller = AdmissionController(
        ByteCostModel(),
        hard_vram_limit_mb=100.0,
        safety_headroom_mb=10.0,
    )
    decision = controller.evaluate_admission(
        n_clone_requested=1000,
        n_split_requested=1000,
        current_num_gaussians=10_000,
        prune_candidates_count=0,
        current_allocated_bytes=20 * 1024 * 1024,
        current_reserved_bytes=89 * 1024 * 1024,
    )
    assert decision.action != AdmissionAction.ACCEPT
    assert decision.stats["current_reserved_mb"] == 89.0


def test_non_pytorch_allowance_reduces_headroom():
    controller = AdmissionController(
        ByteCostModel(),
        hard_vram_limit_mb=100.0,
        safety_headroom_mb=0.0,
        non_pytorch_allowance_mb=25.0,
    )
    decision = controller.evaluate_admission(
        n_clone_requested=0,
        n_split_requested=0,
        current_num_gaussians=1,
        prune_candidates_count=0,
        current_allocated_bytes=50 * 1024 * 1024,
        current_reserved_bytes=60 * 1024 * 1024,
    )
    assert decision.available_headroom_bytes == 15 * 1024 * 1024


def test_cpu_memory_snapshot_is_well_formed():
    snapshot = capture_cuda_memory()
    assert snapshot.conservative_footprint_bytes >= snapshot.allocated_bytes
    assert "conservative_footprint_mb" in snapshot.to_dict()


def test_jsonl_telemetry_is_append_only(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = JsonlTelemetryWriter(str(path))
    writer.write("first", value=1)
    writer.write("second", value=2)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["first", "second"]

    replacement = JsonlTelemetryWriter(str(path), truncate=True)
    replacement.write("replacement")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["replacement"]


def test_physical_stats_validation():
    validate_physical_stats(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 0.2]),
        torch.tensor([0, 3]),
        2,
    )
    with pytest.raises(RuntimeError):
        validate_physical_stats(
            torch.tensor([1.2, 0.5]),
            torch.tensor([0.0, 0.2]),
            torch.tensor([0, 3]),
            2,
        )


def test_budget_guard_raises_for_existing_footprint(monkeypatch):
    from zerogs import admission_controller as module
    from zerogs.memory_telemetry import CudaMemorySnapshot

    monkeypatch.setattr(
        module,
        "capture_cuda_memory",
        lambda: CudaMemorySnapshot(allocated_bytes=100 * 1024**2, reserved_bytes=110 * 1024**2),
    )
    controller = AdmissionController(
        ByteCostModel(), hard_vram_limit_mb=100.0, safety_headroom_mb=20.0
    )
    with pytest.raises(VramBudgetExceededError):
        controller.assert_current_memory_within_budget()


def test_budget_guard_allows_using_safety_zone_without_crossing_hard_cap(monkeypatch):
    from zerogs import admission_controller as module
    from zerogs.memory_telemetry import CudaMemorySnapshot

    monkeypatch.setattr(
        module,
        "capture_cuda_memory",
        lambda: CudaMemorySnapshot(allocated_bytes=85 * 1024**2, reserved_bytes=90 * 1024**2),
    )
    controller = AdmissionController(
        ByteCostModel(), hard_vram_limit_mb=100.0, safety_headroom_mb=20.0
    )
    snapshot = controller.assert_current_memory_within_budget()
    assert snapshot["reserved_mb"] == 90.0

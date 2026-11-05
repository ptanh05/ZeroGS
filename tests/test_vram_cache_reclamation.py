"""
Unit tests for Safe VRAM Cache Reclamation in AdmissionController.
Verifies Tests A through E:
- Test A: Normal footprint below budget (no empty_cache, no raise)
- Test B: Recoverable cache pool (footprint > budget, empty_cache reclaims, proceeds, cache_reclaimed event)
- Test C: Unrecoverable memory (footprint > budget even after empty_cache, raises VramBudgetExceededError)
- Test D: Active allocations exceed budget (allocated > budget, cannot be reclaimed, raises)
- Test E: CUDA unavailable / CPU environment (graceful handling, no crash)
"""

import pytest
import torch
from typing import Dict, Any, List

from zerogs.cost_model import ByteCostModel
from zerogs.admission_controller import (
    AdmissionController,
    VramBudgetExceededError,
)
from zerogs.memory_telemetry import CudaMemorySnapshot


class MockTelemetryWriter:
    """Mock telemetry writer capturing structured events for verification."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def write(self, event: str, **kwargs):
        record = {"event": event, **kwargs}
        self.events.append(record)

    def find_events(self, event_name: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["event"] == event_name]


def test_a_normal_footprint_below_budget(monkeypatch):
    """
    Test A — Normal:
    Footprint is well below hard budget.
    Must NOT call empty_cache and must NOT raise any error.
    """
    empty_cache_called = False

    def mock_empty_cache():
        nonlocal empty_cache_called
        empty_cache_called = True

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from zerogs import admission_controller as ac_module

    # Snapshot: allocated = 2000 MB, reserved = 3000 MB. Hard budget = 6144 MB.
    # Footprint = 3000 MB + 256 MB = 3256 MB <= 6144 MB
    monkeypatch.setattr(
        ac_module,
        "capture_cuda_memory",
        lambda: CudaMemorySnapshot(
            allocated_bytes=2000 * 1024**2,
            reserved_bytes=3000 * 1024**2,
            peak_allocated_bytes=2500 * 1024**2,
            peak_reserved_bytes=3000 * 1024**2,
            total_device_bytes=8192 * 1024**2,
        ),
    )

    writer = MockTelemetryWriter()
    controller = AdmissionController(
        cost_model=ByteCostModel(),
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=256.0,
    )

    result = controller.assert_current_memory_within_budget(
        telemetry_writer=writer, iteration=100
    )

    assert not empty_cache_called, "empty_cache must NOT be called when footprint is under budget"
    assert result["allocated_mb"] == 2000.0
    assert result["reserved_mb"] == 3000.0
    assert len(writer.events) == 0, "No reclaim events should be logged when under budget"


def test_b_recoverable_cache_reclaims_and_proceeds(monkeypatch):
    """
    Test B — Recoverable cache:
    Footprint initially exceeds hard budget due to uncollected reserved cache pool.
    Calling empty_cache reduces reserved memory so footprint falls below budget.
    Must allow continuation, log cache_reclaimed, and call empty_cache EXACTLY ONCE.
    """
    empty_cache_call_count = 0
    call_stage = 0

    def mock_empty_cache():
        nonlocal empty_cache_call_count, call_stage
        empty_cache_call_count += 1
        call_stage = 1

    def mock_capture():
        # Before cleanup (stage 0): allocated = 2800 MB, reserved = 5900 MB.
        # Footprint = 5900 + 256 = 6156 MB > 6144 MB!
        if call_stage == 0:
            return CudaMemorySnapshot(
                allocated_bytes=2800 * 1024**2,
                reserved_bytes=5900 * 1024**2,
                total_device_bytes=8192 * 1024**2,
            )
        # After cleanup (stage 1): reserved shrinks to allocated (2800 MB).
        # Footprint = 2800 + 256 = 3056 MB <= 6144 MB!
        else:
            return CudaMemorySnapshot(
                allocated_bytes=2800 * 1024**2,
                reserved_bytes=2800 * 1024**2,
                total_device_bytes=8192 * 1024**2,
            )

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from zerogs import admission_controller as ac_module

    monkeypatch.setattr(ac_module, "capture_cuda_memory", mock_capture)

    writer = MockTelemetryWriter()
    controller = AdmissionController(
        cost_model=ByteCostModel(),
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=256.0,
    )

    result = controller.assert_current_memory_within_budget(
        telemetry_writer=writer, iteration=146
    )

    assert empty_cache_call_count == 1, "empty_cache must be called exactly once"
    assert result["cache_reclaimed"] is True
    assert result["reclaimed_mb"] == (5900.0 - 2800.0)
    assert controller.total_cache_reclaims == 1

    # Check telemetry events
    triggered = writer.find_events("cache_reclaim_triggered")
    reclaimed = writer.find_events("cache_reclaimed")
    assert len(triggered) == 1
    assert len(reclaimed) == 1
    assert reclaimed[0]["iteration"] == 146
    assert reclaimed[0]["reclaimed_mb"] == 3100.0
    assert reclaimed[0]["reserved_before_mb"] == 5900.0
    assert reclaimed[0]["reserved_after_mb"] == 2800.0


def test_c_unrecoverable_memory_raises_error_with_telemetry(monkeypatch):
    """
    Test C — Unrecoverable memory:
    Footprint initially exceeds budget.
    Even after empty_cache, footprint still exceeds budget.
    Must raise VramBudgetExceededError with full before/after snapshots.
    """
    empty_cache_call_count = 0

    def mock_empty_cache():
        nonlocal empty_cache_call_count
        empty_cache_call_count += 1

    # Both before and after cleanup, reserved is 6200 MB (> 6144 MB)
    def mock_capture():
        return CudaMemorySnapshot(
            allocated_bytes=6000 * 1024**2,
            reserved_bytes=6200 * 1024**2,
            total_device_bytes=8192 * 1024**2,
        )

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from zerogs import admission_controller as ac_module

    monkeypatch.setattr(ac_module, "capture_cuda_memory", mock_capture)

    writer = MockTelemetryWriter()
    controller = AdmissionController(
        cost_model=ByteCostModel(),
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=256.0,
    )

    with pytest.raises(VramBudgetExceededError) as exc_info:
        controller.assert_current_memory_within_budget(
            telemetry_writer=writer, iteration=200
        )

    assert empty_cache_call_count == 1, "Must have attempted empty_cache once"
    err = exc_info.value
    assert err.before_cleanup is not None
    assert err.after_cleanup is not None
    assert "unrecoverable after empty_cache" in str(err)

    # Check telemetry writer recorded both trigger and post-cleanup failure
    assert len(writer.find_events("cache_reclaim_triggered")) == 1
    assert len(writer.find_events("budget_exceeded_post_cleanup")) == 1


def test_d_active_allocations_exceed_budget(monkeypatch):
    """
    Test D — Active allocations:
    Allocated memory itself exceeds budget (e.g. 6200 MB allocated).
    empty_cache cannot free active tensors.
    Must raise VramBudgetExceededError and NOT falsely declare safe.
    """
    def mock_empty_cache():
        pass

    # Active allocated is 6200 MB
    def mock_capture():
        return CudaMemorySnapshot(
            allocated_bytes=6200 * 1024**2,
            reserved_bytes=6200 * 1024**2,
            total_device_bytes=8192 * 1024**2,
        )

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from zerogs import admission_controller as ac_module

    monkeypatch.setattr(ac_module, "capture_cuda_memory", mock_capture)

    controller = AdmissionController(
        cost_model=ByteCostModel(),
        hard_vram_limit_mb=6144.0,
        safety_headroom_mb=512.0,
        non_pytorch_allowance_mb=0.0,
    )

    with pytest.raises(VramBudgetExceededError):
        controller.assert_current_memory_within_budget()


def test_e_cuda_unavailable_cpu_mock_environment(monkeypatch):
    """
    Test E — CUDA unavailable:
    When torch.cuda.is_available() is False (CPU test or mock environment),
    empty_cache must not be called and no crash should occur.
    """
    empty_cache_called = False

    def mock_empty_cache():
        nonlocal empty_cache_called
        empty_cache_called = True

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from zerogs import admission_controller as ac_module

    # Snapshot under budget
    monkeypatch.setattr(
        ac_module,
        "capture_cuda_memory",
        lambda: CudaMemorySnapshot(
            allocated_bytes=100 * 1024**2,
            reserved_bytes=100 * 1024**2,
            total_device_bytes=8192 * 1024**2,
        ),
    )

    controller = AdmissionController(
        cost_model=ByteCostModel(),
        hard_vram_limit_mb=1000.0,
        safety_headroom_mb=100.0,
        non_pytorch_allowance_mb=50.0,
    )

    result = controller.assert_current_memory_within_budget()
    assert not empty_cache_called
    assert result["allocated_mb"] == 100.0

    # Snapshot over budget on CPU mock: raises without crash
    monkeypatch.setattr(
        ac_module,
        "capture_cuda_memory",
        lambda: CudaMemorySnapshot(
            allocated_bytes=1200 * 1024**2,
            reserved_bytes=1200 * 1024**2,
            total_device_bytes=8192 * 1024**2,
        ),
    )

    with pytest.raises(VramBudgetExceededError):
        controller.assert_current_memory_within_budget()
    assert not empty_cache_called


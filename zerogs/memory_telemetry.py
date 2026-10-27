"""CUDA memory observations and append-only experiment telemetry."""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class CudaMemorySnapshot:
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    total_device_bytes: int = 0
    device_index: Optional[int] = None

    @property
    def conservative_footprint_bytes(self) -> int:
        """Memory unavailable to a new PyTorch allocation."""
        return max(self.allocated_bytes, self.reserved_bytes)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["conservative_footprint_bytes"] = self.conservative_footprint_bytes
        for key, value in list(result.items()):
            if key.endswith("_bytes"):
                result[key.replace("_bytes", "_mb")] = round(value / (1024**2), 3)
        return result


def capture_cuda_memory(device: Optional[int] = None) -> CudaMemorySnapshot:
    """Synchronize and capture allocator counters, or return zeros on CPU."""
    if not torch.cuda.is_available():
        return CudaMemorySnapshot()
    index = torch.cuda.current_device() if device is None else int(device)
    torch.cuda.synchronize(index)
    properties = torch.cuda.get_device_properties(index)
    return CudaMemorySnapshot(
        allocated_bytes=int(torch.cuda.memory_allocated(index)),
        reserved_bytes=int(torch.cuda.memory_reserved(index)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(index)),
        total_device_bytes=int(properties.total_memory),
        device_index=index,
    )


class JsonlTelemetryWriter:
    """Append structured records that remain usable after interrupted runs."""

    def __init__(self, path: str, truncate: bool = False):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if truncate:
            with open(self.path, "w", encoding="utf-8"):
                pass

    def write(self, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, default=str) + "\n")

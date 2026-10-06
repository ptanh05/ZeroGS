"""
ZeroGS: Byte-Level Transient Memory Modeling and Occlusion-Aware Admission Control
for Budget-Constrained 3D Gaussian Splatting.

Main Modules:
- ByteCostModel: Analytical and empirical transient/persistent byte cost estimation.
- AdmissionController: Deterministic headroom validation and Prune-First dispatch.
- OcclusionAwareEngine / OcclusionAwareDemandEstimator: Physics-guided demand compensation.
- MarginalUtilityAllocator: Byte-level Knapsack quota allocation (lambda = Gain / Byte).
- MemoryBoundedDensificationScheduler: Unified drop-in scheduler for 3DGS mutation loops.
"""

from .cost_model import ByteCostModel
from .admission_controller import AdmissionController, AdmissionDecision, AdmissionAction
from .occlusion_engine import OcclusionAwareEngine, OcclusionAwareDemandEstimator
from .marginal_allocator import MarginalUtilityAllocator
from .scheduler import MemoryBoundedDensificationScheduler

__version__ = "1.0.0"
__author__ = "Phung The Anh"

__all__ = [
    "ByteCostModel",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionAction",
    "OcclusionAwareEngine",
    "OcclusionAwareDemandEstimator",
    "MarginalUtilityAllocator",
    "MemoryBoundedDensificationScheduler",
]

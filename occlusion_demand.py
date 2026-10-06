"""
Shim for backward compatibility with root-level imports:
from occlusion_demand import OcclusionAwareDemandEstimator
"""

from zerogs.occlusion_engine import OcclusionAwareDemandEstimator

__all__ = ["OcclusionAwareDemandEstimator"]

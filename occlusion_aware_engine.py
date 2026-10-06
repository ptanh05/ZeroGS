"""
Shim for backward compatibility with root-level imports:
from occlusion_aware_engine import OcclusionAwareEngine
"""

from zerogs.occlusion_engine import OcclusionAwareEngine

__all__ = ["OcclusionAwareEngine"]

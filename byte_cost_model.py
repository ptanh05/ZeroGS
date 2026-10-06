"""
Shim for backward compatibility with root-level imports:
from byte_cost_model import ByteCostModel
"""

from zerogs.cost_model import ByteCostModel

__all__ = ["ByteCostModel"]

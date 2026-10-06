"""
Shim for backward compatibility with root-level imports:
from admission_controller import AdmissionController, AdmissionDecision, AdmissionAction
"""

from zerogs.admission_controller import (
    AdmissionController,
    AdmissionDecision,
    AdmissionAction,
)

__all__ = ["AdmissionController", "AdmissionDecision", "AdmissionAction"]

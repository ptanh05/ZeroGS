"""
simple_knn package shim for 3D Gaussian Splatting.
Provides distCUDA2 (3-NN mean squared distance) with pure PyTorch CUDA support.
"""

from . import _C

__all__ = ["_C"]

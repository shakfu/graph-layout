"""
Spectral graph layout algorithms.

This module provides spectral layout algorithms based on eigenvector
decomposition of the graph Laplacian matrix.
"""

from .spectral import SpectralLayout

__all__ = [
    "SpectralLayout",
]

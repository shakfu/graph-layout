"""
Shortest paths calculation with optimized implementations.

This module provides shortest path calculations with automatic implementation selection:
1. Cython-compiled Dijkstra (fastest, pre-built in PyPI wheels)
2. Pure Python Dijkstra (fallback, always available)

The implementation is selected automatically at import time based on availability.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, TypeVar, cast


class PerformanceWarning(UserWarning):
    """Warning about performance-related issues."""

    pass


T = TypeVar("T")

# Determine which implementation to use
_IMPLEMENTATION: str
_Calculator: Any

# Try Cython implementation first
try:
    from .. import _speedups  # type: ignore[attr-defined]

    _Calculator = _speedups.Calculator
    _IMPLEMENTATION = "cython"
except ImportError:
    # Fall back to pure Python
    from ._shortestpaths_py import Calculator as _PyCalculator

    _Calculator = _PyCalculator
    _IMPLEMENTATION = "python"


class Calculator:
    """
    Calculator for all-pairs shortest paths or shortest paths from a single node.

    This is a wrapper that delegates to the best available implementation:
    - Cython (fastest, ~10-30x speedup)
    - Pure Python (baseline)

    Uses Dijkstra's algorithm with a priority queue for efficiency.
    """

    def __init__(
        self,
        n: int,
        edges: list[T],
        get_source_index: Callable[[T], int],
        get_target_index: Callable[[T], int],
        get_length: Callable[[T], float],
    ):
        """
        Initialize shortest path calculator.

        Args:
            n: Number of nodes
            edges: List of edges
            get_source_index: Function to get source node index from edge
            get_target_index: Function to get target node index from edge
            get_length: Function to get edge length
        """
        self.n = n
        self.edges = edges
        self.get_source_index = get_source_index
        self.get_target_index = get_target_index
        self.get_length = get_length

        self._calc: Any = _Calculator(n, edges, get_source_index, get_target_index, get_length)

        # Keep a pure Python calculator for advanced features
        # (e.g., path_from_node_to_node_with_prev_cost)
        if _IMPLEMENTATION == "cython":
            from ._shortestpaths_py import Calculator as _PyCalculator

            self._py_calc = _PyCalculator(n, edges, get_source_index, get_target_index, get_length)
        else:
            self._py_calc = self._calc

    def distance_matrix(self) -> list[list[float]]:
        """
        Compute all-pairs shortest paths.

        Returns:
            Matrix of shortest distances between all pairs of nodes
        """
        return cast(list[list[float]], self._calc.distance_matrix())

    def distances_from_node(self, start: int) -> list[float]:
        """
        Get shortest paths from a specified start node.

        Args:
            start: Starting node index

        Returns:
            Array of shortest distances from start to all other nodes
        """
        return cast(list[float], self._calc.distances_from_node(start))

    def path_from_node_to_node(self, start: int, end: int) -> list[int]:
        """
        Find shortest path from start to end node.

        Args:
            start: Start node index
            end: End node index

        Returns:
            List of node indices in the path (excluding start, including end)
        """
        return cast(list[int], self._calc.path_from_node_to_node(start, end))

    def path_from_node_to_node_with_prev_cost(
        self, start: int, end: int, prev_cost: Callable[[int, int, int], float]
    ) -> list[int]:
        """
        Find shortest path with custom cost function based on previous edge.

        This method always uses the pure Python implementation as it requires
        advanced features not available in the Cython implementation.

        Args:
            start: Start node index
            end: End node index
            prev_cost: Function(prev_node, current_node, next_node) -> cost

        Returns:
            List of node indices in the path
        """
        return self._py_calc.path_from_node_to_node_with_prev_cost(start, end, prev_cost)


def get_implementation() -> str:
    """
    Get the name of the current shortest paths implementation.

    Returns:
        One of: "cython", "python"
    """
    return _IMPLEMENTATION


# Warn user about implementation choice
if _IMPLEMENTATION == "python":
    warnings.warn(
        "Using pure Python shortest paths implementation. "
        "For better performance, install from PyPI (pip install graph-layout) "
        "which includes pre-built Cython extensions.",
        PerformanceWarning,
        stacklevel=2,
    )

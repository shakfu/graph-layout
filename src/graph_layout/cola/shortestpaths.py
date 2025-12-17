"""
Shortest paths calculation with optimized implementations.

This module provides a priority cascade for shortest path calculations:
1. Cython-compiled Dijkstra (fastest, no runtime dependencies)
2. SciPy sparse graph algorithms (fast, requires scipy)
3. Pure Python Dijkstra (slowest, always available)

The implementation is selected automatically at import time based on availability.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, TypeVar, cast


class PerformanceWarning(UserWarning):
    """Warning about performance-related issues."""
    pass

T = TypeVar("T")

# Determine which implementation to use
_IMPLEMENTATION = "unknown"
_Calculator: Optional[Any] = None

# Try Cython implementation first
try:
    from . import _shortestpaths_cy  # type: ignore[attr-defined]
    _Calculator = _shortestpaths_cy.Calculator
    _IMPLEMENTATION = "cython"
except ImportError as e:
    _cython_error = str(e)

    # Try scipy as fallback
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
        _IMPLEMENTATION = "scipy"
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
    - SciPy (fast, ~3-5x speedup)
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

        if _IMPLEMENTATION in ("cython", "python"):
            # Use Cython or pure Python Calculator directly
            assert _Calculator is not None
            self._calc: Any = _Calculator(n, edges, get_source_index, get_target_index, get_length)
            self._scipy_mode = False
        else:
            # Build adjacency matrix for scipy
            self._scipy_mode = True
            self._build_scipy_graph()

        # Always keep a pure Python calculator for advanced features
        # (e.g., path_from_node_to_node_with_prev_cost)
        if _IMPLEMENTATION != "python":
            from ._shortestpaths_py import Calculator as _PyCalculator
            self._py_calc = _PyCalculator(n, edges, get_source_index, get_target_index, get_length)
        else:
            self._py_calc = self._calc

    def _build_scipy_graph(self) -> None:
        """Build scipy sparse graph representation."""
        from scipy.sparse import csr_matrix

        # Build edge lists
        row_ind = []
        col_ind = []
        data = []

        for edge in self.edges:
            u = self.get_source_index(edge)
            v = self.get_target_index(edge)
            d = self.get_length(edge)

            # Undirected graph - add both directions
            row_ind.append(u)
            col_ind.append(v)
            data.append(d)

            row_ind.append(v)
            col_ind.append(u)
            data.append(d)

        # Create sparse adjacency matrix
        self._graph = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(self.n, self.n)
        )

    def distance_matrix(self) -> list[list[float]]:
        """
        Compute all-pairs shortest paths.

        Returns:
            Matrix of shortest distances between all pairs of nodes
        """
        if self._scipy_mode:
            from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

            # Compute all-pairs shortest paths
            dist_matrix = scipy_shortest_path(
                self._graph,
                method='D',  # Dijkstra
                directed=False,
                return_predecessors=False
            )

            # Convert to list of lists
            return cast(list[list[float]], dist_matrix.tolist())
        else:
            return cast(list[list[float]], self._calc.distance_matrix())

    def distances_from_node(self, start: int) -> list[float]:
        """
        Get shortest paths from a specified start node.

        Args:
            start: Starting node index

        Returns:
            Array of shortest distances from start to all other nodes
        """
        if self._scipy_mode:
            from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

            # Compute shortest paths from single source
            distances = scipy_shortest_path(
                self._graph,
                method='D',
                directed=False,
                indices=start,
                return_predecessors=False
            )

            return cast(list[float], distances.tolist())
        else:
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
        if self._scipy_mode:
            from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

            # Get predecessors for path reconstruction
            _, predecessors = scipy_shortest_path(
                self._graph,
                method='D',
                directed=False,
                indices=start,
                return_predecessors=True
            )

            # Reconstruct path
            path = []
            current = end
            while current != start and predecessors[current] != -9999:
                path.append(predecessors[current])
                current = predecessors[current]

            return path
        else:
            return cast(list[int], self._calc.path_from_node_to_node(start, end))

    def path_from_node_to_node_with_prev_cost(
        self, start: int, end: int, prev_cost: Callable[[int, int, int], float]
    ) -> list[int]:
        """
        Find shortest path with custom cost function based on previous edge.

        This method always uses the pure Python implementation as it requires
        advanced features not available in the optimized implementations.

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
        One of: "cython", "scipy", "python"
    """
    return _IMPLEMENTATION


# Re-export classes from pure Python implementation for compatibility


# Warn user about implementation choice
if _IMPLEMENTATION == "python":
    warnings.warn(
        "Using pure Python shortest paths implementation. "
        "For better performance, install scipy (pip install scipy) "
        "or build with Cython extensions.",
        PerformanceWarning,
        stacklevel=2
    )
elif _IMPLEMENTATION == "scipy":
    # Scipy is good, but let user know Cython would be better if they build from source
    pass  # Silent - scipy is fast enough

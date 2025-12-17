"""
3D graph layout using force-directed placement.

This module extends the force-directed layout to 3D space.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from .descent import Descent
from .linklengths import LinkLengthAccessor, jaccard_link_lengths
from .rectangle import GraphNode, Projection
from .shortestpaths import Calculator


class Link3D:
    """Link in 3D layout."""

    def __init__(self, source: int, target: int):
        self.source = source
        self.target = target
        self.length: float = 1.0

    def actual_length(self, x: np.ndarray) -> float:
        """
        Calculate actual length in current layout.

        Args:
            x: Position matrix (3 x n)

        Returns:
            Euclidean distance
        """
        dist_squared = 0.0
        for dim in range(x.shape[0]):
            dx = x[dim, self.target] - x[dim, self.source]
            dist_squared += dx * dx
        return math.sqrt(dist_squared)


class Node3D(GraphNode):
    """Node in 3D layout."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.fixed = False


class _LinkAccessor3D(LinkLengthAccessor[Link3D]):
    """Link accessor for 3D links."""

    def get_source_index(self, e: Link3D) -> int:
        return e.source

    def get_target_index(self, e: Link3D) -> int:
        return e.target

    def set_length(self, l: Link3D, value: float) -> None:
        l.length = value


class Layout3D:
    """
    3D force-directed graph layout.

    Uses gradient descent to minimize stress in 3D space.
    """

    DIMS = ["x", "y", "z"]
    K = len(DIMS)

    def __init__(self, nodes: list[Node3D], links: list[Link3D], ideal_link_length: float = 1.0):
        """
        Initialize 3D layout.

        Args:
            nodes: List of nodes
            links: List of links
            ideal_link_length: Desired length for links
        """
        self.nodes = nodes
        self.links = links
        self.ideal_link_length = ideal_link_length
        self.use_jaccard_link_lengths = True
        self.constraints: Optional[list] = None
        self.descent: Optional[Descent] = None

        # Initialize result matrix (3 x n)
        n = len(nodes)
        self.result = np.zeros((self.K, n))

        # Initialize node positions (random if undefined)
        for i, v in enumerate(nodes):
            for dim_idx, dim in enumerate(self.DIMS):
                val = getattr(v, dim, None)
                if val is None:
                    val = random.random()
                    setattr(v, dim, val)
                self.result[dim_idx, i] = val

    def link_length(self, l: Link3D) -> float:
        """Get actual length of a link in current layout."""
        return l.actual_length(self.result)

    def start(self, iterations: int = 100) -> "Layout3D":
        """
        Run layout algorithm.

        Args:
            iterations: Number of iterations

        Returns:
            Self for chaining
        """
        n = len(self.nodes)

        # Apply Jaccard link lengths if enabled
        link_accessor = _LinkAccessor3D()

        if self.use_jaccard_link_lengths:
            jaccard_link_lengths(self.links, link_accessor, 1.5)

        # Scale by ideal link length
        for e in self.links:
            e.length *= self.ideal_link_length

        # Create distance matrix using shortest paths
        calc = Calculator(n, self.links, lambda e: e.source, lambda e: e.target, lambda e: e.length)
        distance_matrix = calc.distance_matrix()

        # Convert to NumPy array
        D = np.array(distance_matrix)

        # G matrix: 1 for edges, 2 for non-edges
        G = np.full((n, n), 2.0)
        for link in self.links:
            G[link.source, link.target] = 1.0
            G[link.target, link.source] = 1.0

        # Create descent optimizer
        self.descent = Descent(self.result, D, G)
        self.descent.threshold = 1e-3

        # Add constraints if specified
        if self.constraints:
            from typing import cast

            self.descent.project = Projection(
                cast(list[GraphNode], self.nodes), [], constraints=self.constraints
            ).project_functions()

        # Lock fixed nodes
        for i, v in enumerate(self.nodes):
            if v.fixed:
                self.descent.locks.add(i, np.array([v.x, v.y, v.z]))

        # Run descent
        self.descent.run(iterations)

        return self

    def tick(self) -> float:
        """
        Perform one iteration of layout.

        Returns:
            Displacement
        """
        if self.descent is None:
            raise RuntimeError("Must call start() before tick()")

        # Update locks
        self.descent.locks.clear()
        for i, v in enumerate(self.nodes):
            if v.fixed:
                self.descent.locks.add(i, np.array([v.x, v.y, v.z]))

        return self.descent.runge_kutta()

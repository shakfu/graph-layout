"""
Kamada-Kawai force-directed layout algorithm.

Based on the paper:
"An Algorithm for Drawing General Undirected Graphs"
by Kamada and Kawai (1989)

This algorithm minimizes stress (energy) where the ideal distances
between nodes are proportional to their graph-theoretic distances
(shortest path lengths).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Optional, Sequence, cast

import numpy as np

from ..base import IterativeLayout
from ..types import (
    Event,
    EventType,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)


class KamadaKawaiLayout(IterativeLayout):
    """
    Kamada-Kawai stress-minimization layout.

    Positions nodes to minimize a stress function where the ideal
    distance between any two nodes is proportional to their
    graph-theoretic distance (shortest path length).

    The algorithm moves one node at a time (the one with maximum
    partial derivative of energy) until convergence.

    Example:
        layout = KamadaKawaiLayout(
            nodes=[{'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}],
            links=[{'source': 0, 'target': 1}, {'source': 1, 'target': 2}],
            size=(800, 600),
        )
        layout.run()
    """

    def __init__(
        self,
        *,
        nodes: Optional[Sequence[NodeLike]] = None,
        links: Optional[Sequence[LinkLike]] = None,
        groups: Optional[Sequence[GroupLike]] = None,
        size: SizeType = (1.0, 1.0),
        random_seed: Optional[int] = None,
        on_start: Optional[Callable[[Optional[Event]], None]] = None,
        on_tick: Optional[Callable[[Optional[Event]], None]] = None,
        on_end: Optional[Callable[[Optional[Event]], None]] = None,
        # IterativeLayout parameters
        alpha: float = 1.0,
        alpha_min: float = 0.001,
        alpha_decay: float = 0.99,
        iterations: int = 300,
        # KamadaKawai-specific parameters
        edge_length: float = 100.0,
        epsilon: float = 0.0001,
        disconnected_distance: Optional[float] = None,
        max_inner_iterations: int = 30,
    ) -> None:
        """
        Initialize Kamada-Kawai layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            alpha: Initial alpha/temperature (0 to 1)
            alpha_min: Minimum alpha for convergence threshold
            alpha_decay: Alpha decay rate per iteration (0 to 1)
            iterations: Maximum outer iterations
            edge_length: Ideal length for a single edge. Multi-hop paths will have
                ideal lengths that are multiples of this.
            epsilon: Convergence threshold for gradient magnitude.
            disconnected_distance: Distance for disconnected pairs. If None, uses
                diameter * 1.5.
            max_inner_iterations: Maximum Newton-Raphson iterations per node move.
        """
        super().__init__(
            nodes=nodes,
            links=links,
            groups=groups,
            size=size,
            random_seed=random_seed,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
            alpha=alpha,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
            iterations=iterations,
        )

        # KamadaKawai-specific configuration
        self._edge_length: float = float(edge_length)
        self._epsilon: float = float(epsilon)
        self._disconnected_distance: Optional[float] = disconnected_distance
        self._max_inner_iterations: int = int(max_inner_iterations)

        # Internal state
        self._dist_matrix: Optional[np.ndarray] = None
        self._k_matrix: Optional[np.ndarray] = None
        self._l_matrix: Optional[np.ndarray] = None
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def edge_length(self) -> float:
        """Get ideal length for a single edge."""
        return self._edge_length

    @edge_length.setter
    def edge_length(self, value: float) -> None:
        """Set ideal length for a single edge."""
        self._edge_length = float(value)

    @property
    def epsilon(self) -> float:
        """Get convergence threshold."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set convergence threshold."""
        self._epsilon = float(value)

    @property
    def disconnected_distance(self) -> Optional[float]:
        """Get distance for disconnected node pairs."""
        return self._disconnected_distance

    @disconnected_distance.setter
    def disconnected_distance(self, value: Optional[float]) -> None:
        """Set distance for disconnected node pairs."""
        self._disconnected_distance = float(value) if value is not None else None

    @property
    def max_inner_iterations(self) -> int:
        """Get maximum Newton-Raphson iterations per node move."""
        return self._max_inner_iterations

    @max_inner_iterations.setter
    def max_inner_iterations(self, value: int) -> None:
        """Set maximum Newton-Raphson iterations per node move."""
        self._max_inner_iterations = int(value)

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def _compute_shortest_paths(self) -> np.ndarray:
        """
        Compute all-pairs shortest paths using BFS.

        Returns:
            Distance matrix where dist[i,j] is the shortest path length.
        """
        n = len(self._nodes)
        dist = np.full((n, n), float("inf"))

        # Build adjacency list
        adj = self._build_adjacency()

        # BFS from each node
        for start in range(n):
            dist[start, start] = 0
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if dist[start, neighbor] == float("inf"):
                        dist[start, neighbor] = dist[start, curr] + 1
                        queue.append(neighbor)

        return cast(np.ndarray, dist)

    def _initialize_matrices(self) -> None:
        """Initialize the distance, spring constant, and ideal length matrices."""
        n = len(self._nodes)

        # Compute shortest path distances
        self._dist_matrix = self._compute_shortest_paths()

        # Handle disconnected components
        max_dist: float = 0.0
        for i in range(n):
            for j in range(n):
                if self._dist_matrix[i, j] != float("inf"):
                    max_dist = max(max_dist, float(self._dist_matrix[i, j]))

        # Replace infinity with a large distance
        if self._disconnected_distance:
            disconn_dist = self._disconnected_distance
        else:
            disconn_dist = max(max_dist * 1.5, n) if max_dist > 0 else n

        self._dist_matrix = np.where(
            self._dist_matrix == float("inf"), disconn_dist, self._dist_matrix
        )

        # Ideal lengths: l_ij = L * d_ij where L is edge_length
        self._l_matrix = self._dist_matrix * self._edge_length

        # Spring constants: k_ij = K / d_ij^2
        # Using K = 1 for simplicity
        with np.errstate(divide="ignore", invalid="ignore"):
            self._k_matrix = 1.0 / (self._dist_matrix**2)
            self._k_matrix[self._dist_matrix == 0] = 0

    def _compute_energy_gradient(self, m: int) -> tuple[float, float]:
        """
        Compute the partial derivatives of energy with respect to node m.

        Args:
            m: Node index

        Returns:
            Tuple (dE/dx_m, dE/dy_m)
        """
        assert self._k_matrix is not None
        assert self._l_matrix is not None

        n = len(self._nodes)
        dx_sum = 0.0
        dy_sum = 0.0

        x_m = self._nodes[m].x
        y_m = self._nodes[m].y

        for i in range(n):
            if i == m:
                continue

            x_i = self._nodes[i].x
            y_i = self._nodes[i].y

            dx = x_m - x_i
            dy = y_m - y_i
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0:
                k_mi = self._k_matrix[m, i]
                l_mi = self._l_matrix[m, i]

                factor = k_mi * (1 - l_mi / dist)
                dx_sum += factor * dx
                dy_sum += factor * dy

        return dx_sum, dy_sum

    def _compute_delta(self, m: int) -> float:
        """
        Compute Delta_m = sqrt((dE/dx_m)^2 + (dE/dy_m)^2).

        This measures how much node m "wants" to move.
        """
        grad_x, grad_y = self._compute_energy_gradient(m)
        return math.sqrt(grad_x * grad_x + grad_y * grad_y)

    def _move_node(self, m: int) -> None:
        """
        Move node m to minimize its contribution to total energy.

        Uses Newton-Raphson iteration on the gradient equations.
        """
        assert self._k_matrix is not None
        assert self._l_matrix is not None

        n = len(self._nodes)

        for _ in range(self._max_inner_iterations):
            # Compute gradient
            grad_x, grad_y = self._compute_energy_gradient(m)

            delta = math.sqrt(grad_x * grad_x + grad_y * grad_y)
            if delta < self._epsilon:
                break

            # Compute Hessian elements
            x_m = self._nodes[m].x
            y_m = self._nodes[m].y

            d2E_dx2 = 0.0
            d2E_dy2 = 0.0
            d2E_dxdy = 0.0

            for i in range(n):
                if i == m:
                    continue

                x_i = self._nodes[i].x
                y_i = self._nodes[i].y

                dx = x_m - x_i
                dy = y_m - y_i
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq)
                dist_cubed = dist * dist_sq

                if dist > 0:
                    k_mi = self._k_matrix[m, i]
                    l_mi = self._l_matrix[m, i]

                    d2E_dx2 += k_mi * (1 - l_mi * dy * dy / dist_cubed)
                    d2E_dy2 += k_mi * (1 - l_mi * dx * dx / dist_cubed)
                    d2E_dxdy += k_mi * l_mi * dx * dy / dist_cubed

            # Solve 2x2 linear system using Cramer's rule
            det = d2E_dx2 * d2E_dy2 - d2E_dxdy * d2E_dxdy

            if abs(det) < 1e-10:
                break

            delta_x = (-grad_x * d2E_dy2 + grad_y * d2E_dxdy) / det
            delta_y = (grad_x * d2E_dxdy - grad_y * d2E_dx2) / det

            # Update position
            self._nodes[m].x += delta_x
            self._nodes[m].y += delta_y

    def run(self, **kwargs: Any) -> "KamadaKawaiLayout":
        """
        Run the layout algorithm.

        Keyword Args:
            random_init: Initialize positions randomly (default: True)
            center_graph: Center graph after completion (default: True)

        Returns:
            self for chaining
        """
        self._initialize_indices()

        random_init = kwargs.get("random_init", True)
        center = kwargs.get("center_graph", True)

        if random_init:
            self._initialize_positions(random_init=True)

        n = len(self._nodes)
        if n == 0:
            return self

        # Initialize matrices
        self._initialize_matrices()
        self._iteration = 0
        self._alpha = 1.0

        # Fire start event
        self.trigger({"type": EventType.start, "alpha": self._alpha})

        # Run layout
        self.kick()

        if center:
            self._center_graph()

        # Fire end event
        self.trigger({"type": EventType.end, "alpha": 0.0})

        return self

    def tick(self) -> bool:
        """
        Perform one iteration of the layout.

        Each iteration finds the node with maximum gradient and moves it.

        Returns:
            True if converged, False otherwise.
        """
        # These are set in run() before tick() is called
        assert self._k_matrix is not None
        assert self._l_matrix is not None

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        if n == 0:
            return True

        # Find the node with maximum Delta
        max_delta = 0.0
        max_node = -1

        for i in range(n):
            if self._nodes[i].fixed:
                continue

            delta = self._compute_delta(i)
            if delta > max_delta:
                max_delta = delta
                max_node = i

        # Check convergence
        if max_delta < self._epsilon or max_node < 0:
            return True

        # Move the node with maximum delta
        self._move_node(max_node)

        self._iteration += 1
        self._alpha = max_delta  # Use delta as a measure of "energy"

        # Fire tick event
        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": max_delta})

        return False


__all__ = ["KamadaKawaiLayout"]

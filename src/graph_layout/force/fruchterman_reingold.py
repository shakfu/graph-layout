"""
Fruchterman-Reingold force-directed layout algorithm.

Based on the paper:
"Graph Drawing by Force-directed Placement" by Fruchterman and Reingold (1991)

The algorithm simulates a physical system where:
- All nodes repel each other (like electrical charges)
- Connected nodes attract each other (like springs)
- A "temperature" parameter limits movement and decreases over time
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..base import IterativeLayout
from ..spatial.quadtree import Body, QuadTree
from ..types import (
    Event,
    EventType,
    GroupLike,
    LinkLike,
    NodeLike,
    SizeType,
)

# Try to import Cython-optimized functions
try:
    from .. import _speedups  # type: ignore[attr-defined]

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


class FruchtermanReingoldLayout(IterativeLayout):
    """
    Fruchterman-Reingold force-directed graph layout.

    This algorithm positions nodes by simulating a physical system where:
    - All node pairs have repulsive forces (inversely proportional to distance)
    - Connected node pairs have attractive forces (proportional to distance squared)
    - Movement is limited by a "temperature" that cools over iterations

    Example:
        layout = FruchtermanReingoldLayout(
            nodes=[{'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}],
            links=[{'source': 0, 'target': 1}, {'source': 1, 'target': 2}],
            size=(800, 600),
        )
        layout.run()

        for node in layout.nodes:
            print(f"Node {node.index}: ({node.x}, {node.y})")
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
        # FruchtermanReingold-specific parameters
        optimal_distance: Optional[float] = None,
        temperature: Optional[float] = None,
        cooling_factor: float = 0.95,
        gravity: float = 0.1,
        center_gravity: bool = True,
        use_barnes_hut: bool = False,
        barnes_hut_theta: float = 0.5,
    ) -> None:
        """
        Initialize Fruchterman-Reingold layout.

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
            iterations: Maximum number of iterations
            optimal_distance: Optimal distance between nodes. If None, computed from canvas size.
            temperature: Initial temperature. If None, defaults to canvas width / 10.
            cooling_factor: Temperature decay per iteration (0 to 1). Default 0.95.
            gravity: Gravity strength toward center. Default 0.1.
            center_gravity: Whether to apply center gravity. Default True.
            use_barnes_hut: Enable Barnes-Hut O(n log n) approximation.
            barnes_hut_theta: Barnes-Hut accuracy (0 = exact, 0.5 = balanced).
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

        # FruchtermanReingold-specific configuration
        self._optimal_distance: Optional[float] = optimal_distance
        self._temperature: Optional[float] = temperature
        # Cooling factor: multiplicative decay per iteration. 0.95 retains 95% of
        # temperature each step, giving ~50 iterations to reach 10% of initial temp.
        # Standard value from simulated annealing literature.
        self._cooling_factor: float = max(0.0, min(1.0, float(cooling_factor)))
        self._gravity: float = float(gravity)
        self._center_gravity: bool = bool(center_gravity)
        # Minimum temperature threshold for convergence. When temperature falls
        # below this, node displacement becomes negligible (<0.01% of canvas).
        # Prevents infinite loops while allowing fine-grained final positioning.
        self._min_temperature: float = 0.0001

        # Barnes-Hut optimization
        self._use_barnes_hut: bool = bool(use_barnes_hut)
        self._barnes_hut_theta: float = max(0.0, float(barnes_hut_theta))

        # Internal state
        self._disp_x: Optional[np.ndarray] = None
        self._disp_y: Optional[np.ndarray] = None
        self._pos_x: Optional[np.ndarray] = None
        self._pos_y: Optional[np.ndarray] = None
        self._fixed: Optional[np.ndarray] = None
        self._sources: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def optimal_distance(self) -> float:
        """Get optimal distance between connected nodes."""
        if self._optimal_distance is not None:
            return self._optimal_distance
        return self._compute_optimal_distance()

    @optimal_distance.setter
    def optimal_distance(self, value: Optional[float]) -> None:
        """Set optimal distance between connected nodes."""
        self._optimal_distance = float(value) if value is not None else None

    @property
    def temperature(self) -> float:
        """Get current temperature."""
        if self._temperature is not None:
            return self._temperature
        return self._canvas_size[0] / 10

    @temperature.setter
    def temperature(self, value: Optional[float]) -> None:
        """Set temperature."""
        self._temperature = float(value) if value is not None else None

    @property
    def cooling_factor(self) -> float:
        """Get cooling factor (temperature decay per iteration)."""
        return self._cooling_factor

    @cooling_factor.setter
    def cooling_factor(self, value: float) -> None:
        """Set cooling factor (clamped to [0, 1])."""
        self._cooling_factor = max(0.0, min(1.0, float(value)))

    @property
    def gravity(self) -> float:
        """Get gravity strength toward center."""
        return self._gravity

    @gravity.setter
    def gravity(self, value: float) -> None:
        """Set gravity strength."""
        self._gravity = float(value)

    @property
    def center_gravity(self) -> bool:
        """Get whether center gravity is enabled."""
        return self._center_gravity

    @center_gravity.setter
    def center_gravity(self, value: bool) -> None:
        """Set whether center gravity is enabled."""
        self._center_gravity = bool(value)

    @property
    def use_barnes_hut(self) -> bool:
        """Get whether Barnes-Hut approximation is enabled."""
        return self._use_barnes_hut

    @use_barnes_hut.setter
    def use_barnes_hut(self, value: bool) -> None:
        """Enable/disable Barnes-Hut approximation."""
        self._use_barnes_hut = bool(value)

    @property
    def barnes_hut_theta(self) -> float:
        """Get Barnes-Hut theta parameter (accuracy)."""
        return self._barnes_hut_theta

    @barnes_hut_theta.setter
    def barnes_hut_theta(self, value: float) -> None:
        """Set Barnes-Hut theta parameter."""
        self._barnes_hut_theta = max(0.0, float(value))

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def _compute_optimal_distance(self) -> float:
        """Compute the optimal distance based on canvas area and node count."""
        area = self._canvas_size[0] * self._canvas_size[1]
        n = max(1, len(self._nodes))
        return math.sqrt(area / n)

    def run(self, **kwargs: Any) -> "FruchtermanReingoldLayout":
        """
        Run the layout algorithm.

        Keyword Args:
            random_init: Initialize positions randomly (default: True)
            center_graph: Center graph after completion (default: True)

        Returns:
            self for chaining
        """
        # Initialize parameters
        self._initialize_indices()

        random_init = kwargs.get("random_init", True)
        center = kwargs.get("center_graph", True)

        if random_init:
            self._initialize_positions(random_init=True)

        # Compute optimal distance if not set
        k = self._optimal_distance if self._optimal_distance else self._compute_optimal_distance()
        self._optimal_distance = k

        # Initialize temperature if not set
        if self._temperature is None:
            self._temperature = self._canvas_size[0] / 10

        # Initialize arrays for Cython-optimized computation
        n = len(self._nodes)
        m = len(self._links)
        self._disp_x = np.zeros(n, dtype=np.float64)
        self._disp_y = np.zeros(n, dtype=np.float64)
        self._pos_x = np.zeros(n, dtype=np.float64)
        self._pos_y = np.zeros(n, dtype=np.float64)
        self._fixed = np.zeros(n, dtype=np.uint8)
        self._sources = np.zeros(m, dtype=np.int32)
        self._targets = np.zeros(m, dtype=np.int32)

        # Initialize position and fixed arrays
        for i, node in enumerate(self._nodes):
            self._pos_x[i] = node.x
            self._pos_y[i] = node.y
            self._fixed[i] = 1 if node.fixed else 0

        # Initialize edge arrays
        for e, link in enumerate(self._links):
            self._sources[e] = self._get_source_index(link)
            self._targets[e] = self._get_target_index(link)

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

        Returns:
            True if converged, False otherwise.
        """
        # These are set in run() before tick() is called
        assert self._temperature is not None
        assert self._optimal_distance is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._pos_x is not None
        assert self._pos_y is not None

        if self._temperature < self._min_temperature:
            return True

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        if n == 0:
            return True

        k = self._optimal_distance
        k_sq = k * k

        # Sync positions from nodes to arrays
        for i, node in enumerate(self._nodes):
            self._pos_x[i] = node.x
            self._pos_y[i] = node.y

        # Reset displacements
        self._disp_x.fill(0)
        self._disp_y.fill(0)

        # Calculate repulsive forces
        if _HAS_CYTHON:
            if self._use_barnes_hut and n > 50:
                _speedups.compute_repulsive_forces_barnes_hut(
                    self._pos_x,
                    self._pos_y,
                    self._disp_x,
                    self._disp_y,
                    k_sq,
                    n,
                    self._barnes_hut_theta,
                )
            else:
                _speedups.compute_repulsive_forces(
                    self._pos_x, self._pos_y, self._disp_x, self._disp_y, k_sq, n
                )
        else:
            if self._use_barnes_hut and n > 50:
                self._compute_repulsive_barnes_hut(k_sq)
            else:
                self._compute_repulsive_naive(n, k_sq)

        # Calculate attractive forces along edges
        if _HAS_CYTHON and len(self._links) > 0:
            _speedups.compute_attractive_forces(
                self._pos_x,
                self._pos_y,
                self._disp_x,
                self._disp_y,
                self._sources,
                self._targets,
                k,
                len(self._links),
            )
        else:
            for link in self._links:
                src = self._get_source_index(link)
                tgt = self._get_target_index(link)

                dx = self._nodes[src].x - self._nodes[tgt].x
                dy = self._nodes[src].y - self._nodes[tgt].y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001

                # Attractive force: f_a = d^2 / k
                if dist > 0:
                    force = dist_sq / k
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    self._disp_x[src] -= fx
                    self._disp_y[src] -= fy
                    self._disp_x[tgt] += fx
                    self._disp_y[tgt] += fy

        # Apply center gravity if enabled
        if self._center_gravity and self._gravity > 0:
            cx = self._canvas_size[0] / 2
            cy = self._canvas_size[1] / 2
            for i in range(n):
                dx = cx - self._pos_x[i]
                dy = cy - self._pos_y[i]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    self._disp_x[i] += self._gravity * dx
                    self._disp_y[i] += self._gravity * dy

        # Apply displacements, limited by temperature
        if _HAS_CYTHON:
            _speedups.apply_displacements(
                self._pos_x,
                self._pos_y,
                self._disp_x,
                self._disp_y,
                self._fixed,
                self._temperature,
                0.0,
                0.0,
                self._canvas_size[0],
                self._canvas_size[1],
                n,
            )
            # Sync positions back to nodes
            for i, node in enumerate(self._nodes):
                node.x = float(self._pos_x[i])
                node.y = float(self._pos_y[i])
        else:
            for i in range(n):
                node = self._nodes[i]

                # Skip fixed nodes
                if node.fixed:
                    continue

                disp_len = math.sqrt(
                    self._disp_x[i] * self._disp_x[i] + self._disp_y[i] * self._disp_y[i]
                )

                if disp_len > 0:
                    # Limit displacement by temperature
                    scale = min(disp_len, self._temperature) / disp_len
                    node.x += self._disp_x[i] * scale
                    node.y += self._disp_y[i] * scale

                # Keep within canvas bounds (optional)
                node.x = max(0, min(self._canvas_size[0], node.x))
                node.y = max(0, min(self._canvas_size[1], node.y))

        # Cool down
        self._temperature *= self._cooling_factor
        self._iteration += 1
        self._alpha = self._temperature / (self._canvas_size[0] / 10)

        # Fire tick event
        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": None})

        return False

    def _compute_repulsive_naive(self, n: int, k_sq: float) -> None:
        """Compute repulsive forces using O(n^2) pairwise calculation."""
        assert self._disp_x is not None and self._disp_y is not None
        for i in range(n):
            for j in range(i + 1, n):
                dx = self._nodes[i].x - self._nodes[j].x
                dy = self._nodes[i].y - self._nodes[j].y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001

                # Repulsive force: f_r = k^2 / d
                if dist > 0:
                    force = k_sq / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    self._disp_x[i] += fx
                    self._disp_y[i] += fy
                    self._disp_x[j] -= fx
                    self._disp_y[j] -= fy

    def _compute_repulsive_barnes_hut(self, k_sq: float) -> None:
        """Compute repulsive forces using Barnes-Hut O(n log n) approximation."""
        assert self._disp_x is not None and self._disp_y is not None
        # Build quadtree from current node positions
        tree = QuadTree.from_nodes(self._nodes, padding=10.0, theta=self._barnes_hut_theta)

        # Calculate force on each node using the tree
        for i, node in enumerate(self._nodes):
            body = Body(node.x, node.y, mass=1.0, index=i)
            fx, fy = tree.calculate_force(body, repulsion_constant=k_sq)
            self._disp_x[i] += fx
            self._disp_y[i] += fy


__all__ = ["FruchtermanReingoldLayout"]

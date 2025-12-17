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
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self

from ..base import IterativeLayout
from ..spatial.quadtree import Body, QuadTree
from ..types import EventType


class FruchtermanReingoldLayout(IterativeLayout):
    """
    Fruchterman-Reingold force-directed graph layout.

    This algorithm positions nodes by simulating a physical system where:
    - All node pairs have repulsive forces (inversely proportional to distance)
    - Connected node pairs have attractive forces (proportional to distance squared)
    - Movement is limited by a "temperature" that cools over iterations

    Example:
        layout = (FruchtermanReingoldLayout()
            .nodes([{'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}])
            .links([{'source': 0, 'target': 1}, {'source': 1, 'target': 2}])
            .size([800, 600])
            .start())

        for node in layout.nodes():
            print(f"Node {node.index}: ({node.x}, {node.y})")
    """

    def __init__(self) -> None:
        super().__init__()
        self._k: Optional[float] = None  # Optimal distance between nodes
        self._temperature: Optional[float] = None
        self._cooling_factor: float = 0.95
        self._min_temperature: float = 0.0001
        self._gravity: float = 0.1
        self._center_gravity: bool = True

        # Internal state
        self._disp_x: Optional[np.ndarray] = None
        self._disp_y: Optional[np.ndarray] = None
        self._iteration: int = 0

        # Barnes-Hut optimization
        self._use_barnes_hut: bool = False
        self._barnes_hut_theta: float = 0.5

    # -------------------------------------------------------------------------
    # Configuration Methods (Fluent API)
    # -------------------------------------------------------------------------

    def optimal_distance(self, k: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the optimal distance between connected nodes.

        If not set, computed as sqrt(area / n_nodes).

        Args:
            k: Optimal distance. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if k is None:
            return self._k if self._k else self._compute_optimal_distance()
        self._k = float(k)
        return self

    def temperature(self, t: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the initial temperature.

        Temperature limits how far nodes can move in each iteration.
        Higher temperature = more movement. Defaults to canvas width / 10.

        Args:
            t: Temperature value. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if t is None:
            return self._temperature if self._temperature else self._canvas_size[0] / 10
        self._temperature = float(t)
        return self

    def cooling_factor(self, c: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the cooling factor.

        Temperature is multiplied by this factor each iteration.
        Values close to 1.0 cool slowly, values close to 0 cool quickly.

        Args:
            c: Cooling factor (0 to 1). If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if c is None:
            return self._cooling_factor
        self._cooling_factor = max(0.0, min(1.0, float(c)))
        return self

    def gravity(self, g: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the gravity strength toward the center.

        Gravity pulls nodes toward the canvas center, preventing drift.

        Args:
            g: Gravity strength (0 = none, higher = stronger). If None, returns current.

        Returns:
            Current value or self for chaining.
        """
        if g is None:
            return self._gravity
        self._gravity = float(g)
        return self

    def center_gravity(self, enabled: Optional[bool] = None) -> Union[bool, Self]:
        """
        Get or set whether center gravity is enabled.

        Args:
            enabled: Whether to enable center gravity. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if enabled is None:
            return self._center_gravity
        self._center_gravity = bool(enabled)
        return self

    def barnes_hut(
        self, enabled: Optional[bool] = None, theta: Optional[float] = None
    ) -> Union[bool, Self]:
        """
        Get or set Barnes-Hut approximation for repulsive forces.

        Barnes-Hut reduces force calculation complexity from O(n^2) to O(n log n)
        by approximating distant node clusters as single masses.

        Args:
            enabled: Enable/disable Barnes-Hut. If None, returns current enabled state.
            theta: Accuracy parameter (0 = exact, 0.5 = balanced, 1.0+ = fast).
                   Lower values are more accurate but slower.

        Returns:
            Current enabled state (bool) or self for chaining.
        """
        if enabled is None:
            return self._use_barnes_hut
        self._use_barnes_hut = bool(enabled)
        if theta is not None:
            self._barnes_hut_theta = max(0.0, float(theta))
        return self

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def _compute_optimal_distance(self) -> float:
        """Compute the optimal distance based on canvas area and node count."""
        area = self._canvas_size[0] * self._canvas_size[1]
        n = max(1, len(self._nodes))
        return math.sqrt(area / n)

    def start(self, **kwargs: Any) -> Self:
        """
        Start the layout algorithm.

        Keyword Args:
            iterations: Maximum iterations (default: 300)
            random_init: Initialize positions randomly (default: True)
            center_graph: Center graph after completion (default: True)

        Returns:
            self for chaining
        """
        # Initialize parameters
        self._initialize_indices()

        iterations = kwargs.get('iterations', self._iterations)
        random_init = kwargs.get('random_init', True)
        center = kwargs.get('center_graph', True)

        if random_init:
            self._initialize_positions(random_init=True)

        # Compute optimal distance if not set
        if self._k is None:
            self._k = self._compute_optimal_distance()

        # Initialize temperature if not set
        if self._temperature is None:
            self._temperature = self._canvas_size[0] / 10

        # Initialize displacement arrays
        n = len(self._nodes)
        self._disp_x = np.zeros(n)
        self._disp_y = np.zeros(n)
        self._iteration = 0

        # Set iterations
        self._iterations = iterations
        self._alpha = 1.0

        # Fire start event
        self.trigger({'type': EventType.start, 'alpha': self._alpha})

        # Run layout
        self.kick()

        if center:
            self._center_graph()

        # Fire end event
        self.trigger({'type': EventType.end, 'alpha': 0.0})

        return self

    def tick(self) -> bool:
        """
        Perform one iteration of the layout.

        Returns:
            True if converged, False otherwise.
        """
        # These are set in start() before tick() is called
        assert self._temperature is not None
        assert self._k is not None
        assert self._disp_x is not None
        assert self._disp_y is not None

        if self._temperature < self._min_temperature:
            return True

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        if n == 0:
            return True

        k = self._k
        k_sq = k * k

        # Reset displacements
        self._disp_x.fill(0)
        self._disp_y.fill(0)

        # Calculate repulsive forces
        if self._use_barnes_hut and n > 50:
            self._compute_repulsive_barnes_hut(k_sq)
        else:
            self._compute_repulsive_naive(n, k_sq)

        # Calculate attractive forces along edges
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
                dx = cx - self._nodes[i].x
                dy = cy - self._nodes[i].y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    self._disp_x[i] += self._gravity * dx
                    self._disp_y[i] += self._gravity * dy

        # Apply displacements, limited by temperature
        for i in range(n):
            node = self._nodes[i]

            # Skip fixed nodes
            if node.fixed:
                continue

            disp_len = math.sqrt(
                self._disp_x[i] * self._disp_x[i] +
                self._disp_y[i] * self._disp_y[i]
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
        self.trigger({
            'type': EventType.tick,
            'alpha': self._alpha,
            'stress': None
        })

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
        tree = QuadTree.from_nodes(
            self._nodes,
            padding=10.0,
            theta=self._barnes_hut_theta
        )

        # Calculate force on each node using the tree
        for i, node in enumerate(self._nodes):
            body = Body(node.x, node.y, mass=1.0, index=i)
            fx, fy = tree.calculate_force(body, repulsion_constant=k_sq)
            self._disp_x[i] += fx
            self._disp_y[i] += fy


__all__ = ["FruchtermanReingoldLayout"]

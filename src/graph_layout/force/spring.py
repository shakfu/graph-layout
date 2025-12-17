"""
Simple spring-based force-directed layout.

A basic force-directed algorithm using Hooke's law for spring forces.
Simpler than Fruchterman-Reingold, useful as a baseline or for small graphs.
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


class SpringLayout(IterativeLayout):
    """
    Simple spring-based force-directed layout.

    Uses Hooke's law: F = -k * (x - x0) where:
    - k is the spring constant
    - x0 is the rest length of the spring

    All nodes repel each other with a constant force, while connected
    nodes are attracted by spring forces.

    Example:
        layout = SpringLayout(
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
        # Spring-specific parameters
        spring_constant: float = 0.1,
        spring_length: float = 100.0,
        repulsion: float = 10000.0,
        damping: float = 0.5,
        gravity: float = 0.0,
        use_barnes_hut: bool = False,
        barnes_hut_theta: float = 0.5,
    ) -> None:
        """
        Initialize Spring layout.

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
            spring_constant: Spring stiffness (higher = stronger attraction).
            spring_length: Natural/rest length of springs (ideal edge distance).
            repulsion: Repulsion strength between all node pairs.
            damping: Velocity damping factor (0 = no damping, 1 = full damping).
            gravity: Gravity strength toward canvas center.
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

        # Spring-specific configuration
        self._spring_constant: float = float(spring_constant)
        self._spring_length: float = float(spring_length)
        self._repulsion: float = float(repulsion)
        self._damping: float = max(0.0, min(1.0, float(damping)))
        self._gravity: float = float(gravity)

        # Barnes-Hut optimization
        self._use_barnes_hut: bool = bool(use_barnes_hut)
        self._barnes_hut_theta: float = max(0.0, float(barnes_hut_theta))

        # Internal state
        self._vel_x: Optional[np.ndarray] = None
        self._vel_y: Optional[np.ndarray] = None
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def spring_constant(self) -> float:
        """Get spring constant (stiffness)."""
        return self._spring_constant

    @spring_constant.setter
    def spring_constant(self, value: float) -> None:
        """Set spring constant."""
        self._spring_constant = float(value)

    @property
    def spring_length(self) -> float:
        """Get natural/rest length of springs."""
        return self._spring_length

    @spring_length.setter
    def spring_length(self, value: float) -> None:
        """Set natural/rest length of springs."""
        self._spring_length = float(value)

    @property
    def repulsion(self) -> float:
        """Get repulsion strength."""
        return self._repulsion

    @repulsion.setter
    def repulsion(self, value: float) -> None:
        """Set repulsion strength."""
        self._repulsion = float(value)

    @property
    def damping(self) -> float:
        """Get damping factor."""
        return self._damping

    @damping.setter
    def damping(self, value: float) -> None:
        """Set damping factor (clamped to [0, 1])."""
        self._damping = max(0.0, min(1.0, float(value)))

    @property
    def gravity(self) -> float:
        """Get gravity strength toward center."""
        return self._gravity

    @gravity.setter
    def gravity(self, value: float) -> None:
        """Set gravity strength."""
        self._gravity = float(value)

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

    def run(self, **kwargs: Any) -> "SpringLayout":
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

        # Initialize velocity arrays
        n = len(self._nodes)
        self._vel_x = np.zeros(n)
        self._vel_y = np.zeros(n)
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
        assert self._vel_x is not None
        assert self._vel_y is not None

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        if n == 0:
            return True

        # Calculate forces
        force_x = np.zeros(n)
        force_y = np.zeros(n)

        # Repulsive forces
        if self._use_barnes_hut and n > 50:
            self._compute_repulsive_barnes_hut(force_x, force_y)
        else:
            self._compute_repulsive_naive(n, force_x, force_y)

        # Spring forces along edges (Hooke's law)
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)

            dx = self._nodes[tgt].x - self._nodes[src].x
            dy = self._nodes[tgt].y - self._nodes[src].y
            dist = math.sqrt(dx * dx + dy * dy) if (dx != 0 or dy != 0) else 0.0001

            # Use link length if specified, otherwise default spring length
            rest_length = link.length if link.length else self._spring_length

            # Hooke's law: F = k * (d - rest_length)
            displacement = dist - rest_length
            force = self._spring_constant * displacement

            if dist > 0:
                fx = (dx / dist) * force
                fy = (dy / dist) * force

                force_x[src] += fx
                force_y[src] += fy
                force_x[tgt] -= fx
                force_y[tgt] -= fy

        # Apply gravity toward center
        if self._gravity > 0:
            cx = self._canvas_size[0] / 2
            cy = self._canvas_size[1] / 2
            for i in range(n):
                dx = cx - self._nodes[i].x
                dy = cy - self._nodes[i].y
                force_x[i] += self._gravity * dx
                force_y[i] += self._gravity * dy

        # Update velocities and positions
        total_movement = 0.0
        for i in range(n):
            node = self._nodes[i]

            if node.fixed:
                continue

            # Update velocity with damping
            self._vel_x[i] = (self._vel_x[i] + force_x[i]) * (1 - self._damping)
            self._vel_y[i] = (self._vel_y[i] + force_y[i]) * (1 - self._damping)

            # Update position
            node.x += self._vel_x[i]
            node.y += self._vel_y[i]

            total_movement += abs(self._vel_x[i]) + abs(self._vel_y[i])

        self._iteration += 1
        self._alpha = 1.0 - (self._iteration / self._iterations)

        # Check convergence based on total movement
        converged = total_movement < 0.01 * n

        # Fire tick event
        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": total_movement})

        return converged

    def _compute_repulsive_naive(self, n: int, force_x: np.ndarray, force_y: np.ndarray) -> None:
        """Compute repulsive forces using O(n^2) pairwise calculation."""
        for i in range(n):
            for j in range(i + 1, n):
                dx = self._nodes[i].x - self._nodes[j].x
                dy = self._nodes[i].y - self._nodes[j].y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001

                # Coulomb's law: F = k / d^2
                if dist > 0:
                    force = self._repulsion / dist_sq
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    force_x[i] += fx
                    force_y[i] += fy
                    force_x[j] -= fx
                    force_y[j] -= fy

    def _compute_repulsive_barnes_hut(self, force_x: np.ndarray, force_y: np.ndarray) -> None:
        """Compute repulsive forces using Barnes-Hut O(n log n) approximation."""
        # Build quadtree from current node positions
        tree = QuadTree.from_nodes(self._nodes, padding=10.0, theta=self._barnes_hut_theta)

        # Calculate force on each node using the tree
        # We use the tree's structure but compute Coulomb forces (k/d^2)
        for i, node in enumerate(self._nodes):
            body = Body(node.x, node.y, mass=1.0, index=i)
            fx, fy = self._calculate_coulomb_force(tree.root, body)
            force_x[i] += fx
            force_y[i] += fy

    def _calculate_coulomb_force(self, node: Any, body: Body) -> tuple[float, float]:
        """Calculate Coulomb force on body from quadtree node."""
        if node is None or node.is_empty():
            return 0.0, 0.0

        # Skip self-interaction
        if node.is_leaf() and node.body is not None:
            if node.body.index == body.index:
                return 0.0, 0.0

        dx = body.x - node.center_of_mass_x
        dy = body.y - node.center_of_mass_y
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1e-10:
            return 0.0, 0.0

        dist = math.sqrt(dist_sq)

        # Barnes-Hut criterion: s/d < theta
        if node.is_leaf() or (node.half_size * 2 / dist) < self._barnes_hut_theta:
            # Coulomb's law: F = k * m / d^2
            force = self._repulsion * node.total_mass / dist_sq
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            return fx, fy

        # Recurse into children
        fx, fy = 0.0, 0.0
        if node.children:
            for child in node.children:
                if child is not None:
                    cfx, cfy = self._calculate_coulomb_force(child, body)
                    fx += cfx
                    fy += cfy
        return fx, fy


__all__ = ["SpringLayout"]

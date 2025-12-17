"""
Simple spring-based force-directed layout.

A basic force-directed algorithm using Hooke's law for spring forces.
Simpler than Fruchterman-Reingold, useful as a baseline or for small graphs.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self

from ..base import IterativeLayout
from ..types import EventType


class SpringLayout(IterativeLayout):
    """
    Simple spring-based force-directed layout.

    Uses Hooke's law: F = -k * (x - x0) where:
    - k is the spring constant
    - x0 is the rest length of the spring

    All nodes repel each other with a constant force, while connected
    nodes are attracted by spring forces.

    Example:
        layout = (SpringLayout()
            .nodes([{'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}])
            .links([{'source': 0, 'target': 1}, {'source': 1, 'target': 2}])
            .size([800, 600])
            .start())
    """

    def __init__(self) -> None:
        super().__init__()
        self._spring_constant: float = 0.1
        self._spring_length: float = 100.0
        self._repulsion: float = 10000.0
        self._damping: float = 0.5
        self._gravity: float = 0.0

        # Internal state
        self._vel_x: Optional[np.ndarray] = None
        self._vel_y: Optional[np.ndarray] = None
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Configuration Methods (Fluent API)
    # -------------------------------------------------------------------------

    def spring_constant(self, k: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the spring constant.

        Higher values make springs stiffer (stronger attraction).

        Args:
            k: Spring constant. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if k is None:
            return self._spring_constant
        self._spring_constant = float(k)
        return self

    def spring_length(self, length: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the natural (rest) length of springs.

        This is the ideal distance between connected nodes.

        Args:
            length: Spring rest length. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if length is None:
            return self._spring_length
        self._spring_length = float(length)
        return self

    def repulsion(self, r: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the repulsion strength.

        Controls how strongly all nodes repel each other.

        Args:
            r: Repulsion strength. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if r is None:
            return self._repulsion
        self._repulsion = float(r)
        return self

    def damping(self, d: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set the damping factor.

        Damping reduces velocity each iteration (0 = no damping, 1 = full damping).

        Args:
            d: Damping factor (0 to 1). If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if d is None:
            return self._damping
        self._damping = max(0.0, min(1.0, float(d)))
        return self

    def gravity(self, g: Optional[float] = None) -> Union[float, Self]:
        """
        Get or set gravity toward center.

        Args:
            g: Gravity strength. If None, returns current value.

        Returns:
            Current value or self for chaining.
        """
        if g is None:
            return self._gravity
        self._gravity = float(g)
        return self

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

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
        self._initialize_indices()

        iterations = kwargs.get('iterations', self._iterations)
        random_init = kwargs.get('random_init', True)
        center = kwargs.get('center_graph', True)

        if random_init:
            self._initialize_positions(random_init=True)

        # Initialize velocity arrays
        n = len(self._nodes)
        self._vel_x = np.zeros(n)
        self._vel_y = np.zeros(n)
        self._iteration = 0
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

        # Repulsive forces between all pairs
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
        self.trigger({
            'type': EventType.tick,
            'alpha': self._alpha,
            'stress': total_movement
        })

        return converged


__all__ = ["SpringLayout"]

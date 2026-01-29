"""
ForceAtlas2 force-directed layout algorithm.

Based on the paper:
"ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization
Designed for the Gephi Software" by Jacomy, Venturini, Heymann, and Bastian (2014)

Key features:
- Degree-weighted repulsion (hubs repel more strongly)
- Adaptive per-node speed based on swing/traction
- No temperature cooling - uses swing tolerance for convergence
- LinLog mode for tighter clusters
- Optional overlap prevention
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
    from .. import _speedups

    # Check that FA2-specific functions exist (module may be old compiled version)
    _HAS_CYTHON = hasattr(_speedups, "_compute_fa2_repulsive_forces")
except ImportError:
    _HAS_CYTHON = False


class ForceAtlas2Layout(IterativeLayout):
    """
    ForceAtlas2 force-directed graph layout.

    This algorithm positions nodes using forces with several key differences
    from Fruchterman-Reingold:
    - Repulsion is degree-weighted: hubs repel more strongly
    - Uses adaptive per-node speed based on swing/traction instead of temperature
    - LinLog mode produces tighter clusters
    - Strong gravity mode prevents component drift

    Example:
        layout = ForceAtlas2Layout(
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
        # ForceAtlas2-specific parameters
        scaling: float = 2.0,
        gravity: float = 1.0,
        strong_gravity_mode: bool = False,
        linlog_mode: bool = False,
        prevent_overlap: bool = False,
        edge_weight_influence: float = 1.0,
        tolerance: float = 1.0,
        use_barnes_hut: bool = True,
        barnes_hut_theta: float = 1.2,
    ) -> None:
        """
        Initialize ForceAtlas2 layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            alpha: Initial alpha (0 to 1)
            alpha_min: Minimum alpha for convergence threshold
            alpha_decay: Alpha decay rate per iteration (0 to 1)
            iterations: Maximum number of iterations
            scaling: Repulsion scaling factor. Higher values spread nodes more. Default 2.0.
            gravity: Gravity strength toward center. Default 1.0.
            strong_gravity_mode: If True, gravity is constant regardless of distance.
                If False (default), gravity scales with distance from center.
            linlog_mode: If True, use log attraction for tighter clusters.
                If False (default), use linear attraction.
            prevent_overlap: If True, consider node sizes in repulsion. Default False.
            edge_weight_influence: How much edge weight affects attraction (0 to 1).
                0 = ignore weights, 1 = full weight. Default 1.0.
            tolerance: Swing tolerance for adaptive speed. Higher = faster but less stable.
                Default 1.0.
            use_barnes_hut: Enable Barnes-Hut O(n log n) approximation. Default True.
            barnes_hut_theta: Barnes-Hut accuracy parameter. Higher = faster but less
                accurate. FA2 uses 1.2 by default (higher than FR's 0.5).
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

        # ForceAtlas2-specific configuration
        self._scaling: float = float(scaling)
        self._gravity: float = float(gravity)
        self._strong_gravity_mode: bool = bool(strong_gravity_mode)
        self._linlog_mode: bool = bool(linlog_mode)
        self._prevent_overlap: bool = bool(prevent_overlap)
        self._edge_weight_influence: float = max(0.0, min(1.0, float(edge_weight_influence)))
        self._tolerance: float = max(0.1, float(tolerance))

        # Barnes-Hut optimization
        self._use_barnes_hut: bool = bool(use_barnes_hut)
        self._barnes_hut_theta: float = max(0.0, float(barnes_hut_theta))

        # Convergence threshold for total displacement
        self._convergence_threshold: float = 0.01

        # Internal state (initialized in run())
        self._pos_x: Optional[np.ndarray] = None
        self._pos_y: Optional[np.ndarray] = None
        self._disp_x: Optional[np.ndarray] = None
        self._disp_y: Optional[np.ndarray] = None
        self._prev_disp_x: Optional[np.ndarray] = None
        self._prev_disp_y: Optional[np.ndarray] = None
        self._degrees: Optional[np.ndarray] = None
        self._sizes: Optional[np.ndarray] = None
        self._fixed: Optional[np.ndarray] = None
        self._sources: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._global_speed: float = 1.0
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def scaling(self) -> float:
        """Get repulsion scaling factor."""
        return self._scaling

    @scaling.setter
    def scaling(self, value: float) -> None:
        """Set repulsion scaling factor."""
        self._scaling = float(value)

    @property
    def gravity(self) -> float:
        """Get gravity strength."""
        return self._gravity

    @gravity.setter
    def gravity(self, value: float) -> None:
        """Set gravity strength."""
        self._gravity = float(value)

    @property
    def strong_gravity_mode(self) -> bool:
        """Get whether strong gravity mode is enabled."""
        return self._strong_gravity_mode

    @strong_gravity_mode.setter
    def strong_gravity_mode(self, value: bool) -> None:
        """Set strong gravity mode."""
        self._strong_gravity_mode = bool(value)

    @property
    def linlog_mode(self) -> bool:
        """Get whether LinLog mode is enabled."""
        return self._linlog_mode

    @linlog_mode.setter
    def linlog_mode(self, value: bool) -> None:
        """Set LinLog mode."""
        self._linlog_mode = bool(value)

    @property
    def prevent_overlap(self) -> bool:
        """Get whether overlap prevention is enabled."""
        return self._prevent_overlap

    @prevent_overlap.setter
    def prevent_overlap(self, value: bool) -> None:
        """Set overlap prevention."""
        self._prevent_overlap = bool(value)

    @property
    def edge_weight_influence(self) -> float:
        """Get edge weight influence."""
        return self._edge_weight_influence

    @edge_weight_influence.setter
    def edge_weight_influence(self, value: float) -> None:
        """Set edge weight influence (clamped to [0, 1])."""
        self._edge_weight_influence = max(0.0, min(1.0, float(value)))

    @property
    def tolerance(self) -> float:
        """Get swing tolerance for adaptive speed."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set swing tolerance (minimum 0.1)."""
        self._tolerance = max(0.1, float(value))

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

    def run(self, **kwargs: Any) -> "ForceAtlas2Layout":
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

        # Initialize arrays
        n = len(self._nodes)
        m = len(self._links)

        self._pos_x = np.zeros(n, dtype=np.float64)
        self._pos_y = np.zeros(n, dtype=np.float64)
        self._disp_x = np.zeros(n, dtype=np.float64)
        self._disp_y = np.zeros(n, dtype=np.float64)
        self._prev_disp_x = np.zeros(n, dtype=np.float64)
        self._prev_disp_y = np.zeros(n, dtype=np.float64)
        self._degrees = np.zeros(n, dtype=np.float64)
        self._sizes = np.zeros(n, dtype=np.float64)
        self._fixed = np.zeros(n, dtype=np.uint8)
        self._sources = np.zeros(m, dtype=np.int32)
        self._targets = np.zeros(m, dtype=np.int32)
        self._weights = np.ones(m, dtype=np.float64)

        # Initialize position, size, and fixed arrays
        for i, node in enumerate(self._nodes):
            self._pos_x[i] = node.x
            self._pos_y[i] = node.y
            self._fixed[i] = 1 if node.fixed else 0
            # Get node size for overlap prevention
            width = getattr(node, "width", 10.0) or 10.0
            height = getattr(node, "height", 10.0) or 10.0
            self._sizes[i] = (width + height) / 4.0  # radius approximation

        # Initialize edge arrays and compute degrees
        for e, link in enumerate(self._links):
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            self._sources[e] = src
            self._targets[e] = tgt
            self._degrees[src] += 1
            self._degrees[tgt] += 1
            # Get edge weight
            weight = getattr(link, "weight", 1.0)
            if weight is None:
                weight = 1.0
            self._weights[e] = float(weight)

        self._iteration = 0
        self._alpha = 1.0
        self._global_speed = 1.0

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
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._prev_disp_x is not None
        assert self._prev_disp_y is not None
        assert self._degrees is not None
        assert self._sizes is not None
        assert self._fixed is not None
        assert self._sources is not None
        assert self._targets is not None
        assert self._weights is not None

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        m = len(self._links)
        if n == 0:
            return True

        # Sync positions from nodes to arrays
        for i, node in enumerate(self._nodes):
            self._pos_x[i] = node.x
            self._pos_y[i] = node.y

        # Save previous displacements for swing calculation
        self._prev_disp_x[:] = self._disp_x
        self._prev_disp_y[:] = self._disp_y

        # Reset displacements
        self._disp_x.fill(0)
        self._disp_y.fill(0)

        # Compute repulsive forces (degree-weighted)
        if _HAS_CYTHON:
            if self._use_barnes_hut and n > 50:
                _speedups._compute_fa2_repulsive_forces_barnes_hut(
                    self._pos_x,
                    self._pos_y,
                    self._disp_x,
                    self._disp_y,
                    self._degrees,
                    self._scaling,
                    n,
                    self._barnes_hut_theta,
                )
            elif self._prevent_overlap:
                _speedups._compute_fa2_repulsive_forces_overlap(
                    self._pos_x,
                    self._pos_y,
                    self._disp_x,
                    self._disp_y,
                    self._degrees,
                    self._sizes,
                    self._scaling,
                    n,
                )
            else:
                _speedups._compute_fa2_repulsive_forces(
                    self._pos_x,
                    self._pos_y,
                    self._disp_x,
                    self._disp_y,
                    self._degrees,
                    self._scaling,
                    n,
                )
        else:
            if self._use_barnes_hut and n > 50:
                self.compute_repulsive_barnes_hut()
            else:
                self.compute_repulsive_naive()

        # Compute attractive forces
        if _HAS_CYTHON and m > 0:
            _speedups._compute_fa2_attractive_forces(
                self._pos_x,
                self._pos_y,
                self._disp_x,
                self._disp_y,
                self._sources,
                self._targets,
                self._weights,
                self._edge_weight_influence,
                self._linlog_mode,
                m,
            )
        else:
            self.compute_attractive_forces()

        # Compute gravity forces
        if _HAS_CYTHON:
            cx = self._canvas_size[0] / 2
            cy = self._canvas_size[1] / 2
            _speedups._compute_fa2_gravity(
                self._pos_x,
                self._pos_y,
                self._disp_x,
                self._disp_y,
                self._degrees,
                self._gravity,
                cx,
                cy,
                self._strong_gravity_mode,
                n,
            )
        else:
            self.compute_gravity_forces()

        # Calculate swing and traction, update speeds
        if _HAS_CYTHON:
            total_swing, total_traction = _speedups._compute_fa2_swing_traction(
                self._disp_x,
                self._disp_y,
                self._prev_disp_x,
                self._prev_disp_y,
                self._degrees,
                n,
            )
        else:
            total_swing, total_traction = self.compute_swing_traction()

        # Update global speed
        if total_swing > 0:
            self._global_speed = self._tolerance * total_traction / total_swing
        else:
            self._global_speed = 1.0

        # Apply displacements with adaptive per-node speeds
        if _HAS_CYTHON:
            total_displacement = _speedups._apply_fa2_displacements(
                self._pos_x,
                self._pos_y,
                self._disp_x,
                self._disp_y,
                self._prev_disp_x,
                self._prev_disp_y,
                self._fixed,
                self._global_speed,
                10.0,  # max_displacement
                n,
            )
            # Sync positions back to nodes
            for i, node in enumerate(self._nodes):
                node.x = float(self._pos_x[i])
                node.y = float(self._pos_y[i])
        else:
            total_displacement = self.apply_displacements(total_swing)

        self._iteration += 1

        # Update alpha based on convergence
        avg_displacement = total_displacement / n if n > 0 else 0
        canvas_diag = math.sqrt(self._canvas_size[0] ** 2 + self._canvas_size[1] ** 2)
        self._alpha = min(1.0, avg_displacement / (canvas_diag * 0.01))

        # Fire tick event
        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": None})

        # Check convergence
        if avg_displacement < self._convergence_threshold:
            return True

        return False

    def compute_repulsive_naive(self) -> None:
        """Compute degree-weighted repulsive forces using O(n^2) pairwise calculation."""
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._degrees is not None
        assert self._sizes is not None

        n = len(self._nodes)
        scaling = self._scaling

        for i in range(n):
            for j in range(i + 1, n):
                dx = self._pos_x[i] - self._pos_x[j]
                dy = self._pos_y[i] - self._pos_y[j]
                dist_sq = dx * dx + dy * dy

                if self._prevent_overlap:
                    # Subtract node radii from distance
                    overlap_dist = self._sizes[i] + self._sizes[j]
                    dist = math.sqrt(dist_sq) - overlap_dist
                    if dist < 0:
                        dist = 0.01  # Prevent negative/zero distance
                else:
                    dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.01

                if dist > 0:
                    # ForceAtlas2 repulsion: scaling * (deg_i + 1) * (deg_j + 1) / distance
                    deg_factor = (self._degrees[i] + 1) * (self._degrees[j] + 1)
                    force = scaling * deg_factor / dist

                    # Normalize direction
                    if dist_sq > 0:
                        inv_dist = 1.0 / math.sqrt(dist_sq)
                        fx = dx * inv_dist * force
                        fy = dy * inv_dist * force
                    else:
                        fx = force
                        fy = 0

                    self._disp_x[i] += fx
                    self._disp_y[i] += fy
                    self._disp_x[j] -= fx
                    self._disp_y[j] -= fy

    def compute_repulsive_barnes_hut(self) -> None:
        """Compute repulsive forces using Barnes-Hut O(n log n) approximation."""
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._degrees is not None

        # Build quadtree with degree as mass
        tree = QuadTree.from_nodes(self._nodes, padding=10.0, theta=self._barnes_hut_theta)

        scaling = self._scaling

        for i, node in enumerate(self._nodes):
            # Use degree+1 as mass for degree-weighted repulsion
            deg_i: float = float(self._degrees[i] + 1)
            body = Body(node.x, node.y, mass=deg_i, index=i)
            # The quadtree returns force with repulsion_constant factor
            fx, fy = tree.calculate_force(body, repulsion_constant=scaling)
            self._disp_x[i] += fx
            self._disp_y[i] += fy

    def compute_attractive_forces(self) -> None:
        """Compute attractive forces along edges."""
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._sources is not None
        assert self._targets is not None
        assert self._weights is not None

        for e in range(len(self._links)):
            src = self._sources[e]
            tgt = self._targets[e]

            dx = self._pos_x[src] - self._pos_x[tgt]
            dy = self._pos_y[src] - self._pos_y[tgt]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.01:
                continue

            # Apply edge weight influence
            weight = self._weights[e]
            if self._edge_weight_influence < 1.0:
                weight = pow(weight, self._edge_weight_influence)

            # ForceAtlas2 attraction
            if self._linlog_mode:
                # LinLog: log(1 + distance) for tighter clusters
                force = weight * math.log(1 + dist)
            else:
                # Linear: just distance
                force = weight * dist

            # Normalize and apply
            fx = (dx / dist) * force
            fy = (dy / dist) * force

            self._disp_x[src] -= fx
            self._disp_y[src] -= fy
            self._disp_x[tgt] += fx
            self._disp_y[tgt] += fy

    def compute_gravity_forces(self) -> None:
        """Compute gravity forces toward center."""
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._degrees is not None

        if self._gravity <= 0:
            return

        n = len(self._nodes)
        cx = self._canvas_size[0] / 2
        cy = self._canvas_size[1] / 2

        for i in range(n):
            dx = cx - self._pos_x[i]
            dy = cy - self._pos_y[i]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.01:
                continue

            # Gravity is degree-weighted: gravity * (deg + 1)
            deg_factor: float = float(self._degrees[i] + 1)

            if self._strong_gravity_mode:
                # Strong gravity: constant force regardless of distance
                force = self._gravity * deg_factor
            else:
                # Normal gravity: force scales with distance
                force = self._gravity * deg_factor * dist

            # Normalize and apply
            self._disp_x[i] += (dx / dist) * force
            self._disp_y[i] += (dy / dist) * force

    def compute_swing_traction(self) -> tuple[float, float]:
        """
        Compute global swing and traction for adaptive speed.

        Swing measures oscillation (force direction changing).
        Traction measures consistent movement (force direction stable).
        """
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._prev_disp_x is not None
        assert self._prev_disp_y is not None
        assert self._degrees is not None

        n = len(self._nodes)
        total_swing = 0.0
        total_traction = 0.0

        for i in range(n):
            # Current and previous force vectors
            fx = self._disp_x[i]
            fy = self._disp_y[i]
            pfx = self._prev_disp_x[i]
            pfy = self._prev_disp_y[i]

            # Swing: |F(t) - F(t-1)|
            swing_x = fx - pfx
            swing_y = fy - pfy
            swing = math.sqrt(swing_x * swing_x + swing_y * swing_y)

            # Traction: |F(t) + F(t-1)| / 2
            trac_x = (fx + pfx) / 2
            trac_y = (fy + pfy) / 2
            traction = math.sqrt(trac_x * trac_x + trac_y * trac_y)

            # Weight by degree
            deg_factor: float = float(self._degrees[i] + 1)
            total_swing += deg_factor * swing
            total_traction += deg_factor * traction

        return total_swing, total_traction

    def apply_displacements(self, total_swing: float) -> float:
        """
        Apply displacements with adaptive per-node speeds.

        Returns total displacement for convergence check.
        """
        assert self._pos_x is not None
        assert self._pos_y is not None
        assert self._disp_x is not None
        assert self._disp_y is not None
        assert self._prev_disp_x is not None
        assert self._prev_disp_y is not None
        assert self._fixed is not None
        assert self._degrees is not None

        n = len(self._nodes)
        total_displacement = 0.0

        for i in range(n):
            # Skip fixed nodes
            if self._fixed[i]:
                continue

            fx = self._disp_x[i]
            fy = self._disp_y[i]
            force_mag = math.sqrt(fx * fx + fy * fy)

            if force_mag < 0.01:
                continue

            # Compute per-node swing
            pfx = self._prev_disp_x[i]
            pfy = self._prev_disp_y[i]
            swing_x = fx - pfx
            swing_y = fy - pfy
            node_swing = math.sqrt(swing_x * swing_x + swing_y * swing_y)

            # Per-node speed: global_speed / (1 + global_speed * sqrt(swing))
            node_speed = self._global_speed / (1 + self._global_speed * math.sqrt(node_swing))

            # Limit displacement to prevent instability
            max_displacement = 10.0  # Reasonable max per iteration
            displacement = min(force_mag * node_speed, max_displacement)

            # Apply displacement
            dx = (fx / force_mag) * displacement
            dy = (fy / force_mag) * displacement

            self._pos_x[i] += dx
            self._pos_y[i] += dy
            self._nodes[i].x = float(self._pos_x[i])
            self._nodes[i].y = float(self._pos_y[i])

            total_displacement += displacement

        return total_displacement


__all__ = ["ForceAtlas2Layout"]

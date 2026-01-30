"""
Yifan Hu Multilevel force-directed layout algorithm.

Based on the paper:
"Efficient and High Quality Force-Directed Graph Drawing" by Yifan Hu (2005)

Key features:
- Spring-electrical force model with repulsion and attraction
- Multilevel coarsening to escape local minima
- Adaptive step length control
- Barnes-Hut O(n log n) approximation for repulsive forces
"""

from __future__ import annotations

import math
import random
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

# Try to import Cython-optimized functions, fallback to pure Python if unavailable
try:
    from .. import _speedups

    # Yifan Hu uses FR-compatible force formulas, check for those functions
    _HAS_CYTHON = hasattr(_speedups, "_compute_repulsive_forces")
except ImportError:
    _HAS_CYTHON = False


class YifanHuLayout(IterativeLayout):
    """
    Yifan Hu Multilevel force-directed graph layout.

    This algorithm uses a multilevel approach to effectively overcome local
    minima, combined with Barnes-Hut approximation for efficient force
    calculation. It's suitable for medium to large graphs (1K-100K nodes).

    The algorithm has three phases:
    1. Coarsening: Collapse graph iteratively into coarser graphs
    2. Layout: Layout the coarsest graph with force-directed algorithm
    3. Refinement: Uncoarsen and refine positions at each level

    Example:
        layout = YifanHuLayout(
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
        # YifanHu-specific parameters
        optimal_distance: Optional[float] = None,
        relative_strength: float = 0.2,
        step_ratio: float = 0.9,
        convergence_tolerance: float = 0.01,
        use_barnes_hut: bool = True,
        barnes_hut_theta: float = 1.2,
        # Multilevel parameters
        coarsening_threshold: float = 0.75,
        min_coarsest_size: int = 10,
        level_iterations: int = 50,
    ) -> None:
        """
        Initialize Yifan Hu layout.

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
            iterations: Maximum number of iterations per level
            optimal_distance: Optimal distance K between connected nodes.
                If None, computed automatically from graph.
            relative_strength: C parameter controlling repulsion vs attraction.
                Default 0.2 as in the paper.
            step_ratio: Step decay/growth ratio t. Default 0.9.
            convergence_tolerance: Tolerance for convergence check. Default 0.01.
            use_barnes_hut: Enable Barnes-Hut O(n log n) approximation. Default True.
            barnes_hut_theta: Barnes-Hut accuracy parameter. 1.2 as in paper
                (higher = faster but less accurate).
            coarsening_threshold: Stop coarsening if ratio > this. Default 0.75.
            min_coarsest_size: Minimum vertices in coarsest graph. Default 10.
            level_iterations: Iterations per refinement level. Default 50.
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

        # YifanHu-specific configuration
        self._optimal_distance: Optional[float] = optimal_distance
        self._relative_strength: float = float(relative_strength)
        self._step_ratio: float = max(0.1, min(0.99, float(step_ratio)))
        self._convergence_tolerance: float = float(convergence_tolerance)

        # Barnes-Hut optimization
        self._use_barnes_hut: bool = bool(use_barnes_hut)
        self._barnes_hut_theta: float = max(0.0, float(barnes_hut_theta))

        # Multilevel parameters
        self._coarsening_threshold: float = max(0.5, min(0.99, float(coarsening_threshold)))
        self._min_coarsest_size: int = max(2, int(min_coarsest_size))
        self._level_iterations: int = max(10, int(level_iterations))

        # Internal state (initialized in run())
        self._pos_x: Optional[np.ndarray] = None
        self._pos_y: Optional[np.ndarray] = None
        self._disp_x: Optional[np.ndarray] = None
        self._disp_y: Optional[np.ndarray] = None
        self._fixed: Optional[np.ndarray] = None
        self._sources: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None

        # Adaptive step control
        self._step: float = 1.0
        self._energy: float = float("inf")
        self._progress: int = 0
        self._iteration: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def optimal_distance(self) -> float:
        """Get optimal distance K between connected nodes."""
        if self._optimal_distance is not None:
            return self._optimal_distance
        return self._compute_optimal_distance()

    @optimal_distance.setter
    def optimal_distance(self, value: Optional[float]) -> None:
        """Set optimal distance K."""
        self._optimal_distance = float(value) if value is not None else None

    @property
    def relative_strength(self) -> float:
        """Get relative strength C of repulsive force."""
        return self._relative_strength

    @relative_strength.setter
    def relative_strength(self, value: float) -> None:
        """Set relative strength C."""
        self._relative_strength = float(value)

    @property
    def step_ratio(self) -> float:
        """Get step decay/growth ratio t."""
        return self._step_ratio

    @step_ratio.setter
    def step_ratio(self, value: float) -> None:
        """Set step ratio (clamped to [0.1, 0.99])."""
        self._step_ratio = max(0.1, min(0.99, float(value)))

    @property
    def convergence_tolerance(self) -> float:
        """Get convergence tolerance."""
        return self._convergence_tolerance

    @convergence_tolerance.setter
    def convergence_tolerance(self, value: float) -> None:
        """Set convergence tolerance."""
        self._convergence_tolerance = float(value)

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

    @property
    def coarsening_threshold(self) -> float:
        """Get coarsening threshold rho."""
        return self._coarsening_threshold

    @coarsening_threshold.setter
    def coarsening_threshold(self, value: float) -> None:
        """Set coarsening threshold."""
        self._coarsening_threshold = max(0.5, min(0.99, float(value)))

    @property
    def min_coarsest_size(self) -> int:
        """Get minimum coarsest graph size."""
        return self._min_coarsest_size

    @min_coarsest_size.setter
    def min_coarsest_size(self, value: int) -> None:
        """Set minimum coarsest graph size."""
        self._min_coarsest_size = max(2, int(value))

    @property
    def level_iterations(self) -> int:
        """Get iterations per refinement level."""
        return self._level_iterations

    @level_iterations.setter
    def level_iterations(self, value: int) -> None:
        """Set iterations per refinement level."""
        self._level_iterations = max(10, int(value))

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def _compute_optimal_distance(self) -> float:
        """Compute optimal distance K based on canvas area and node count."""
        area = self._canvas_size[0] * self._canvas_size[1]
        n = max(1, len(self._nodes))
        return math.sqrt(area / n)

    def run(self, **kwargs: Any) -> "YifanHuLayout":
        """
        Run the multilevel layout algorithm.

        Keyword Args:
            random_init: Initialize positions randomly (default: True)
            center_graph: Center graph after completion (default: True)

        Returns:
            self for chaining
        """
        self._initialize_indices()

        random_init = kwargs.get("random_init", True)
        center = kwargs.get("center_graph", True)

        n = len(self._nodes)

        if n == 0:
            return self

        # Compute optimal distance if not set
        if self._optimal_distance is None:
            self._optimal_distance = self._compute_optimal_distance()

        # Build coarsening hierarchy
        levels = self._build_coarsening_hierarchy()

        # Fire start event
        self.trigger({"type": EventType.start, "alpha": self._alpha})

        if len(levels) == 1:
            # No coarsening needed, just run single level
            self._run_single_level(random_init=random_init)
        else:
            # Run multilevel algorithm
            self._run_multilevel(levels, random_init=random_init)

        if center:
            self._center_graph()

        # Fire end event
        self.trigger({"type": EventType.end, "alpha": 0.0})

        return self

    def _build_coarsening_hierarchy(self) -> list[dict[str, Any]]:
        """
        Build hierarchy of coarsened graphs.

        Returns:
            List of level dictionaries, from finest (0) to coarsest.
        """
        n = len(self._nodes)

        # Level 0 is the original graph
        levels = [
            {
                "n": n,
                "adjacency": self._build_adjacency(),
                "sources": [self._get_source_index(link) for link in self._links],
                "targets": [self._get_target_index(link) for link in self._links],
                "mapping": list(range(n)),  # Identity mapping
                "parent_mapping": None,
            }
        ]

        current_n = n
        current_adj: list[list[int]] = levels[0]["adjacency"]  # type: ignore[assignment]
        current_sources: list[int] = levels[0]["sources"]  # type: ignore[assignment]
        current_targets: list[int] = levels[0]["targets"]  # type: ignore[assignment]

        while current_n > self._min_coarsest_size:
            # Perform edge collapsing coarsening
            coarsened = self._coarsen_graph(
                current_n, current_adj, current_sources, current_targets
            )

            new_n = coarsened["n"]

            # Check coarsening ratio
            if new_n / current_n > self._coarsening_threshold:
                # Not enough coarsening, stop
                break

            levels.append(coarsened)
            current_n = new_n
            current_adj = coarsened["adjacency"]
            current_sources = coarsened["sources"]
            current_targets = coarsened["targets"]

        return levels

    def _coarsen_graph(
        self,
        n: int,
        adjacency: list[list[int]],
        sources: list[int],
        targets: list[int],
    ) -> dict[str, Any]:
        """
        Coarsen graph using edge collapsing with maximal matching.

        Args:
            n: Number of vertices
            adjacency: Adjacency list
            sources: Edge source indices
            targets: Edge target indices

        Returns:
            Dictionary with coarsened graph information
        """
        # Find maximal matching using greedy heavy-edge matching
        matched = [False] * n
        mapping = [-1] * n  # Maps fine vertex to coarse vertex
        coarse_vertex = 0

        # Process vertices in random order for better matching
        vertex_order = list(range(n))
        random.shuffle(vertex_order)

        for v in vertex_order:
            if matched[v]:
                continue

            # Find best unmatched neighbor (heavy-edge matching)
            best_neighbor = -1
            for neighbor in adjacency[v]:
                if not matched[neighbor]:
                    best_neighbor = neighbor
                    break

            if best_neighbor >= 0:
                # Match v with best_neighbor
                mapping[v] = coarse_vertex
                mapping[best_neighbor] = coarse_vertex
                matched[v] = True
                matched[best_neighbor] = True
            else:
                # No match, v becomes its own coarse vertex
                mapping[v] = coarse_vertex
                matched[v] = True

            coarse_vertex += 1

        new_n = coarse_vertex

        # Build coarse graph edges
        coarse_edges: set[tuple[int, int]] = set()
        for src, tgt in zip(sources, targets):
            cs = mapping[src]
            ct = mapping[tgt]
            if cs != ct:
                edge = (min(cs, ct), max(cs, ct))
                coarse_edges.add(edge)

        new_sources = [e[0] for e in coarse_edges]
        new_targets = [e[1] for e in coarse_edges]

        # Build coarse adjacency
        new_adj: list[list[int]] = [[] for _ in range(new_n)]
        for src, tgt in zip(new_sources, new_targets):
            new_adj[src].append(tgt)
            new_adj[tgt].append(src)

        return {
            "n": new_n,
            "adjacency": new_adj,
            "sources": new_sources,
            "targets": new_targets,
            "mapping": mapping,
            "parent_mapping": None,
        }

    def _run_multilevel(self, levels: list[dict[str, Any]], random_init: bool = True) -> None:
        """
        Run the multilevel algorithm.

        Args:
            levels: List of coarsening levels (finest to coarsest)
            random_init: Whether to initialize coarsest level randomly
        """
        n_levels = len(levels)

        # Layout coarsest level
        coarsest = levels[-1]
        coarsest_n = coarsest["n"]

        # Initialize coarsest positions randomly
        pos_x = np.zeros(coarsest_n, dtype=np.float64)
        pos_y = np.zeros(coarsest_n, dtype=np.float64)

        if random_init:
            w, h = self._canvas_size
            for i in range(coarsest_n):
                pos_x[i] = random.uniform(0, w)
                pos_y[i] = random.uniform(0, h)

        # Compute K for coarsest level (scale by level)
        if self._optimal_distance:
            base_k = self._optimal_distance
        else:
            base_k = self._compute_optimal_distance()

        # Layout coarsest level with adaptive step
        pos_x, pos_y = self._layout_level(
            coarsest_n,
            coarsest["sources"],
            coarsest["targets"],
            pos_x,
            pos_y,
            base_k,
            self._iterations,  # More iterations for coarsest
            adaptive_step=True,
        )

        # Refine from coarsest to finest
        for level_idx in range(n_levels - 2, -1, -1):
            level = levels[level_idx]
            coarser = levels[level_idx + 1]

            level_n = level["n"]
            mapping = coarser["mapping"]

            # Prolongate: interpolate positions from coarser level
            new_pos_x = np.zeros(level_n, dtype=np.float64)
            new_pos_y = np.zeros(level_n, dtype=np.float64)

            # Count how many fine vertices map to each coarse vertex
            counts = np.zeros(coarsest_n, dtype=np.float64)
            for fine_v, coarse_v in enumerate(mapping):
                counts[coarse_v] += 1

            # Assign positions with small random perturbation
            for fine_v, coarse_v in enumerate(mapping):
                new_pos_x[fine_v] = pos_x[coarse_v] + random.gauss(0, 1)
                new_pos_y[fine_v] = pos_y[coarse_v] + random.gauss(0, 1)

            # Scale K based on diameter ratio (simplified: use sqrt of vertex ratio)
            gamma = math.sqrt(level_n / coarsest_n) if coarsest_n > 0 else 1.0
            level_k = base_k / gamma if gamma > 0 else base_k

            # Refine this level with simple cooling
            pos_x, pos_y = self._layout_level(
                level_n,
                level["sources"],
                level["targets"],
                new_pos_x,
                new_pos_y,
                level_k,
                self._level_iterations,
                adaptive_step=False,  # Simple cooling for refinement
            )

            coarsest_n = level_n

        # Copy final positions to nodes
        for i, node in enumerate(self._nodes):
            if not node.fixed:
                node.x = float(pos_x[i])
                node.y = float(pos_y[i])

    def _layout_level(
        self,
        n: int,
        sources: list[int],
        targets: list[int],
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        k: float,
        max_iterations: int,
        adaptive_step: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Layout a single level using force-directed algorithm.

        Args:
            n: Number of vertices
            sources: Edge sources
            targets: Edge targets
            pos_x: Initial x positions
            pos_y: Initial y positions
            k: Optimal distance
            max_iterations: Maximum iterations
            adaptive_step: Use adaptive step control

        Returns:
            Tuple of (pos_x, pos_y) arrays
        """
        if n == 0:
            return pos_x, pos_y

        m = len(sources)
        disp_x = np.zeros(n, dtype=np.float64)
        disp_y = np.zeros(n, dtype=np.float64)

        sources_arr = np.array(sources, dtype=np.int32)
        targets_arr = np.array(targets, dtype=np.int32)

        # Adaptive step control state
        step = k * 0.1  # Initial step proportional to K
        energy = float("inf")
        progress = 0
        t = self._step_ratio

        c = self._relative_strength
        k_sq = k * k

        for iteration in range(max_iterations):
            prev_pos_x = pos_x.copy()
            prev_pos_y = pos_y.copy()

            # Reset displacements
            disp_x.fill(0)
            disp_y.fill(0)

            # Compute repulsive forces
            # Yifan Hu uses C * K² / d, Cython uses k² / d, so pass c * k_sq
            if _HAS_CYTHON:
                if self._use_barnes_hut and n > 50:
                    _speedups._compute_repulsive_forces_barnes_hut(
                        pos_x,
                        pos_y,
                        disp_x,
                        disp_y,
                        c * k_sq,
                        n,
                        self._barnes_hut_theta,
                    )
                else:
                    _speedups._compute_repulsive_forces(pos_x, pos_y, disp_x, disp_y, c * k_sq, n)
            else:
                if self._use_barnes_hut and n > 50:
                    self.compute_repulsive_barnes_hut(pos_x, pos_y, disp_x, disp_y, c, k_sq, n)
                else:
                    self.compute_repulsive_naive(pos_x, pos_y, disp_x, disp_y, c, k_sq, n)

            # Compute attractive forces (d² / K formula matches Cython exactly)
            if _HAS_CYTHON and m > 0:
                _speedups._compute_attractive_forces(
                    pos_x, pos_y, disp_x, disp_y, sources_arr, targets_arr, k, m
                )
            else:
                self.compute_attractive(
                    pos_x, pos_y, disp_x, disp_y, sources_arr, targets_arr, k, m
                )

            # Compute energy (sum of squared forces)
            new_energy = 0.0
            for i in range(n):
                new_energy += disp_x[i] ** 2 + disp_y[i] ** 2

            # Update step using adaptive scheme
            if adaptive_step:
                if new_energy < energy:
                    progress += 1
                    if progress >= 5:
                        progress = 0
                        step = step / t
                else:
                    progress = 0
                    step = t * step
            else:
                # Simple cooling for refinement levels
                step = t * step

            energy = new_energy

            # Apply displacements with step limit
            total_displacement = 0.0
            for i in range(n):
                disp_len = math.sqrt(disp_x[i] ** 2 + disp_y[i] ** 2)
                if disp_len > 0:
                    # Limit displacement by step
                    scale = min(disp_len, step) / disp_len
                    dx = disp_x[i] * scale
                    dy = disp_y[i] * scale
                    pos_x[i] += dx
                    pos_y[i] += dy
                    total_displacement += math.sqrt(dx * dx + dy * dy)

            # Fire tick event
            self._alpha = step / (k * 0.1) if k > 0 else 0
            self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": None})

            # Check convergence
            movement = np.sqrt(np.sum((pos_x - prev_pos_x) ** 2 + (pos_y - prev_pos_y) ** 2))
            if movement < self._convergence_tolerance * k * n:
                break

        return pos_x, pos_y

    def compute_repulsive_naive(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        disp_x: np.ndarray,
        disp_y: np.ndarray,
        c: float,
        k_sq: float,
        n: int,
    ) -> None:
        """Compute repulsive forces using O(n^2) pairwise calculation."""
        # Spring-electrical model: f_r = -C * K^2 / d
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001

                if dist > 0:
                    # Repulsive force: C * K^2 / d
                    force = c * k_sq / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    disp_x[i] += fx
                    disp_y[i] += fy
                    disp_x[j] -= fx
                    disp_y[j] -= fy

    def compute_repulsive_barnes_hut(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        disp_x: np.ndarray,
        disp_y: np.ndarray,
        c: float,
        k_sq: float,
        n: int,
    ) -> None:
        """Compute repulsive forces using Barnes-Hut O(n log n) approximation."""

        # Build quadtree from positions
        # Create temporary node-like objects for QuadTree
        class TempNode:
            def __init__(self, x: float, y: float, idx: int):
                self.x = x
                self.y = y
                self.index = idx

        temp_nodes = [TempNode(float(pos_x[i]), float(pos_y[i]), i) for i in range(n)]
        tree = QuadTree.from_nodes(temp_nodes, padding=10.0, theta=self._barnes_hut_theta)  # type: ignore[arg-type]

        # Calculate force on each node
        for i in range(n):
            body = Body(float(pos_x[i]), float(pos_y[i]), mass=1.0, index=i)
            fx, fy = tree.calculate_force(body, repulsion_constant=c * k_sq)
            disp_x[i] += fx
            disp_y[i] += fy

    def compute_attractive(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        disp_x: np.ndarray,
        disp_y: np.ndarray,
        sources: np.ndarray,
        targets: np.ndarray,
        k: float,
        m: int,
    ) -> None:
        """Compute attractive forces along edges."""
        # Spring-electrical model: f_a = d^2 / K
        for e in range(m):
            src = sources[e]
            tgt = targets[e]

            dx = pos_x[src] - pos_x[tgt]
            dy = pos_y[src] - pos_y[tgt]
            dist_sq = dx * dx + dy * dy
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0001

            if dist > 0:
                # Attractive force: d^2 / K (simplified: d / K for linear)
                force = dist_sq / k
                fx = (dx / dist) * force
                fy = (dy / dist) * force

                disp_x[src] -= fx
                disp_y[src] -= fy
                disp_x[tgt] += fx
                disp_y[tgt] += fy

    def _run_single_level(self, random_init: bool = True) -> None:
        """Run layout without multilevel coarsening (for small graphs)."""
        n = len(self._nodes)

        if n == 0:
            return

        # Initialize arrays
        pos_x = np.zeros(n, dtype=np.float64)
        pos_y = np.zeros(n, dtype=np.float64)

        # Initialize positions
        if random_init:
            self._initialize_positions(random_init=True)

        for i, node in enumerate(self._nodes):
            pos_x[i] = node.x
            pos_y[i] = node.y

        sources = [self._get_source_index(link) for link in self._links]
        targets = [self._get_target_index(link) for link in self._links]

        k = self._optimal_distance if self._optimal_distance else self._compute_optimal_distance()

        # Run layout
        pos_x, pos_y = self._layout_level(
            n, sources, targets, pos_x, pos_y, k, self._iterations, adaptive_step=True
        )

        # Copy positions back to nodes
        for i, node in enumerate(self._nodes):
            if not node.fixed:
                node.x = float(pos_x[i])
                node.y = float(pos_y[i])

    def tick(self) -> bool:
        """
        Perform one iteration of the layout.

        Note: The multilevel algorithm runs to completion in run().
        This method is provided for API compatibility but returns True immediately.

        Returns:
            True (always converged since multilevel runs to completion)
        """
        return True


__all__ = ["YifanHuLayout"]

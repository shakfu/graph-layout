"""
SMACOF stress-majorization layout.

SMACOF ("Scaling by MAjorizing a COmplicated Function") minimizes the same
graph-drawing stress as Kamada-Kawai -- placing nodes so that Euclidean
distances match graph-theoretic (shortest-path) distances -- but optimizes it
by *majorization* rather than the per-node Newton-Raphson steps of
Kamada-Kawai. Each iteration replaces the stress by a simple quadratic upper
bound that touches it at the current layout and moves every node at once to
that bound's global minimum (the Guttman transform). Because each step is the
exact minimizer of a majorizing quadratic, the stress decreases monotonically
and the method converges reliably, typically in fewer iterations than
gradient descent.

Reference:
    Gansner, E. R., Koren, Y., & North, S. (2004). Graph Drawing by Stress
    Majorization. Graph Drawing (GD 2004), LNCS 3383, 239-250.
    https://graphviz.org/documentation/GKN04.pdf
"""

from __future__ import annotations

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


class SMACOFLayout(IterativeLayout):
    """
    Stress-majorization (SMACOF) layout.

    Positions nodes to minimize the weighted stress

        stress(X) = sum_{i<j} w_ij (||x_i - x_j|| - d_ij)^2

    where ``d_ij`` is the ideal distance (edge_length times the graph-theoretic
    distance between i and j) and ``w_ij = d_ij^-2`` is the standard weighting
    that keeps every pair's contribution scale-invariant. Unlike Kamada-Kawai,
    which moves one node at a time, each SMACOF iteration moves *all* nodes to
    the minimizer of a quadratic that majorizes the stress (the Guttman
    transform), so the stress decreases monotonically.

    Example:
        layout = SMACOFLayout(
            nodes=[{}, {}, {}],
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
        # SMACOF-specific parameters
        edge_length: float = 100.0,
        epsilon: float = 1e-4,
        disconnected_distance: Optional[float] = None,
    ) -> None:
        """
        Initialize SMACOF layout.

        Args:
            nodes: List of nodes
            links: List of links
            groups: List of groups
            size: Canvas size as (width, height)
            random_seed: Random seed for reproducible layouts
            on_start: Callback for start event
            on_tick: Callback for tick event
            on_end: Callback for end event
            alpha: Initial alpha/temperature (0 to 1); unused by the majorization
                step but kept for the IterativeLayout interface.
            alpha_min: Minimum alpha (IterativeLayout interface).
            alpha_decay: Alpha decay rate (IterativeLayout interface).
            iterations: Maximum number of majorization iterations.
            edge_length: Ideal length for a single edge. Multi-hop pairs get ideal
                lengths that are multiples of this.
            epsilon: Convergence threshold on the *relative* stress decrease per
                iteration. Iteration stops when (prev - cur) / cur < epsilon.
            disconnected_distance: Ideal graph distance for pairs in different
                components. If None, uses max(diameter * 1.5, n).
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

        # SMACOF-specific configuration
        self._edge_length: float = float(edge_length)
        self._epsilon: float = float(epsilon)
        self._disconnected_distance: Optional[float] = disconnected_distance

        # Internal state (allocated in run())
        self._d_matrix: Optional[np.ndarray] = None  # ideal distances
        self._w_matrix: Optional[np.ndarray] = None  # weights
        self._v_pinv: Optional[np.ndarray] = None  # pseudo-inverse of weighted Laplacian
        self._prev_stress: float = float("inf")
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
        """Get the relative-stress convergence threshold."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set the relative-stress convergence threshold."""
        self._epsilon = float(value)

    @property
    def disconnected_distance(self) -> Optional[float]:
        """Get ideal graph distance for disconnected node pairs."""
        return self._disconnected_distance

    @disconnected_distance.setter
    def disconnected_distance(self, value: Optional[float]) -> None:
        """Set ideal graph distance for disconnected node pairs."""
        self._disconnected_distance = float(value) if value is not None else None

    # -------------------------------------------------------------------------
    # Layout Implementation
    # -------------------------------------------------------------------------

    def _compute_shortest_paths(self) -> np.ndarray:
        """
        Compute all-pairs shortest-path lengths via BFS (unit edge weights).

        Returns:
            Distance matrix where dist[i, j] is the hop count, inf if unreachable.
        """
        n = len(self._nodes)
        dist = np.full((n, n), float("inf"))
        adj = self._build_adjacency()

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
        """Build the ideal-distance, weight, and pseudo-inverse-Laplacian matrices."""
        n = len(self._nodes)

        hop = self._compute_shortest_paths()

        # Largest finite hop distance (graph diameter within a component).
        finite = hop[np.isfinite(hop)]
        max_dist = float(finite.max()) if finite.size else 0.0

        if self._disconnected_distance is not None:
            disconn = self._disconnected_distance
        else:
            disconn = max(max_dist * 1.5, n) if max_dist > 0 else float(n)

        hop = np.where(np.isfinite(hop), hop, disconn)

        # Ideal Euclidean distances and standard w = 1/d^2 weights.
        self._d_matrix = hop * self._edge_length
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (self._d_matrix**2)
        w[self._d_matrix == 0] = 0.0  # zero self-weight (and any zero-distance pair)
        np.fill_diagonal(w, 0.0)
        self._w_matrix = w

        # Weighted Laplacian V (v_ij = -w_ij, v_ii = sum_j w_ij) and its
        # Moore-Penrose pseudo-inverse (V is singular by translation invariance).
        v = -w.copy()
        np.fill_diagonal(v, 0.0)
        np.fill_diagonal(v, -v.sum(axis=1))
        self._v_pinv = np.linalg.pinv(v)

    def _positions_array(self) -> np.ndarray:
        """Current node positions as an (n, 2) array."""
        return cast(np.ndarray, np.array([[node.x, node.y] for node in self._nodes], dtype=float))

    def _compute_stress(self, positions: np.ndarray) -> float:
        """Weighted stress sum_{i<j} w_ij (dist_ij - d_ij)^2 for a layout."""
        assert self._d_matrix is not None
        assert self._w_matrix is not None
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        s = self._w_matrix * (dist - self._d_matrix) ** 2
        # Each unordered pair is counted twice above (i,j and j,i).
        return float(s.sum() / 2.0)

    def _guttman_transform(self, positions: np.ndarray) -> np.ndarray:
        """One Guttman transform: minimize the majorizing quadratic at ``positions``.

        Returns the updated (n, 2) positions ``X_new = V^+ B(X) X`` where
        ``B(X)_ij = -w_ij d_ij / ||x_i - x_j||`` off-diagonal (0 when the current
        distance is 0) and ``B_ii = -sum_{j != i} B_ij``.
        """
        assert self._d_matrix is not None
        assert self._w_matrix is not None
        assert self._v_pinv is not None

        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 1e-9, -self._w_matrix * self._d_matrix / dist, 0.0)
        np.fill_diagonal(ratio, 0.0)
        b = ratio.copy()
        np.fill_diagonal(b, -ratio.sum(axis=1))

        return cast(np.ndarray, self._v_pinv @ (b @ positions))

    def run(self, **kwargs: Any) -> "SMACOFLayout":
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

        self._initialize_matrices()
        self._iteration = 0
        self._prev_stress = self._compute_stress(self._positions_array())
        self._alpha = 1.0

        self.trigger({"type": EventType.start, "alpha": self._alpha})

        self.kick()

        if center:
            self._center_graph()

        self.trigger({"type": EventType.end, "alpha": 0.0})

        return self

    def tick(self) -> bool:
        """
        Perform one majorization iteration (a single Guttman transform).

        Returns:
            True if converged or the iteration budget is spent, False otherwise.
        """
        assert self._d_matrix is not None

        if self._iteration >= self._iterations:
            return True

        n = len(self._nodes)
        if n == 0:
            return True

        positions = self._positions_array()
        new_positions = self._guttman_transform(positions)

        # Fixed nodes keep their positions (majorization moves every node, so
        # restore them afterwards).
        for i, node in enumerate(self._nodes):
            if node.fixed:
                new_positions[i, 0] = node.x
                new_positions[i, 1] = node.y

        for i, node in enumerate(self._nodes):
            node.x = float(new_positions[i, 0])
            node.y = float(new_positions[i, 1])

        stress = self._compute_stress(new_positions)
        self._iteration += 1
        self._alpha = stress

        self.trigger({"type": EventType.tick, "alpha": self._alpha, "stress": stress})

        # Relative stress decrease; also stop if stress vanished entirely.
        converged = False
        if stress < 1e-12:
            converged = True
        elif self._prev_stress - stress < self._epsilon * stress:
            converged = True
        self._prev_stress = stress

        return converged


__all__ = ["SMACOFLayout"]

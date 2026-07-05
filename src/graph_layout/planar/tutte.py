"""Tutte barycentric straight-line layout.

Draws a 3-connected planar graph with straight-line edges and no crossings by
nailing the vertices of one face to a convex polygon and placing every other
vertex at the centroid (barycentre) of its neighbours. Tutte's spring theorem
guarantees the result is a planar drawing in which every interior face is convex.

The equilibrium positions are the solution of a linear system: for each free
vertex ``i``, ``deg(i) * p_i - sum_{j ~ i} p_j = 0`` with the fixed boundary
vertices moved to the right-hand side. The system matrix is the (uniformly
weighted) Dirichlet Laplacian of the free vertices, which is symmetric positive
definite for a connected graph, so the solution is unique.

Reference:
    Tutte, W. T. (1963). "How to draw a graph." Proc. London Math. Soc. 13,
    743-767.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..base import StaticLayout
from ..types import Event, GroupLike, LinkLike, NodeLike, SizeType
from ._shared import Edge, build_embedding


def tutte_coordinates(
    num_nodes: int, edges: Sequence[Edge]
) -> Optional[dict[int, tuple[float, float]]]:
    """Compute Tutte barycentric coordinates for a planar graph.

    Fixes the largest face on a regular convex polygon and solves for the
    barycentric equilibrium of the remaining vertices. Returns a mapping
    ``vertex -> (x, y)`` in ``[-1, 1]^2``, or ``None`` when the graph is not a
    connected planar simple graph of at least three vertices, or the system is
    numerically singular.

    The drawing is guaranteed crossing-free (with convex faces) only when the
    graph is 3-connected; for graphs that are merely biconnected the barycentric
    map can place distinct vertices at the same point.
    """
    emb = build_embedding(num_nodes, edges)
    if emb is None:
        return None
    rotation, faces = emb

    # Fix the largest face as the outer boundary polygon. Its cyclic order comes
    # from the embedding, so the convex polygon respects the rotation system.
    outer = max(faces, key=len)
    if len(outer) < 3:
        return None
    outer_set = set(outer)

    adj: list[list[int]] = [list(rotation[v]) for v in range(num_nodes)]

    pos = np.zeros((num_nodes, 2), dtype=float)
    m = len(outer)
    for i, v in enumerate(outer):
        angle = 2.0 * math.pi * i / m
        pos[v] = (math.cos(angle), math.sin(angle))

    free = [v for v in range(num_nodes) if v not in outer_set]
    if not free:
        # Every vertex is on the outer face (e.g. a single cycle): the polygon
        # placement is already the whole drawing.
        return {v: (float(pos[v][0]), float(pos[v][1])) for v in range(num_nodes)}

    index = {v: i for i, v in enumerate(free)}
    k = len(free)
    A = np.zeros((k, k), dtype=float)
    bx = np.zeros(k, dtype=float)
    by = np.zeros(k, dtype=float)

    for v in free:
        i = index[v]
        deg = len(adj[v])
        if deg == 0:
            return None
        A[i, i] = float(deg)
        for w in adj[v]:
            if w in index:
                A[i, index[w]] -= 1.0
            else:
                bx[i] += pos[w][0]
                by[i] += pos[w][1]

    try:
        sol_x = np.linalg.solve(A, bx)
        sol_y = np.linalg.solve(A, by)
    except np.linalg.LinAlgError:
        return None

    for v in free:
        i = index[v]
        pos[v] = (sol_x[i], sol_y[i])

    return {v: (float(pos[v][0]), float(pos[v][1])) for v in range(num_nodes)}


class TutteLayout(StaticLayout):
    """Barycentric (Tutte spring) straight-line layout for planar graphs.

    Nails one face of the graph to a convex polygon and relaxes every other
    vertex to the average of its neighbours, giving a crossing-free straight-line
    drawing with convex faces for 3-connected planar graphs.

    Non-planar, disconnected, or trivially small graphs fall back to a
    deterministic circular placement; ``used_tutte`` reports which path ran.

    Example:
        layout = TutteLayout(
            nodes=[{}, {}, {}, {}, {}, {}],
            links=[  # a triangular prism (3-connected planar)
                {'source': 0, 'target': 1}, {'source': 1, 'target': 2},
                {'source': 2, 'target': 0}, {'source': 3, 'target': 4},
                {'source': 4, 'target': 5}, {'source': 5, 'target': 3},
                {'source': 0, 'target': 3}, {'source': 1, 'target': 4},
                {'source': 2, 'target': 5},
            ],
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
        padding: float = 40.0,
    ) -> None:
        super().__init__(
            nodes=nodes,
            links=links,
            groups=groups,
            size=size,
            random_seed=random_seed,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
        )
        self._padding: float = float(padding)
        self._used_tutte: bool = False

    @property
    def padding(self) -> float:
        """Border, in canvas units, kept clear around the drawing."""
        return self._padding

    @padding.setter
    def padding(self, value: float) -> None:
        self._padding = float(value)

    @property
    def used_tutte(self) -> bool:
        """True when the last run drew via Tutte's method (not the fallback)."""
        return self._used_tutte

    def _compute(self, **kwargs: Any) -> None:
        self._used_tutte = False
        n = len(self._nodes)
        if n == 0:
            return

        edges: list[Edge] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))

        coords = tutte_coordinates(n, edges)
        if coords is None:
            self._fallback_layout()
            return

        self._place_from_unit(coords)
        self._used_tutte = True

    def _place_from_unit(self, coords: dict[int, tuple[float, float]]) -> None:
        """Scale coordinates in [-1, 1]^2 onto the canvas, preserving aspect."""
        xs = [p[0] for p in coords.values()]
        ys = [p[1] for p in coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1e-9, max_x - min_x)
        span_y = max(1e-9, max_y - min_y)

        pad = self._padding
        avail_w = max(1.0, self._canvas_size[0] - 2 * pad)
        avail_h = max(1.0, self._canvas_size[1] - 2 * pad)
        scale = min(avail_w / span_x, avail_h / span_y)

        draw_w = span_x * scale
        draw_h = span_y * scale
        off_x = (self._canvas_size[0] - draw_w) / 2.0
        off_y = (self._canvas_size[1] - draw_h) / 2.0

        for i in range(len(self._nodes)):
            gx, gy = coords[i]
            self._nodes[i].x = off_x + (gx - min_x) * scale
            self._nodes[i].y = off_y + (gy - min_y) * scale

    def _fallback_layout(self) -> None:
        n = len(self._nodes)
        cx, cy = self._canvas_size[0] / 2.0, self._canvas_size[1] / 2.0
        radius = max(min(cx, cy) - self._padding, 1.0)
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            self._nodes[i].x = cx + radius * math.cos(angle)
            self._nodes[i].y = cy + radius * math.sin(angle)


__all__ = ["TutteLayout", "tutte_coordinates"]

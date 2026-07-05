"""Planarization layout for non-planar graphs.

Draws an arbitrary graph -- planar or not -- with straight-line segments by
first *planarizing* it: a maximal planar subgraph is embedded and the remaining
edges are reinserted along minimum-crossing paths, each crossing becoming a
degree-four dummy vertex (topological planarization, Batini et al. 1986). The
resulting planar graph is drawn with a straight-line planar method (FPP by
default), which places every real *and* dummy vertex on a grid point. Each
original edge is then rendered as the polyline through the dummy vertices it was
routed across, so every crossing appears as a clean, explicit point and no two
edges cross anywhere else.

For a genuinely planar graph the planarization adds no dummy vertices and the
result is an ordinary straight-line drawing.

Reference:
    Batini, C., Talamo, M., Tamassia, R. (1986). "Computer aided layout of
    entity-relationship diagrams." J. Systems and Software 4, 163-173.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence

from ..base import StaticLayout
from ..orthogonal.planarization import planarize_graph
from ..types import Event, GroupLike, LinkLike, NodeLike, SizeType
from ._shared import Edge
from .fpp import fpp_coordinates
from .schnyder import schnyder_coordinates

_METHODS = {
    "fpp": fpp_coordinates,
    "schnyder": schnyder_coordinates,
}


class PlanarizationLayout(StaticLayout):
    """Straight-line layout for non-planar graphs via planarization.

    Replaces edge crossings with dummy vertices, draws the resulting planar
    graph with a straight-line planar method, and routes each original edge as a
    polyline through its crossing points. After :meth:`run`:

    - every original node has ``x`` / ``y`` set on the canvas;
    - :attr:`crossings` lists the crossing points as ``(x, y)`` tuples;
    - :attr:`edge_routes` maps each original link index to the ordered list of
      canvas points (endpoints plus crossings) its polyline visits;
    - :attr:`crossing_count` is the number of crossings, and
      :attr:`used_planarization` reports whether the method ran (vs. fallback).

    Disconnected or trivially small graphs fall back to a deterministic circular
    placement.

    Example:
        layout = PlanarizationLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[  # K5 -- non-planar
                {'source': i, 'target': j}
                for i in range(5) for j in range(i + 1, 5)
            ],
            size=(800, 600),
        )
        layout.run()
        print(layout.crossing_count)
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
        method: str = "fpp",
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
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
        self._method: str = method
        self._used_planarization: bool = False
        self._crossings: list[tuple[float, float]] = []
        self._edge_routes: dict[int, list[tuple[float, float]]] = {}

    @property
    def padding(self) -> float:
        """Border, in canvas units, kept clear around the drawing."""
        return self._padding

    @padding.setter
    def padding(self, value: float) -> None:
        self._padding = float(value)

    @property
    def method(self) -> str:
        """Straight-line backend used to draw the planarized graph."""
        return self._method

    @property
    def used_planarization(self) -> bool:
        """True when the last run drew via planarization (not the fallback)."""
        return self._used_planarization

    @property
    def crossings(self) -> list[tuple[float, float]]:
        """Canvas positions of the crossing (dummy) vertices."""
        return self._crossings

    @property
    def crossing_count(self) -> int:
        """Number of crossings introduced by the planarization."""
        return len(self._crossings)

    @property
    def edge_routes(self) -> dict[int, list[tuple[float, float]]]:
        """Link index -> ordered canvas polyline points (endpoints + crossings)."""
        return self._edge_routes

    def _compute(self, **kwargs: Any) -> None:
        self._used_planarization = False
        self._crossings = []
        self._edge_routes = {}
        n = len(self._nodes)
        if n == 0:
            return

        edges: list[Edge] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            edges.append((src, tgt))

        pg = planarize_graph(n, edges)

        coords = _METHODS[self._method](pg.num_total_nodes, pg.edges)
        if coords is None:
            self._fallback_layout()
            return

        canvas = self._scale_to_canvas(coords)

        for i in range(n):
            self._nodes[i].x, self._nodes[i].y = canvas[i]

        # Crossing points are the dummy vertices (indices >= n).
        self._crossings = [canvas[c.index] for c in pg.crossings]

        # Each original edge's polyline: chain its ordered segments into a vertex
        # path, then map to canvas points.
        for orig, seg_idxs in pg.original_to_edges.items():
            src, tgt = edges[orig]
            if src == tgt:
                # Self-loop: no meaningful polyline; skip.
                continue
            verts = _chain_segments(src, tgt, [pg.edges[i] for i in seg_idxs])
            self._edge_routes[orig] = [canvas[v] for v in verts]

        self._used_planarization = True

    def _center_graph(self) -> None:
        """Center as usual, keeping crossings and edge routes aligned.

        The base implementation centres on the *original* node bounding box only;
        applying the same translation to the cached crossing points and polyline
        routes keeps them synchronised with the nodes.
        """
        if not self._nodes:
            return
        before_x, before_y = self._nodes[0].x, self._nodes[0].y
        super()._center_graph()
        dx = self._nodes[0].x - before_x
        dy = self._nodes[0].y - before_y
        if dx == 0.0 and dy == 0.0:
            return
        self._crossings = [(x + dx, y + dy) for x, y in self._crossings]
        self._edge_routes = {
            k: [(x + dx, y + dy) for x, y in pts] for k, pts in self._edge_routes.items()
        }

    def _scale_to_canvas(
        self, coords: Mapping[int, tuple[float, float]]
    ) -> dict[int, tuple[float, float]]:
        """Scale grid coordinates (real + dummy vertices) onto the canvas."""
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

        off_x = (self._canvas_size[0] - span_x * scale) / 2.0
        off_y = (self._canvas_size[1] - span_y * scale) / 2.0

        return {
            v: (off_x + (p[0] - min_x) * scale, off_y + (p[1] - min_y) * scale)
            for v, p in coords.items()
        }

    def _fallback_layout(self) -> None:
        n = len(self._nodes)
        cx, cy = self._canvas_size[0] / 2.0, self._canvas_size[1] / 2.0
        radius = max(min(cx, cy) - self._padding, 1.0)
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            self._nodes[i].x = cx + radius * math.cos(angle)
            self._nodes[i].y = cy + radius * math.sin(angle)


def _chain_segments(src: int, tgt: int, segments: list[Edge]) -> list[int]:
    """Chain ordered segments into the vertex path from ``src`` to ``tgt``."""
    path = [src]
    cur = src
    for a, b in segments:
        nxt = b if a == cur else a
        path.append(nxt)
        cur = nxt
    if path[-1] != tgt:
        # Ordering was already source->target; if the last vertex is not tgt the
        # segment list was degenerate -- fall back to the direct endpoints.
        return [src, tgt]
    return path


__all__ = ["PlanarizationLayout"]

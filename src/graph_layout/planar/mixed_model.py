"""Mixed-model layout via a visibility representation.

Draws a connected planar graph in the *mixed model*: each vertex is a horizontal
bar (box) and each edge a vertical segment attaching at a distinct port on each
bar, so high-degree vertices spread their edges out for good angular resolution
and every edge is drawn with no bends. The drawing is a Tamassia-Tollis
*visibility representation* built from two topological numberings:

- the vertices are numbered by a canonical (st) ordering, giving each a
  ``y``-coordinate (longest path from the source);
- the faces of the st-oriented, embedded graph are numbered by a longest path in
  the dual, giving each edge (and the span of each bar) an ``x``-coordinate.

A light barycentric refinement then places each vertex's point at the mean of
its ports, centering the node marker over its incident edges without moving any
bar or edge (so the drawing stays crossing-free).

Reference:
    Tamassia, R., Tollis, I. G. (1986). "A unified approach to visibility
    representations of planar graphs." Discrete & Computational Geometry 1,
    321-341. (Visibility half of Kant's mixed model, Algorithmica 16, 1996.)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..types import Event, GroupLike, LinkLike, NodeLike, SizeType
from ._shared import (
    Edge,
    build_embedding,
    canonical_order,
    choose_outer_triangle,
    trace_faces,
    triangulate,
)


def _uedge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def visibility_representation(num_nodes: int, edges: Sequence[Edge]) -> Optional[dict[str, Any]]:
    """Compute a visibility representation of a connected planar graph.

    Returns a dict with integer-grid data, or ``None`` when the graph is not a
    connected planar simple graph of at least three vertices:

    - ``"bars"``: ``vertex -> (x_left, x_right, y)`` horizontal bar;
    - ``"routes"``: ``original-edge-index -> (x, y_lo, y_hi)`` vertical segment;
    - ``"node_x"``: ``vertex -> float`` barycentre of the vertex's ports;
    - ``"width"`` / ``"height"``: grid extents.
    """
    emb = build_embedding(num_nodes, edges)
    if emb is None:
        return None
    rotation, orig_faces = emb

    trot, _added = triangulate(rotation)
    outer = choose_outer_triangle(trot, orig_faces)
    order = canonical_order(trot, outer)
    st = {v: i for i, v in enumerate(order)}
    s, t = order[0], order[-1]

    adj = {v: list(nbrs) for v, nbrs in trot.items()}

    # y(v): longest path from the source s in the st-oriented DAG (edges point
    # from lower to higher st-number).
    y = {v: 0 for v in range(num_nodes)}
    for v in order:
        for w in adj[v]:
            if st[w] < st[v]:
                y[v] = max(y[v], y[w] + 1)

    # Face of each directed dart. left(u->v) is the face on dart (v, u).
    faces = trace_faces(trot)
    dart_face: dict[Edge, int] = {}
    for fi, face in enumerate(faces):
        k = len(face)
        for i in range(k):
            dart_face[(face[i], face[(i + 1) % k])] = fi

    def left_face(u: int, v: int) -> int:
        return dart_face[(v, u)]

    def right_face(u: int, v: int) -> int:
        return dart_face[(u, v)]

    # Dual DAG: an edge left(e) -> right(e) for every primal edge except (s, t).
    num_faces = len(faces)
    dual: list[list[int]] = [[] for _ in range(num_faces)]
    indeg = [0] * num_faces
    for v in order:
        for w in adj[v]:
            if st[w] <= st[v]:
                continue
            if {v, w} == {s, t}:
                continue
            lf, rf = left_face(v, w), right_face(v, w)
            dual[lf].append(rf)
            indeg[rf] += 1

    # x(f): longest path in the dual (topological, Kahn).
    xf = [0] * num_faces
    remaining = list(indeg)
    queue = deque(f for f in range(num_faces) if remaining[f] == 0)
    processed = 0
    while queue:
        f = queue.popleft()
        processed += 1
        for g in dual[f]:
            if xf[f] + 1 > xf[g]:
                xf[g] = xf[f] + 1
            remaining[g] -= 1
            if remaining[g] == 0:
                queue.append(g)
    if processed != num_faces:
        # Dual is not acyclic -- inconsistent embedding; fail safely.
        return None

    original = {_uedge(u, v) for u, v in edges if u != v}

    # Edge x-coordinate = x of its left face. Only original edges are drawn, and
    # bars are shrunk to span just their real ports.
    routes: dict[int, tuple[int, int, int]] = {}
    ports: dict[int, list[int]] = {v: [] for v in range(num_nodes)}
    edge_index = {_uedge(u, v): i for i, (u, v) in enumerate(edges) if u != v}
    for v in order:
        for w in adj[v]:
            if st[w] <= st[v]:
                continue
            key = _uedge(v, w)
            if key not in original:
                continue
            xe = xf[left_face(v, w)]
            ports[v].append(xe)
            ports[w].append(xe)
            routes[edge_index[key]] = (xe, y[v], y[w])

    bars: dict[int, tuple[int, int, int]] = {}
    node_x: dict[int, float] = {}
    for v in range(num_nodes):
        pv = ports[v]
        if not pv:
            return None  # disconnected vertex has no place
        bars[v] = (min(pv), max(pv), y[v])
        node_x[v] = sum(pv) / len(pv)

    width = max(xf) if xf else 0
    height = max(y.values())
    return {
        "bars": bars,
        "routes": routes,
        "node_x": node_x,
        "width": width,
        "height": height,
    }


class MixedModelLayout(StaticLayout):
    """Mixed-model (visibility-representation) layout for planar graphs.

    Draws each vertex as a horizontal bar and each edge as a bendless vertical
    segment attaching at a distinct port, giving high angular resolution even
    for high-degree vertices. After :meth:`run`:

    - each node's ``x`` / ``y`` is set on the canvas (the node point sits at the
      barycentre of its ports);
    - :attr:`vertex_bars` maps each node to its ``(x_left, x_right, y)`` bar in
      canvas coordinates;
    - :attr:`edge_routes` maps each link index to its two-point vertical polyline
      in canvas coordinates;
    - :attr:`used_mixed_model` reports whether the method ran (vs. fallback).

    Non-planar, disconnected, or trivially small graphs fall back to a
    deterministic circular placement.

    Example:
        layout = MixedModelLayout(
            nodes=[{}, {}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1}, {'source': 0, 'target': 2},
                {'source': 0, 'target': 3}, {'source': 0, 'target': 4},
                {'source': 1, 'target': 2}, {'source': 2, 'target': 3},
                {'source': 3, 'target': 4},
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
        self._used_mixed_model: bool = False
        self._vertex_bars: dict[int, tuple[float, float, float]] = {}
        self._edge_routes: dict[int, list[tuple[float, float]]] = {}

    @property
    def padding(self) -> float:
        """Border, in canvas units, kept clear around the drawing."""
        return self._padding

    @padding.setter
    def padding(self, value: float) -> None:
        self._padding = float(value)

    @property
    def used_mixed_model(self) -> bool:
        """True when the last run drew via the mixed model (not the fallback)."""
        return self._used_mixed_model

    @property
    def vertex_bars(self) -> dict[int, tuple[float, float, float]]:
        """Node -> (x_left, x_right, y) horizontal bar, in canvas coordinates."""
        return self._vertex_bars

    @property
    def edge_routes(self) -> dict[int, list[tuple[float, float]]]:
        """Link index -> two-point vertical polyline, in canvas coordinates."""
        return self._edge_routes

    def _compute(self, **kwargs: Any) -> None:
        self._used_mixed_model = False
        self._vertex_bars = {}
        self._edge_routes = {}
        n = len(self._nodes)
        if n == 0:
            return

        edges: list[Edge] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            edges.append((src, tgt))

        vis = visibility_representation(n, edges)
        if vis is None:
            self._fallback_layout()
            return

        height: int = vis["height"]
        sx, sy, ox, oy = self._fit(vis["width"], height)

        def to_canvas_x(gx: float) -> float:
            return ox + gx * sx

        def to_canvas_y(gy: float) -> float:
            # Flip so the source (y=0) sits at the bottom of the canvas.
            return oy + (height - gy) * sy

        for v in range(n):
            self._nodes[v].x = to_canvas_x(vis["node_x"][v])
            self._nodes[v].y = to_canvas_y(vis["bars"][v][2])

        for v, (xl, xr, yb) in vis["bars"].items():
            self._vertex_bars[v] = (to_canvas_x(xl), to_canvas_x(xr), to_canvas_y(yb))

        for orig, (xe, ylo, yhi) in vis["routes"].items():
            self._edge_routes[orig] = [
                (to_canvas_x(xe), to_canvas_y(ylo)),
                (to_canvas_x(xe), to_canvas_y(yhi)),
            ]

        self._used_mixed_model = True

    def _fit(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Return (scale_x, scale_y, offset_x, offset_y) fitting the grid."""
        span_x = max(1, width)
        span_y = max(1, height)
        pad = self._padding
        avail_w = max(1.0, self._canvas_size[0] - 2 * pad)
        avail_h = max(1.0, self._canvas_size[1] - 2 * pad)
        scale = min(avail_w / span_x, avail_h / span_y)
        off_x = (self._canvas_size[0] - span_x * scale) / 2.0
        off_y = (self._canvas_size[1] - span_y * scale) / 2.0
        return scale, scale, off_x, off_y

    def _center_graph(self) -> None:
        """Center as usual, keeping bars and edge routes aligned with nodes."""
        if not self._nodes:
            return
        before_x, before_y = self._nodes[0].x, self._nodes[0].y
        super()._center_graph()
        dx = self._nodes[0].x - before_x
        dy = self._nodes[0].y - before_y
        if dx == 0.0 and dy == 0.0:
            return
        self._vertex_bars = {
            v: (xl + dx, xr + dx, yb + dy) for v, (xl, xr, yb) in self._vertex_bars.items()
        }
        self._edge_routes = {
            k: [(x + dx, y + dy) for x, y in pts] for k, pts in self._edge_routes.items()
        }

    def _fallback_layout(self) -> None:
        n = len(self._nodes)
        cx, cy = self._canvas_size[0] / 2.0, self._canvas_size[1] / 2.0
        radius = max(min(cx, cy) - self._padding, 1.0)
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            self._nodes[i].x = cx + radius * math.cos(angle)
            self._nodes[i].y = cy + radius * math.sin(angle)


__all__ = ["MixedModelLayout", "visibility_representation"]

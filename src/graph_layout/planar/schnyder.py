"""Schnyder straight-line grid layout.

Draws a planar graph with straight-line edges on an integer grid of side
``n - 1``, using Schnyder's realizer (a decomposition of a triangulation into
three edge-disjoint trees). For each interior vertex the three trees give three
paths to the three outer corners; these paths split the triangulation into
three regions, and the *number of vertices* in each region gives barycentric
coordinates ``(r1, r2, r3)`` with ``r1 + r2 + r3 = n - 1`` that place the vertex
at ``(r1, r2)``. Schnyder proved the resulting straight-line drawing has no
crossings.

Vertex counting (this module) is more compact than counting faces: it lands on
the ``(n-1) x (n-1)`` grid rather than ``2n-5``. Schnyder's classical optimum is
one unit tighter, the ``(n-2) x (n-2)`` grid, but reaching it requires a
boundary tie-breaking that permits controlled collinearity at the outer edges
(e.g. for K4 the sole interior grid point on the ``2 x 2`` grid lies on an outer
edge); the ``n-1`` placement here is strictly non-degenerate.

Reference:
    Schnyder, W. (1990). "Embedding planar graphs on the grid." Proc. 1st ACM-SIAM
    Symposium on Discrete Algorithms (SODA), 138-148.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..types import Event, GroupLike, LinkLike, NodeLike, SizeType
from ._shared import (
    Edge,
    build_embedding,
    build_realizer,
    canonical_order,
    choose_outer_triangle,
    trace_faces,
    triangulate,
)


def _uedge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _path_edges(v: int, parent: dict[int, int]) -> set[Edge]:
    """Undirected edges on the tree path from ``v`` up to its root."""
    edges: set[Edge] = set()
    cur = v
    while cur in parent:
        nxt = parent[cur]
        edges.add(_uedge(cur, nxt))
        cur = nxt
    return edges


def _path_vertices(v: int, parent: dict[int, int]) -> list[int]:
    """Vertices on the tree path from ``v`` up to and including its root."""
    seq = [v]
    cur = v
    while cur in parent:
        cur = parent[cur]
        seq.append(cur)
    return seq


def schnyder_coordinates(
    num_nodes: int, edges: Sequence[Edge]
) -> Optional[dict[int, tuple[int, int]]]:
    """Compute integer Schnyder grid coordinates for a planar graph.

    Returns a mapping ``vertex -> (x, y)`` on an integer grid, or ``None`` when
    the graph is not a connected planar simple graph of at least three vertices
    (the domain of a single-triangle straight-line drawing).
    """
    emb = build_embedding(num_nodes, edges)
    if emb is None:
        return None
    rotation, orig_faces = emb

    trot, _added = triangulate(rotation)
    outer = choose_outer_triangle(trot, orig_faces)
    v1, v2, vn = outer
    order = canonical_order(trot, outer)
    parent_1, parent_2, parent_3 = build_realizer(trot, order, outer)

    # Triangular faces, excluding the outer face (v1 -> v2 -> vn).
    all_faces = trace_faces(trot)
    interior_faces: list[tuple[int, int, int]] = []
    outer_seen = False
    for f in all_faces:
        if not outer_seen and len(f) == 3 and f[0] == v1 and f[1] == v2 and f[2] == vn:
            outer_seen = True
            continue
        interior_faces.append((f[0], f[1], f[2]))

    # Edge -> incident interior face indices (each interior edge borders two).
    edge_faces: dict[Edge, list[int]] = {}
    incident_faces: list[set[int]] = [set() for _ in range(num_nodes)]
    for fi, (a, b, c) in enumerate(interior_faces):
        for x, y in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(_uedge(x, y), []).append(fi)
        incident_faces[a].add(fi)
        incident_faces[b].add(fi)
        incident_faces[c].add(fi)

    def seed_face(oa: int, ob: int) -> Optional[int]:
        """Interior face bordering outer edge (oa, ob)."""
        faces_on = edge_faces.get(_uedge(oa, ob), [])
        return faces_on[0] if faces_on else None

    # Region i is seeded from the interior face on the outer edge opposite root i.
    seed = {1: seed_face(v2, vn), 2: seed_face(v1, vn), 3: seed_face(v1, v2)}

    def flood(start: Optional[int], blocked: set[Edge]) -> set[int]:
        """Faces reachable from ``start`` without crossing a ``blocked`` edge."""
        if start is None:
            return set()
        reached: set[int] = {start}
        queue: deque[int] = deque([start])
        while queue:
            fi = queue.popleft()
            a, b, c = interior_faces[fi]
            for x, y in ((a, b), (b, c), (c, a)):
                e = _uedge(x, y)
                if e in blocked:
                    continue
                for nf in edge_faces.get(e, ()):
                    if nf not in reached:
                        reached.add(nf)
                        queue.append(nf)
        return reached

    coords: dict[int, tuple[int, int]] = {}
    S = num_nodes - 1
    coords[v1] = (S, 0)
    coords[v2] = (0, S)
    coords[vn] = (0, 0)

    parents = {1: parent_1, 2: parent_2, 3: parent_3}
    # Path P_i (to root i) is credited, on its clockwise side, to region R_{i+1}.
    path_region = {1: 2, 2: 3, 3: 1}

    interior = set(range(num_nodes)) - {v1, v2, vn}
    for v in interior:
        # The three monochromatic paths from v to the corners, as vertex sets.
        paths = {i: _path_vertices(v, parents[i]) for i in (1, 2, 3)}
        on_path: dict[int, int] = {}
        blocked: set[Edge] = set()
        for i in (1, 2, 3):
            pv = paths[i]
            for u in pv[1:]:  # every path vertex except v itself (incl. the root)
                on_path[u] = i
            for a, b in zip(pv, pv[1:]):
                blocked.add(_uedge(a, b))

        # The three paths cut the interior faces into three regions.
        face_region: dict[int, int] = {}
        for ri in (1, 2, 3):
            for fi in flood(seed[ri], blocked):
                face_region[fi] = ri

        # Tally vertices: a path vertex counts for its clockwise-adjacent region;
        # every other vertex counts for the region its incident faces all lie in.
        r = {1: 0, 2: 0, 3: 0}
        ok = True
        for u in range(num_nodes):
            if u == v:
                continue
            if u in on_path:
                r[path_region[on_path[u]]] += 1
                continue
            touched = {fr for fi in incident_faces[u] if (fr := face_region.get(fi)) is not None}
            if len(touched) != 1:
                ok = False
                break
            r[next(iter(touched))] += 1

        if not ok or r[1] + r[2] + r[3] != S:
            # Realizer/paths inconsistent for this vertex; fail rather than emit
            # a bad drawing.
            return None
        coords[v] = (r[1], r[2])

    return coords


class SchnyderLayout(StaticLayout):
    """Straight-line planar grid layout via Schnyder's realizer.

    Positions the vertices of a connected planar graph with straight-line edges
    and no crossings on the ``(n-1) x (n-1)`` integer grid (via vertex-count
    barycentric coordinates from the realizer), then scales the drawing onto the
    canvas. The graph is internally triangulated, so the algorithm draws any
    connected planar simple graph with at least three vertices; the added chords
    are used only to compute positions and are not drawn.

    Non-planar, disconnected, or trivially small graphs fall back to a
    deterministic circular placement; ``used_schnyder`` reports which path ran.

    Example:
        layout = SchnyderLayout(
            nodes=[{}, {}, {}, {}],
            links=[
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2},
                {'source': 2, 'target': 3},
                {'source': 3, 'target': 0},
                {'source': 0, 'target': 2},
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
        self._used_schnyder: bool = False

    @property
    def padding(self) -> float:
        """Border, in canvas units, kept clear around the drawing."""
        return self._padding

    @padding.setter
    def padding(self, value: float) -> None:
        self._padding = float(value)

    @property
    def used_schnyder(self) -> bool:
        """True when the last run drew via Schnyder's method (not the fallback)."""
        return self._used_schnyder

    def _compute(self, **kwargs: Any) -> None:
        self._used_schnyder = False
        n = len(self._nodes)
        if n == 0:
            return

        edges: list[Edge] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))

        coords = schnyder_coordinates(n, edges)
        if coords is None:
            self._fallback_layout()
            return

        self._place_from_grid(coords)
        self._used_schnyder = True

    def _place_from_grid(self, coords: dict[int, tuple[int, int]]) -> None:
        """Scale integer grid coordinates onto the canvas, preserving aspect."""
        xs = [p[0] for p in coords.values()]
        ys = [p[1] for p in coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1, max_x - min_x)
        span_y = max(1, max_y - min_y)

        pad = self._padding
        avail_w = max(1.0, self._canvas_size[0] - 2 * pad)
        avail_h = max(1.0, self._canvas_size[1] - 2 * pad)
        # Uniform scale keeps the outer triangle undistorted.
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
        """Deterministic circular placement for out-of-domain graphs."""
        import math

        n = len(self._nodes)
        cx, cy = self._canvas_size[0] / 2.0, self._canvas_size[1] / 2.0
        radius = min(cx, cy) - self._padding
        radius = max(radius, 1.0)
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            self._nodes[i].x = cx + radius * math.cos(angle)
            self._nodes[i].y = cy + radius * math.sin(angle)


__all__ = ["SchnyderLayout", "schnyder_coordinates"]

"""de Fraysseix-Pach-Pollack straight-line grid layout (shift method).

Draws a planar graph with straight-line edges and no crossings on the
``(2n - 4) x (n - 2)`` integer grid. Vertices are installed in canonical order;
each new vertex is placed as the apex of a "tent" (edges of slope +1 and -1)
over a contiguous run of the current outer contour, after shifting the contour
apart to make room. The slope-(+/-1) contour invariant guarantees the apex lands
on an integer grid point and no edge crosses another.

Reference:
    de Fraysseix, H., Pach, J., Pollack, R. (1990). "How to draw a planar graph
    on a grid." Combinatorica 10(1), 41-51.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ..base import StaticLayout
from ..types import Event, GroupLike, LinkLike, NodeLike, SizeType
from ._shared import (
    Edge,
    build_embedding,
    canonical_order,
    choose_outer_triangle,
    triangulate,
)


def fpp_coordinates(num_nodes: int, edges: Sequence[Edge]) -> Optional[dict[int, tuple[int, int]]]:
    """Compute integer FPP grid coordinates for a planar graph.

    Returns a mapping ``vertex -> (x, y)`` on the ``(2n-4) x (n-2)`` grid, or
    ``None`` when the graph is not a connected planar simple graph of at least
    three vertices.
    """
    emb = build_embedding(num_nodes, edges)
    if emb is None:
        return None
    rotation, orig_faces = emb

    trot, _added = triangulate(rotation)
    outer = choose_outer_triangle(trot, orig_faces)
    order = canonical_order(trot, outer)

    n = num_nodes
    adj: list[set[int]] = [set() for _ in range(n)]
    for v, nbrs in trot.items():
        for w in nbrs:
            adj[v].add(w)

    x: dict[int, int] = {}
    y: dict[int, int] = {}
    # ``under[v]`` holds every already-placed vertex whose x-coordinate must move
    # in lockstep with contour vertex ``v`` (its subtree in the shift forest).
    under: dict[int, set[int]] = {}

    v1, v2, v3 = order[0], order[1], order[2]
    x[v1], y[v1] = 0, 0
    x[v2], y[v2] = 2, 0
    x[v3], y[v3] = 1, 1
    under[v1] = {v1}
    under[v2] = {v2}
    under[v3] = {v3}
    contour = [v1, v3, v2]

    for k in range(3, n):
        vk = order[k]
        lower = adj[vk]
        idxs = [i for i, c in enumerate(contour) if c in lower]
        if len(idxs) < 2:
            return None
        left, right = idxs[0], idxs[-1]
        if idxs != list(range(left, right + 1)):
            # Earlier neighbours must be contiguous on the contour; if not, the
            # canonical order is inconsistent for this input.
            return None

        cl = contour[left]
        cr = contour[right]

        # Make room: shift the contour vertices strictly right of cl by 1, and
        # those from cr rightward by 1 more (so the covered span opens up).
        _shift(contour, left + 1, 1, under, x)
        _shift(contour, right, 1, under, x)

        xl, yl = x[cl], y[cl]
        xr, yr = x[cr], y[cr]
        # Apex of the tent: intersection of slope +1 through cl and slope -1
        # through cr. The contour invariant makes this an integer point.
        xk2 = xl + xr + yr - yl
        yk2 = xr - xl + yr + yl
        if xk2 % 2 != 0 or yk2 % 2 != 0:
            return None
        x[vk] = xk2 // 2
        y[vk] = yk2 // 2

        # Covered contour vertices (strictly between cl and cr) join vk's subtree.
        covered = contour[left + 1 : right]
        sub: set[int] = {vk}
        for c in covered:
            sub |= under[c]
        under[vk] = sub
        contour = contour[: left + 1] + [vk] + contour[right:]

    return {v: (x[v], y[v]) for v in range(n)}


def _shift(
    contour: list[int],
    start_idx: int,
    delta: int,
    under: dict[int, set[int]],
    x: dict[int, int],
) -> None:
    """Add ``delta`` to the x-coordinate of every subtree from ``start_idx`` on."""
    if delta == 0:
        return
    for i in range(start_idx, len(contour)):
        for v in under[contour[i]]:
            x[v] += delta


class FPPLayout(StaticLayout):
    """Straight-line planar grid layout via the de Fraysseix-Pach-Pollack shift.

    Positions the vertices of a connected planar graph with straight-line edges
    and no crossings on the ``(2n-4) x (n-2)`` integer grid, then scales the
    drawing onto the canvas. The graph is internally triangulated to run the
    algorithm; the added chords are used only for positioning and are not drawn.

    Non-planar, disconnected, or trivially small graphs fall back to a
    deterministic circular placement; ``used_fpp`` reports which path ran.

    Example:
        layout = FPPLayout(
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
        self._used_fpp: bool = False

    @property
    def padding(self) -> float:
        """Border, in canvas units, kept clear around the drawing."""
        return self._padding

    @padding.setter
    def padding(self, value: float) -> None:
        self._padding = float(value)

    @property
    def used_fpp(self) -> bool:
        """True when the last run drew via the FPP method (not the fallback)."""
        return self._used_fpp

    def _compute(self, **kwargs: Any) -> None:
        self._used_fpp = False
        n = len(self._nodes)
        if n == 0:
            return

        edges: list[Edge] = []
        for link in self._links:
            src = self._get_source_index(link)
            tgt = self._get_target_index(link)
            if 0 <= src < n and 0 <= tgt < n:
                edges.append((src, tgt))

        coords = fpp_coordinates(n, edges)
        if coords is None:
            self._fallback_layout()
            return

        self._place_from_grid(coords)
        self._used_fpp = True

    def _place_from_grid(self, coords: dict[int, tuple[int, int]]) -> None:
        xs = [p[0] for p in coords.values()]
        ys = [p[1] for p in coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1, max_x - min_x)
        span_y = max(1, max_y - min_y)

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
        import math

        n = len(self._nodes)
        cx, cy = self._canvas_size[0] / 2.0, self._canvas_size[1] / 2.0
        radius = max(min(cx, cy) - self._padding, 1.0)
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            self._nodes[i].x = cx + radius * math.cos(angle)
            self._nodes[i].y = cy + radius * math.sin(angle)


__all__ = ["FPPLayout", "fpp_coordinates"]

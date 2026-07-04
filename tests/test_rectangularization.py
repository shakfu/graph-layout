"""Tests for rectangularization (turn-regularization) of orthogonal drawings.

The coordinate assignment's per-edge constraint graphs are provably sufficient
only when every face is a rectangle. ``_rectangularize`` dissects bounded faces
(reflex-corner projection) and encloses the outer face in a dummy rectangle so
that GIOTTOLayout's bend-optimal drawing covers the entire in-scope domain
(biconnected planar graphs of maximum degree 4).
"""

import random

from graph_layout import GIOTTOLayout
from graph_layout.orthogonal.metrics import (
    EAST,
    NORTH,
    SOUTH,
    WEST,
    _refine_to_rectangles,
)
from graph_layout.planarity import check_planarity


def _layout(n, edges, **kwargs):
    layout = GIOTTOLayout(
        nodes=[{} for _ in range(n)],
        links=[{"source": u, "target": v} for u, v in edges],
        size=(600, 600),
        strict=False,
        **kwargs,
    )
    layout.run()
    return layout


class TestRefinement:
    def test_l_face_single_projection(self):
        """An L-shaped face has one reflex corner; refinement projects it onto
        the facing wall (the textbook dissection into two rectangles)."""
        import itertools

        # L-shape walked interior-on-left: A(0,0) B(2,0) C(2,1) D(1,1) E(1,2) F(0,2).
        walk = [
            ("A", "B", EAST),
            ("B", "C", NORTH),
            ("C", "D", WEST),
            ("D", "E", NORTH),
            ("E", "F", WEST),
            ("F", "A", SOUTH),
        ]
        extra = _refine_to_rectangles([walk], itertools.count())
        assert extra is not None
        # One projection: the front dart (F, A, S) is split at W and the dummy
        # dart (D, W) runs in the incoming direction West.
        assert len(extra) == 3
        (a1, w1, d1), (w2, b2, d2), (p3, w3, d3) = extra
        assert (a1, d1) == ("F", SOUTH) and b2 == "A" and d2 == SOUTH
        assert p3 == "D" and d3 == WEST and w1 == w2 == w3

    def test_rectangle_needs_no_refinement(self):
        import itertools

        walk = [
            ("A", "B", EAST),
            ("B", "C", NORTH),
            ("C", "D", WEST),
            ("D", "A", SOUTH),
        ]
        assert _refine_to_rectangles([walk], itertools.count()) == []


class TestBendOptimalCoverage:
    def test_formerly_crossing_cases_now_draw(self):
        """In-domain graphs whose plain coordinate assignment produced crossings
        (the pre-rectangularization ~11% fallback) now draw bend-minimally."""
        cases = [
            # inner-face conflicts
            (10, [(0, 1), (1, 2), (3, 4), (2, 7), (4, 9), (2, 3), (6, 7),
                  (8, 9), (5, 6), (0, 5), (1, 6), (7, 8)]),
            (10, [(0, 1), (3, 8), (1, 2), (3, 4), (4, 9), (2, 3), (6, 7),
                  (8, 9), (5, 6), (0, 5), (1, 6), (7, 8)]),
            # outer-face conflict (boundary arms separated only through the
            # outer region; needs the enclosing rectangle)
            (13, [(0, 1), (9, 10), (3, 4), (5, 8), (6, 9), (7, 10), (9, 12),
                  (11, 12), (0, 2), (3, 6), (2, 5), (1, 3), (4, 7), (8, 11)]),
        ]
        for n, edges in cases:
            assert _layout(n, edges).used_bend_optimal is True, (n, edges)

    def test_random_in_domain_graphs_all_draw_bend_optimal(self):
        """Every random biconnected max-degree-4 planar graph draws from the
        bend-minimal representation (100% in-scope coverage)."""
        rng = random.Random(4242)
        drawn = 0
        attempts = 0
        while drawn < 60 and attempts < 6000:
            attempts += 1
            w, h = rng.randint(2, 5), rng.randint(2, 5)
            keep = rng.uniform(0.6, 1.0)
            edges = set()
            for r in range(h):
                for c in range(w):
                    v = r * w + c
                    if c + 1 < w and rng.random() < keep:
                        edges.add((v, v + 1))
                    if r + 1 < h and rng.random() < keep:
                        edges.add((v, v + w))
            used = sorted({x for e in edges for x in e})
            idx = {v: k for k, v in enumerate(used)}
            n = len(used)
            e = sorted((idx[u], idx[v]) for u, v in edges)
            if not (3 <= n <= 16) or not self._in_domain(n, e):
                continue
            drawn += 1
            assert _layout(n, e).used_bend_optimal is True, (n, e)
        assert drawn == 60  # the generator produced enough in-domain samples

    @staticmethod
    def _in_domain(n, edges):
        deg = [0] * n
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        if max(deg, default=0) > 4 or not check_planarity(n, edges).is_planar:
            return False

        def connected(banned):
            start = next(i for i in range(n) if i != banned)
            seen = {start}
            stack = [start]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y != banned and y not in seen:
                        seen.add(y)
                        stack.append(y)
            return len(seen) == n - (1 if banned is not None else 0)

        if n < 3 or not connected(None):
            return False
        return all(connected(r) for r in range(n))

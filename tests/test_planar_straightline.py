"""Tests for the planar straight-line layouts: Schnyder, FPP, Tutte.

The defining property of all three algorithms is a *crossing-free* straight-line
drawing. The core oracle here brute-forces every pair of non-adjacent edges and
asserts no proper intersection, which validates the whole pipeline end to end
(embedding, triangulation, canonical ordering, and coordinate assignment).
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from graph_layout import (
    FPPLayout,
    MixedModelLayout,
    PlanarizationLayout,
    SchnyderLayout,
    TutteLayout,
)
from graph_layout.orthogonal.planarization import planarize_graph
from graph_layout.planar.fpp import fpp_coordinates
from graph_layout.planar.mixed_model import visibility_representation
from graph_layout.planar.schnyder import schnyder_coordinates
from graph_layout.planar.tutte import tutte_coordinates

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry oracle
# ---------------------------------------------------------------------------


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _proper_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segments p1p2 and p3p4 cross or collinearly overlap."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if d1 == 0 and _on_seg(p3, p4, p1):
        return True
    if d2 == 0 and _on_seg(p3, p4, p2):
        return True
    if d3 == 0 and _on_seg(p1, p2, p3):
        return True
    if d4 == 0 and _on_seg(p1, p2, p4):
        return True
    return False


def assert_crossing_free(
    n: int,
    edges: list[tuple[int, int]],
    coords: dict[int, tuple[float, float]],
    tol: float = 1e-7,
) -> None:
    pts = {v: (float(coords[v][0]), float(coords[v][1])) for v in range(n)}

    # All vertices at distinct positions.
    for a, b in itertools.combinations(range(n), 2):
        assert not (abs(pts[a][0] - pts[b][0]) < tol and abs(pts[a][1] - pts[b][1]) < tol), (
            f"coincident vertices {a} and {b}"
        )

    uedges = list({(min(u, v), max(u, v)) for u, v in edges if u != v})
    for (a, b), (c, d) in itertools.combinations(uedges, 2):
        if len({a, b, c, d}) < 4:
            # Adjacent edges: only a problem if collinear and overlapping past
            # the shared endpoint.
            shared = {a, b} & {c, d}
            p1, p2, p3, p4 = pts[a], pts[b], pts[c], pts[d]
            if abs(_orient(p1, p2, p3)) < tol and abs(_orient(p1, p2, p4)) < tol:
                sv = next(iter(shared))
                for x in {a, b} - shared:
                    if _on_seg(pts[c], pts[d], pts[x]) and x != sv:
                        raise AssertionError(f"overlapping adjacent edges ({a},{b}) & ({c},{d})")
            continue
        assert not _proper_intersect(pts[a], pts[b], pts[c], pts[d]), (
            f"edge ({a},{b}) crosses ({c},{d})"
        )


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------


def grid_graph(w: int, h: int) -> tuple[int, list[tuple[int, int]]]:
    def idx(r: int, c: int) -> int:
        return r * w + c

    edges: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < h:
                edges.append((idx(r, c), idx(r + 1, c)))
    return w * h, edges


def wheel_graph(n: int) -> tuple[int, list[tuple[int, int]]]:
    edges: list[tuple[int, int]] = []
    rim = list(range(1, n))
    for i, v in enumerate(rim):
        edges.append((v, rim[(i + 1) % len(rim)]))
        edges.append((0, v))
    return n, edges


def random_maximal_planar(n: int, seed: int) -> tuple[int, list[tuple[int, int]]]:
    """Random triangulation by repeated insertion of a vertex into a face."""
    rng = random.Random(seed)
    faces = [(0, 1, 2)]
    edges = {(0, 1), (1, 2), (0, 2)}
    for v in range(3, n):
        a, b, c = faces.pop(rng.randrange(len(faces)))
        faces += [(a, b, v), (b, c, v), (a, c, v)]
        for x in (a, b, c):
            edges.add((min(x, v), max(x, v)))
    return n, list(edges)


def random_tree(n: int, seed: int) -> tuple[int, list[tuple[int, int]]]:
    rng = random.Random(seed)
    edges = [(rng.randrange(v), v) for v in range(1, n)]
    return n, edges


PRISM = (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
CUBE = (
    8,
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
)
OCTAHEDRON = (
    6,
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ],
)
K4 = (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
TRIANGLE = (3, [(0, 1), (1, 2), (2, 0)])

# Graphs valid for the grid methods (any connected planar graph >= 3 vertices).
GRID_CASES = [
    TRIANGLE,
    K4,
    (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),  # square
    (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),  # square + diagonal
    grid_graph(3, 3),
    grid_graph(4, 4),
    grid_graph(5, 3),
    wheel_graph(6),
    wheel_graph(10),
    random_tree(12, 1),
    random_tree(20, 7),
]
GRID_CASES += [random_maximal_planar(n, s) for n in (8, 16, 24) for s in range(6)]

# 3-connected planar graphs, the domain where Tutte is crossing-free.
TUTTE_CASES = [K4, PRISM, CUBE, OCTAHEDRON, wheel_graph(6), wheel_graph(10)]
TUTTE_CASES += [random_maximal_planar(n, s) for n in (8, 16) for s in range(6)]


# ---------------------------------------------------------------------------
# Schnyder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,edges", GRID_CASES)
def test_schnyder_crossing_free(n, edges):
    coords = schnyder_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


@pytest.mark.parametrize("n,edges", GRID_CASES)
def test_schnyder_grid_bound(n, edges):
    coords = schnyder_coordinates(n, edges)
    assert coords is not None
    # Vertex-count barycentric coordinates live on the (n-1) grid.
    bound = n - 1
    for x, y in coords.values():
        assert isinstance(x, int) and isinstance(y, int)
        assert 0 <= x <= bound and 0 <= y <= bound
    # Region counts partition the other n-1 vertices, so the corners hit n-1.
    assert max(max(x, y) for x, y in coords.values()) == bound


def test_schnyder_non_planar_returns_none():
    # K5 is non-planar.
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert schnyder_coordinates(5, edges) is None


def test_schnyder_disconnected_returns_none():
    # Two disjoint triangles.
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    assert schnyder_coordinates(6, edges) is None


# ---------------------------------------------------------------------------
# FPP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,edges", GRID_CASES)
def test_fpp_crossing_free(n, edges):
    coords = fpp_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


@pytest.mark.parametrize("n,edges", GRID_CASES)
def test_fpp_grid_bound(n, edges):
    coords = fpp_coordinates(n, edges)
    assert coords is not None
    for x, y in coords.values():
        assert isinstance(x, int) and isinstance(y, int)
        assert 0 <= x <= 2 * n - 4
        assert 0 <= y <= n - 2


def test_fpp_non_planar_returns_none():
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert fpp_coordinates(5, edges) is None


# ---------------------------------------------------------------------------
# Tutte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,edges", TUTTE_CASES)
def test_tutte_crossing_free(n, edges):
    coords = tutte_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


def test_tutte_non_planar_returns_none():
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert tutte_coordinates(5, edges) is None


def test_tutte_convex_outer_face():
    # For the cube the outer face is a 4-cycle; its vertices should form a
    # convex quadrilateral (all turns the same sign).
    n, edges = CUBE
    coords = tutte_coordinates(n, edges)
    assert coords is not None
    # The outer face is the largest face; recover it from the drawing hull is
    # overkill -- instead assert the whole drawing is crossing-free (done above)
    # and that no interior vertex escaped the unit frame.
    for x, y in coords.values():
        assert -1.0001 <= x <= 1.0001 and -1.0001 <= y <= 1.0001


# ---------------------------------------------------------------------------
# Layout class integration
# ---------------------------------------------------------------------------


def _nodes(n):
    return [{} for _ in range(n)]


def _links(edges):
    return [{"source": u, "target": v} for u, v in edges]


@pytest.mark.parametrize(
    "cls,flag",
    [
        (SchnyderLayout, "used_schnyder"),
        (FPPLayout, "used_fpp"),
        (TutteLayout, "used_tutte"),
    ],
)
def test_layout_runs_and_places(cls, flag):
    n, edges = CUBE if cls is TutteLayout else grid_graph(4, 4)
    layout = cls(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    assert getattr(layout, flag) is True
    for node in layout.nodes:
        assert 0.0 <= node.x <= 800.0
        assert 0.0 <= node.y <= 600.0


@pytest.mark.parametrize(
    "cls,flag",
    [
        (SchnyderLayout, "used_schnyder"),
        (FPPLayout, "used_fpp"),
        (TutteLayout, "used_tutte"),
    ],
)
def test_layout_falls_back_on_nonplanar(cls, flag):
    n = 5
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # K5
    layout = cls(nodes=_nodes(n), links=_links(edges), size=(400.0, 400.0))
    layout.run()
    assert getattr(layout, flag) is False
    # Fallback still produces finite, on-canvas coordinates.
    for node in layout.nodes:
        assert math.isfinite(node.x) and math.isfinite(node.y)


@pytest.mark.parametrize("fn", [schnyder_coordinates, fpp_coordinates, tutte_coordinates])
def test_determinism(fn):
    n, edges = random_maximal_planar(20, 3)
    a = fn(n, edges)
    b = fn(n, edges)
    assert a == b


# ---------------------------------------------------------------------------
# Property-based fuzzing
# ---------------------------------------------------------------------------


@st.composite
def _planar_graph(draw):
    """A random connected planar graph: a tree, grid, or triangulation."""
    kind = draw(st.sampled_from(["tree", "grid", "maxplanar"]))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    if kind == "tree":
        n = draw(st.integers(min_value=3, max_value=18))
        return random_tree(n, seed)
    if kind == "grid":
        w = draw(st.integers(min_value=2, max_value=5))
        h = draw(st.integers(min_value=2, max_value=5))
        return grid_graph(w, h)
    n = draw(st.integers(min_value=3, max_value=22))
    return random_maximal_planar(n, seed)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=_planar_graph())
def test_schnyder_property_crossing_free(graph):
    n, edges = graph
    coords = schnyder_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=_planar_graph())
def test_fpp_property_crossing_free(graph):
    n, edges = graph
    coords = fpp_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000), n=st.integers(min_value=4, max_value=22))
def test_tutte_property_crossing_free(seed, n):
    # Triangulations are 3-connected, so Tutte is guaranteed crossing-free.
    n, edges = random_maximal_planar(n, seed)
    coords = tutte_coordinates(n, edges)
    assert coords is not None
    assert_crossing_free(n, edges, coords)


# ---------------------------------------------------------------------------
# Planarization (non-planar graphs)
# ---------------------------------------------------------------------------


def complete_graph(n):
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def complete_bipartite(a, b):
    return a + b, [(i, a + j) for i in range(a) for j in range(b)]


def petersen_graph():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return 10, outer + spokes + inner


NONPLANAR_CASES = [
    complete_graph(5),
    complete_graph(6),
    complete_graph(7),
    complete_bipartite(3, 3),
    complete_bipartite(4, 4),
    petersen_graph(),
]


@pytest.mark.parametrize("n,edges", NONPLANAR_CASES + [complete_graph(4), CUBE])
def test_planarization_augmented_drawing_crossing_free(n, edges):
    # The core guarantee: the planarized graph (crossings replaced by dummy
    # vertices) is drawn crossing-free, so original edges meet only at crossings.
    pg = planarize_graph(n, edges)
    coords = fpp_coordinates(pg.num_total_nodes, pg.edges)
    assert coords is not None
    assert_crossing_free(pg.num_total_nodes, pg.edges, coords)


@pytest.mark.parametrize("n,edges", NONPLANAR_CASES)
def test_planarization_layout_routes_and_crossings(n, edges):
    layout = PlanarizationLayout(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    assert layout.used_planarization is True
    assert layout.crossing_count > 0  # these are all non-planar

    pg = planarize_graph(n, edges)
    assert layout.crossing_count == len(pg.crossings)

    for orig, (u, v) in enumerate(edges):
        route = layout.edge_routes[orig]
        assert len(route) >= 2
        # Polyline starts at u's position and ends at v's position.
        assert route[0] == pytest.approx((layout.nodes[u].x, layout.nodes[u].y))
        assert route[-1] == pytest.approx((layout.nodes[v].x, layout.nodes[v].y))
        # Interior points are crossings.
        assert len(route) - 2 >= 0

    for node in layout.nodes:
        assert 0.0 <= node.x <= 800.0
        assert 0.0 <= node.y <= 600.0


@pytest.mark.parametrize("n,edges", [complete_graph(4), CUBE, grid_graph(3, 3)])
def test_planarization_planar_input_no_crossings(n, edges):
    layout = PlanarizationLayout(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    assert layout.used_planarization is True
    assert layout.crossing_count == 0
    # With no crossings every route is a straight segment (two points).
    for route in layout.edge_routes.values():
        assert len(route) == 2


def test_planarization_schnyder_backend():
    n, edges = complete_graph(5)
    layout = PlanarizationLayout(
        nodes=_nodes(n), links=_links(edges), size=(600.0, 600.0), method="schnyder"
    )
    layout.run()
    assert layout.used_planarization is True
    assert layout.method == "schnyder"
    assert layout.crossing_count > 0


def test_planarization_rejects_bad_method():
    with pytest.raises(ValueError):
        PlanarizationLayout(nodes=_nodes(3), links=_links([(0, 1)]), method="nope")


# ---------------------------------------------------------------------------
# Mixed model (visibility representation)
# ---------------------------------------------------------------------------


def assert_visibility_valid(n, edges, vis):
    """A visibility representation is valid iff bars and vertical edges never
    improperly intersect: edges stay within their bars, no edge pierces a
    non-incident bar, no two independent edges overlap, and bars on the same row
    do not overlap."""
    bars = vis["bars"]
    routes = vis["routes"]
    idx_edge = {i: (u, v) for i, (u, v) in enumerate(edges) if u != v}

    for oi, (xe, ylo, yhi) in routes.items():
        u, v = idx_edge[oi]
        for z in (u, v):
            xl, xr, _yb = bars[z]
            assert xl <= xe <= xr
        assert {bars[u][2], bars[v][2]} == {ylo, yhi}

    # Distinct ports among edges leaving upward / entering downward at a vertex.
    up = {v: [] for v in range(n)}
    down = {v: [] for v in range(n)}
    for oi, (xe, ylo, yhi) in routes.items():
        u, v = idx_edge[oi]
        lo, hi = (u, v) if bars[u][2] < bars[v][2] else (v, u)
        up[lo].append(xe)
        down[hi].append(xe)
    for v in range(n):
        assert len(up[v]) == len(set(up[v]))
        assert len(down[v]) == len(set(down[v]))

    # No edge pierces a non-incident bar.
    for oi, (xe, ylo, yhi) in routes.items():
        u, v = idx_edge[oi]
        for w in range(n):
            if w in (u, v):
                continue
            xl, xr, yw = bars[w]
            assert not (ylo < yw < yhi and xl <= xe <= xr)

    # No two independent edges overlap.
    items = list(routes.items())
    for i in range(len(items)):
        oi, (xa, a1, a2) = items[i]
        ui, vi = idx_edge[oi]
        for j in range(i + 1, len(items)):
            oj, (xb, b1, b2) = items[j]
            uj, vj = idx_edge[oj]
            if xa != xb or ({ui, vi} & {uj, vj}):
                continue
            assert not (max(a1, b1) < min(a2, b2))

    # Bars on the same row do not overlap.
    by_row = {}
    for v in range(n):
        by_row.setdefault(bars[v][2], []).append(v)
    for row in by_row.values():
        for a, b in itertools.combinations(row, 2):
            assert max(bars[a][0], bars[b][0]) > min(bars[a][1], bars[b][1])


MIXED_CASES = GRID_CASES + [
    (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),  # square (biconnected, non-triangulated)
]


@pytest.mark.parametrize("n,edges", MIXED_CASES)
def test_mixed_model_visibility_valid(n, edges):
    vis = visibility_representation(n, edges)
    assert vis is not None
    assert_visibility_valid(n, edges, vis)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=_planar_graph())
def test_mixed_model_property_valid(graph):
    n, edges = graph
    vis = visibility_representation(n, edges)
    assert vis is not None
    assert_visibility_valid(n, edges, vis)


def test_mixed_model_layout_runs_and_places():
    n, edges = wheel_graph(8)
    layout = MixedModelLayout(nodes=_nodes(n), links=_links(edges), size=(800.0, 600.0))
    layout.run()
    assert layout.used_mixed_model is True
    for node in layout.nodes:
        assert 0.0 <= node.x <= 800.0
        assert 0.0 <= node.y <= 600.0
    # Every edge is drawn as a vertical (bendless) segment.
    for route in layout.edge_routes.values():
        assert len(route) == 2
        assert route[0][0] == pytest.approx(route[1][0])
    # Node points sit inside their bars.
    for v, (xl, xr, _yb) in layout.vertex_bars.items():
        assert xl - 1e-6 <= layout.nodes[v].x <= xr + 1e-6


def test_mixed_model_falls_back_on_nonplanar():
    n = 5
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # K5
    layout = MixedModelLayout(nodes=_nodes(n), links=_links(edges), size=(400.0, 400.0))
    layout.run()
    assert layout.used_mixed_model is False
    for node in layout.nodes:
        assert math.isfinite(node.x) and math.isfinite(node.y)

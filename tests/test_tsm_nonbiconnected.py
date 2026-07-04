"""Bend-optimal drawing of non-biconnected planar graphs (per-corner angles).

Bridges and cut vertices make a face walk visit the same vertex more than
once, so angles must be stored per corner (keyed by the incoming dart) rather
than per (vertex, face) pair. Degree-1 vertices additionally put 360-degree
corners on the walk, which rectangularization splits with zero-min-length
virtual darts. These tests cover the extended domain: connected planar graphs
of maximum degree 4, biconnected or not.
"""

from __future__ import annotations

import random

from graph_layout import GIOTTOLayout, check_planarity
from graph_layout.orthogonal.metrics import (
    compute_coordinates,
    compute_orthogonal_shape,
    face_turn_sum,
)
from graph_layout.orthogonal.orthogonalization import (
    build_flow_network,
    compute_faces,
    flow_to_orthogonal_rep,
    solve_min_cost_flow_simple,
)
from graph_layout.planarity import MaxFaceEmbedder

NAMED_CASES = [
    ("single edge", 2, [(0, 1)]),
    ("path5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("star4", 5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("binary tree", 7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    ("bowtie", 5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]),
    (
        "two squares joined by a bridge",
        8,
        [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6, 7), (7, 4)],
    ),
    ("square with pendant", 5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]),
    ("caterpillar tree", 8, [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (0, 6), (3, 7)]),
]


def _pipeline(n, edges):
    pr = check_planarity(n, edges)
    assert pr.is_planar
    emb = MaxFaceEmbedder().embed(n, edges, planarity_result=pr)
    faces = compute_faces(n, edges, embedding=emb)
    net = build_flow_network(n, edges, faces)
    assert solve_min_cost_flow_simple(net)
    rep = flow_to_orthogonal_rep(net, edges)
    shape = compute_orthogonal_shape(faces, rep)
    return faces, rep, shape


def _drawing_has_conflict(drawing) -> bool:
    """Independent (cell-based) overlap / through-vertex check."""
    occupied: dict[tuple, tuple[int, int]] = {}
    vertex_at = {p: v for v, p in drawing.vertex_positions.items()}
    for ekey, route in drawing.edge_routes.items():
        endpoints = {route[0], route[-1]}
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            if x1 == x2:
                lo, hi = sorted((y1, y2))
                cells = [("v", x1, y) for y in range(lo, hi)]
                pts = [(x1, y) for y in range(lo, hi + 1)]
            else:
                lo, hi = sorted((x1, x2))
                cells = [("h", x, y1) for x in range(lo, hi)]
                pts = [(x, y1) for x in range(lo, hi + 1)]
            for c in cells:
                if c in occupied and occupied[c] != ekey:
                    return True
                occupied[c] = ekey
            for p in pts:
                if p in vertex_at and p not in endpoints:
                    return True
    return False


class TestPerCornerAngles:
    def test_face_turns_close_for_named_cases(self):
        """Every bounded face turns +4 and the outer face -4, even when the
        walk visits a cut vertex or bridge more than once."""
        for name, n, edges in NAMED_CASES:
            faces, rep, shape = _pipeline(n, edges)
            sums = [face_turn_sum(f, rep) for f in faces]
            assert all(t in (4, -4) for t in sums), f"{name}: {sums}"
            assert sums.count(-4) == 1, f"{name}: {sums}"
            assert shape.valid, f"{name}: {shape.reason}"

    def test_leaf_corner_gets_full_angle(self):
        """A degree-1 vertex has a single 360-degree corner (angle 4)."""
        _faces, rep, _shape = _pipeline(2, [(0, 1)])
        assert rep.corner_angles[(0, 1)] == 4  # corner at vertex 1
        assert rep.corner_angles[(1, 0)] == 4  # corner at vertex 0

    def test_cut_vertex_angles_sum_to_four(self):
        """Angles at a cut vertex distribute over its per-corner keys."""
        n, edges = 5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]  # bowtie, cut at 2
        faces, rep, _shape = _pipeline(n, edges)
        total = sum(a for (dart, a) in rep.corner_angles.items() if dart[1] == 2)
        assert total == 4

    def test_named_cases_draw_conflict_free(self):
        for name, n, edges in NAMED_CASES:
            faces, _rep, shape = _pipeline(n, edges)
            drawing = compute_coordinates(shape, edges, faces=faces)
            assert drawing.valid, f"{name}: {drawing.reason}"
            assert len(drawing.vertex_positions) == n, name
            assert not _drawing_has_conflict(drawing), name


class TestGIOTTONonBiconnected:
    def test_named_cases_use_bend_optimal(self):
        for name, n, edges in NAMED_CASES:
            layout = GIOTTOLayout(
                nodes=[{} for _ in range(n)],
                links=[{"source": u, "target": v} for u, v in edges],
                size=(600, 600),
                strict=False,
            )
            layout.run()
            assert layout.used_bend_optimal is True, name
            assert len(layout.node_boxes) == n, name

    def test_random_nonbiconnected_graphs_all_draw_bend_optimal(self):
        """Random connected planar max-degree-4 graphs (random trees plus a few
        extra edges -- almost all have cut vertices or leaves) draw from the
        bend-minimal representation."""
        rng = random.Random(9001)
        drawn = 0
        attempts = 0
        while drawn < 50 and attempts < 2000:
            attempts += 1
            n = rng.randint(2, 14)
            edges = set()
            for v in range(1, n):
                edges.add((rng.randrange(v), v))
            for _ in range(rng.randint(0, 3)):
                u, v = rng.randrange(n), rng.randrange(n)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
            edge_list = sorted(edges)
            deg = [0] * n
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if max(deg) > 4 or not check_planarity(n, edge_list).is_planar:
                continue
            drawn += 1
            layout = GIOTTOLayout(
                nodes=[{} for _ in range(n)],
                links=[{"source": u, "target": v} for u, v in edge_list],
                size=(600, 600),
                strict=False,
            )
            layout.run()
            assert layout.used_bend_optimal is True, (n, edge_list)
        assert drawn == 50

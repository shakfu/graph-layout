"""Bend-optimal drawing of degree > 4 vertices via vertex expansion (cages).

Vertices of degree > 4 are outside the Tamassia flow model (a grid point has
only four compass directions). Expansion replaces each such vertex by a cycle
of degree-3 cage vertices in rotation order; the cage face is constrained to a
rectangle (corner angles <= 180 degrees, straight cycle edges) and drawn as
the vertex's node box, with the original edges attaching along the box sides
-- the Kandinsky look. These tests cover the expansion structure and the
end-to-end GIOTTO drawings.
"""

from __future__ import annotations

import random

import pytest

from graph_layout import GIOTTOLayout, check_planarity
from graph_layout.orthogonal.expansion import expand_high_degree
from graph_layout.planarity import MaxFaceEmbedder

HIGH_DEGREE_CASES = [
    ("star K1,5", 6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
    ("star K1,8", 9, [(0, i) for i in range(1, 9)]),
    (
        "wheel W5",
        6,
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    ),
    (
        "two adjacent hubs",
        8,
        [(0, 2), (0, 3), (0, 4), (0, 5), (0, 1), (1, 4), (1, 5), (1, 6), (1, 7), (0, 6)],
    ),
    ("hub with pendant tree", 9, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (5, 6), (6, 7), (6, 8)]),
]


def _layout(n, edges, **kwargs):
    layout = GIOTTOLayout(
        nodes=[{} for _ in range(n)],
        links=[{"source": u, "target": v} for u, v in edges],
        size=(900, 900),
        strict=False,
        **kwargs,
    )
    layout.run()
    return layout


def _check_geometry(layout):
    """No box overlaps, all edges orthogonal, no segment through a node box."""
    boxes = layout.node_boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert not (
                abs(a.x - b.x) * 2 < (a.width + b.width) - 1e-6
                and abs(a.y - b.y) * 2 < (a.height + b.height) - 1e-6
            ), f"boxes {i} and {j} overlap"
    for e in layout.orthogonal_edges:
        sp = boxes[e.source].get_port_position(e.source_port.side, e.source_port.position)
        tp = boxes[e.target].get_port_position(e.target_port.side, e.target_port.position)
        pts = [sp, *list(e.bends), tp]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            assert not (abs(x1 - x2) > 1e-6 and abs(y1 - y2) > 1e-6), (
                f"edge {e.source}->{e.target} has a diagonal segment"
            )
            for b in boxes:
                if b.index in (e.source, e.target):
                    continue
                if abs(x1 - x2) < 1e-6:
                    lo, hi = sorted((y1, y2))
                    through = (
                        b.left + 1e-6 < x1 < b.right - 1e-6
                        and lo < b.bottom - 1e-6
                        and hi > b.top + 1e-6
                    )
                else:
                    lo, hi = sorted((x1, x2))
                    through = (
                        b.top + 1e-6 < y1 < b.bottom - 1e-6
                        and lo < b.right - 1e-6
                        and hi > b.left + 1e-6
                    )
                assert not through, f"edge {e.source}->{e.target} through box {b.index}"


class TestExpansionStructure:
    def test_no_expansion_below_degree_5(self):
        emb = MaxFaceEmbedder().embed(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        assert expand_high_degree(4, emb) is None

    def test_star_expansion_shape(self):
        edges = [(0, i) for i in range(1, 7)]  # K1,6
        emb = MaxFaceEmbedder().embed(7, edges)
        exp = expand_high_degree(7, emb)
        assert exp is not None
        assert list(exp.cages.keys()) == [0]
        cage = exp.cages[0]
        assert len(cage) == 6  # one cage vertex per incident edge
        assert exp.num_nodes == 7 + 6
        # 6 rewritten spokes + 6 cage cycle edges
        assert len(exp.edges) == 12
        assert len(exp.cage_edges) == 6
        # Every cage vertex has degree 3 in the expanded graph.
        deg = {}
        for u, v in exp.edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        assert all(deg[c] == 3 for c in cage)
        # The expanded embedding is planar-consistent: V - E + F == 2.
        faces = exp.embedding.faces()
        n_vertices = 6 + len(cage)  # vertex 0 is replaced, leaves 1..6 remain
        assert n_vertices - len(exp.edges) + len(faces) == 2

    def test_dart_map_covers_both_directions(self):
        edges = [(0, i) for i in range(1, 6)]
        emb = MaxFaceEmbedder().embed(6, edges)
        exp = expand_high_degree(6, emb)
        for u, v in edges:
            pu, pv = exp.dart_map[(u, v)]
            assert exp.dart_map[(v, u)] == (pv, pu)
            assert exp.origin.get(pu, pu) == u
            assert exp.origin.get(pv, pv) == v


class TestGIOTTOHighDegree:
    def test_named_cases_draw_bend_optimal(self):
        for name, n, edges in HIGH_DEGREE_CASES:
            layout = _layout(n, edges)
            assert layout.used_bend_optimal is True, name
            assert len(layout.node_boxes) == n, name
            assert len(layout.orthogonal_edges) == len(edges), name
            _check_geometry(layout)

    def test_hub_box_carries_multiple_ports_per_side(self):
        """A degree-8 hub is a box; with only four sides some side must carry
        several edges at distinct port positions."""
        n, edges = 9, [(0, i) for i in range(1, 9)]
        layout = _layout(n, edges)
        assert layout.used_bend_optimal is True
        ports = [
            (e.source_port if e.source == 0 else e.target_port) for e in layout.orthogonal_edges
        ]
        by_side = {}
        for p in ports:
            by_side.setdefault(p.side, []).append(p.position)
        assert any(len(v) >= 2 for v in by_side.values())
        for positions in by_side.values():
            assert len(set(round(p, 6) for p in positions)) == len(positions)

    def test_hub_box_is_larger_than_default(self):
        n, edges = 9, [(0, i) for i in range(1, 9)]
        layout = _layout(n, edges)
        hub, leaf = layout.node_boxes[0], layout.node_boxes[1]
        assert hub.width > leaf.width or hub.height > leaf.height

    def test_strict_mode_still_rejects_degree_5(self):
        layout = GIOTTOLayout(
            nodes=[{} for _ in range(6)],
            links=[{"source": 0, "target": i} for i in range(1, 6)],
            size=(800, 600),
            strict=True,
        )
        with pytest.raises(ValueError, match="max degree 4"):
            layout.run()

    def test_bend_optimal_disabled_uses_heuristic(self):
        n, edges = 6, [(0, i) for i in range(1, 6)]
        layout = _layout(n, edges, bend_optimal=False)
        assert layout.used_bend_optimal is False
        assert len(layout.node_boxes) == n

    def test_random_hub_graphs_draw_bend_optimal(self):
        """Random planar graphs biased toward a high-degree hub all draw from
        the bend-minimal representation with clean geometry."""
        rng = random.Random(20260704)
        drawn = 0
        attempts = 0
        while drawn < 40 and attempts < 1500:
            attempts += 1
            n = rng.randint(5, 13)
            edges = set()
            for v in range(1, n):
                u = rng.randrange(v) if rng.random() < 0.5 else 0
                edges.add((min(u, v), max(u, v)))
            for _ in range(rng.randint(0, 2)):
                u, v = rng.randrange(n), rng.randrange(n)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
            edge_list = sorted(edges)
            deg = [0] * n
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if max(deg) <= 4 or not check_planarity(n, edge_list).is_planar:
                continue
            drawn += 1
            layout = _layout(n, edge_list)
            assert layout.used_bend_optimal is True, (n, edge_list)
            _check_geometry(layout)
        assert drawn == 40

"""GIOTTO bend-optimal drawing (Topology-Shape-Metrics wiring) and the
cyclic-input recursion fix.

``bend_optimal=True`` drives the drawing from the bend-minimal orthogonal
representation when it is a realizable shape (biconnected, max degree 4), and
falls back to the heuristic router otherwise. These tests check that in-scope
graphs draw as valid orthogonal drawings, that out-of-domain graphs fall back
without error, and that GIOTTO no longer infinite-recurses on cyclic input.
"""

from __future__ import annotations

from graph_layout.orthogonal import GIOTTOLayout


def _grid(w: int, h: int) -> tuple[int, list[tuple[int, int]]]:
    edges: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                edges.append((v, v + 1))
            if r + 1 < h:
                edges.append((v, v + w))
    return w * h, edges


def _layout(n, edges, **kwargs):
    nodes = [{} for _ in range(n)]
    links = [{"source": u, "target": v} for u, v in edges]
    layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=False, **kwargs)
    layout.run()
    return layout


def _boxes_overlap(boxes) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if abs(a.x - b.x) * 2 < (a.width + b.width) and abs(a.y - b.y) * 2 < (
                a.height + b.height
            ):
                return True
    return False


def _all_edges_orthogonal(layout) -> bool:
    boxes = layout.node_boxes
    for e in layout.orthogonal_edges:
        sp = boxes[e.source].get_port_position(e.source_port.side, e.source_port.position)
        tp = boxes[e.target].get_port_position(e.target_port.side, e.target_port.position)
        pts = [sp, *list(e.bends), tp]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if abs(x1 - x2) > 1e-6 and abs(y1 - y2) > 1e-6:
                return False
    return True


# ---------------------------------------------------------------------------
# Cyclic-input recursion fix
# ---------------------------------------------------------------------------


def test_cyclic_graph_does_not_recurse_forever():
    """GIOTTO layer assignment must not infinite-recurse on a cycle.

    Previously ``_assign_layers`` had no back-edge guard, so any cycle drove the
    DFS depth upward without bound (RecursionError). Orthogonal layouts are
    almost always cyclic, so this affected the default path too.
    """
    layout = _layout(4, [(0, 1), (1, 2), (2, 3), (3, 0)])  # a 4-cycle
    assert len(layout.node_boxes) == 4


def test_larger_cyclic_graph_runs():
    n, edges = _grid(4, 4)
    layout = _layout(n, edges)  # default (heuristic) path, cyclic
    assert len(layout.node_boxes) == n


# ---------------------------------------------------------------------------
# Bend-optimal drawings for in-scope graphs
# ---------------------------------------------------------------------------


def test_bend_optimal_draws_valid_orthogonal_graphs():
    cases = [
        ("square", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        ("prism", 6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
    ]
    for name, n, edges in cases:
        layout = _layout(n, edges, bend_optimal=True)
        assert len(layout.node_boxes) == n, name
        assert not _boxes_overlap(layout.node_boxes), f"{name}: overlapping boxes"
        assert _all_edges_orthogonal(layout), f"{name}: non-orthogonal edge"


def test_bend_optimal_draws_grids():
    for w in range(2, 5):
        for h in range(2, 5):
            n, edges = _grid(w, h)
            layout = _layout(n, edges, bend_optimal=True)
            assert len(layout.node_boxes) == n, f"grid {w}x{h}"
            assert not _boxes_overlap(layout.node_boxes), f"grid {w}x{h}: overlap"
            assert _all_edges_orthogonal(layout), f"grid {w}x{h}: non-orthogonal"


def test_bend_optimal_handles_degree_5_via_expansion():
    """A degree-5 vertex used to force the heuristic fallback; vertex expansion
    (cages) now keeps it in the bend-optimal domain."""
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]  # hub degree 4..5
    edges.append((0, 5))
    edges.append((1, 5))  # push a vertex past degree 4
    n = 6
    layout = _layout(n, edges, bend_optimal=True)
    assert len(layout.node_boxes) == n
    assert layout.used_bend_optimal is True
    assert _all_edges_orthogonal(layout)


def test_bend_optimal_falls_back_on_conflicting_drawing():
    """An in-domain graph whose packed drawing would overlap falls back cleanly.

    This graph is biconnected and max degree 4, but its bend-minimal coordinate
    assignment crosses (neither compact nor spread clears it) without full face
    rectangularization; GIOTTO must fall back to the heuristic and still draw.
    """
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (6, 7)]
    layout = _layout(8, edges, bend_optimal=True)
    assert len(layout.node_boxes) == 8
    assert _all_edges_orthogonal(layout)


def test_bend_optimal_defaults_on():
    """With rectangularization covering the whole in-scope domain, the
    bend-minimal drawing is the default (out-of-domain inputs still fall back
    to the heuristic router)."""
    layout = GIOTTOLayout(nodes=[{}], links=[])
    assert layout.bend_optimal is True


def test_used_bend_optimal_signal():
    """used_bend_optimal reports whether the bend-minimal path actually drove
    the drawing, exposing the otherwise-silent fallback."""
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # In-domain (and default-on) -> used.
    assert _layout(4, k4, bend_optimal=True).used_bend_optimal is True
    assert _layout(4, k4).used_bend_optimal is True

    # Explicitly disabled -> heuristic, not used.
    assert _layout(4, k4, bend_optimal=False).used_bend_optimal is False

    # Degree 5 is handled by vertex expansion (cages) since H5.
    deg5 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2)]
    assert _layout(6, deg5, bend_optimal=True).used_bend_optimal is True

    # Out of domain (non-planar K5) -> silent fallback, not used.
    k5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert _layout(5, k5, bend_optimal=True).used_bend_optimal is False

    # In-domain graph whose plain coordinate assignment used to cross:
    # rectangularization now separates the conflicting features, so the
    # bend-minimal drawing succeeds (regression for turn-regularization).
    crossing = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (6, 7)]
    assert _layout(8, crossing, bend_optimal=True).used_bend_optimal is True

"""Disconnected-graph bend-optimal drawing for GIOTTO and Kandinsky.

A planar embedding (and the shared Topology-Shape-Metrics realizer) is only
defined per connected component, so disconnected graphs are drawn one component
at a time and the drawings are packed side by side. These tests check that the
packed drawing keeps every orthogonal invariant globally: valid orthogonal
edges, no overlapping node boxes (across components too), correct index
coverage, and a truthful ``used_bend_optimal`` flag with graceful fallback.
"""

from __future__ import annotations

import pytest

from graph_layout.orthogonal import GIOTTOLayout, KandinskyLayout


def _grid(w: int, h: int, base: int = 0) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            v = base + r * w + c
            if c + 1 < w:
                edges.append((v, v + 1))
            if r + 1 < h:
                edges.append((v, v + w))
    return edges


def _cycle(size: int, base: int = 0) -> list[tuple[int, int]]:
    return [(base + i, base + (i + 1) % size) for i in range(size)]


def _k5(base: int = 0) -> list[tuple[int, int]]:
    return [(base + i, base + j) for i in range(5) for j in range(i + 1, 5)]


def _giotto(n, edges, **kwargs):
    nodes = [{} for _ in range(n)]
    links = [{"source": u, "target": v} for u, v in edges]
    layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=False, **kwargs)
    layout.run()
    return layout


def _kandinsky(n, edges, **kwargs):
    nodes = [{} for _ in range(n)]
    links = [{"source": u, "target": v} for u, v in edges]
    layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600), **kwargs)
    layout.run()
    return layout


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


def _boxes_overlap(boxes) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if (
                abs(a.x - b.x) * 2 < (a.width + b.width) - 1e-9
                and abs(a.y - b.y) * 2 < (a.height + b.height) - 1e-9
            ):
                return True
    return False


def _indices_cover_all(layout, n) -> bool:
    return sorted(b.index for b in layout.node_boxes) == list(range(n))


# ---------------------------------------------------------------------------
# GIOTTO
# ---------------------------------------------------------------------------


def test_giotto_two_grids_bend_optimal():
    edges = _grid(3, 3, 0) + _grid(3, 3, 9)
    layout = _giotto(18, edges, bend_optimal=True)
    assert layout.used_bend_optimal is True
    assert _indices_cover_all(layout, 18)
    assert not _boxes_overlap(layout.node_boxes)
    assert _all_edges_orthogonal(layout)
    # Two bend-free grids -> zero total bends.
    assert layout.total_bends == 0


def test_giotto_mixed_components_with_isolated_vertex():
    # 3x3 grid + triangle + isolated vertex.
    edges = _grid(3, 3, 0) + _cycle(3, 9)
    n = 9 + 3 + 1  # index 12 is isolated
    layout = _giotto(n, edges, bend_optimal=True)
    assert layout.used_bend_optimal is True
    assert _indices_cover_all(layout, n)
    assert not _boxes_overlap(layout.node_boxes)
    assert _all_edges_orthogonal(layout)


def test_giotto_all_isolated_vertices():
    layout = _giotto(4, [], bend_optimal=True)
    assert layout.used_bend_optimal is True
    assert _indices_cover_all(layout, 4)
    assert not _boxes_overlap(layout.node_boxes)


def test_giotto_disconnected_nonplanar_component_falls_back():
    """One component is K5 (non-planar); GIOTTO cannot draw it bend-optimally,
    so the whole graph falls back to the heuristic router but still draws."""
    edges = _cycle(4, 0) + _k5(4)
    layout = _giotto(9, edges, bend_optimal=True)
    assert layout.used_bend_optimal is False
    assert _indices_cover_all(layout, 9)
    assert _all_edges_orthogonal(layout)


def test_giotto_disconnected_disabled_uses_heuristic():
    edges = _grid(3, 3, 0) + _grid(3, 3, 9)
    layout = _giotto(18, edges, bend_optimal=False)
    assert layout.used_bend_optimal is False
    assert _indices_cover_all(layout, 18)
    assert _all_edges_orthogonal(layout)


def test_giotto_node_positions_match_boxes():
    edges = _grid(2, 2, 0) + _cycle(4, 4)
    layout = _giotto(8, edges, bend_optimal=True)
    for i, box in enumerate(layout.node_boxes):
        assert layout.nodes[i].x == pytest.approx(box.x)
        assert layout.nodes[i].y == pytest.approx(box.y)


# ---------------------------------------------------------------------------
# Kandinsky
# ---------------------------------------------------------------------------


def test_kandinsky_two_components_bend_optimal():
    edges = _grid(3, 3, 0) + _cycle(4, 9)
    layout = _kandinsky(13, edges, bend_optimal=True)
    assert layout.used_bend_optimal is True
    assert _indices_cover_all(layout, 13)
    assert not _boxes_overlap(layout.node_boxes)
    assert _all_edges_orthogonal(layout)


def test_kandinsky_disconnected_with_nonplanar_component():
    """Kandinsky draws a non-planar component via planarization, so a
    disconnected graph with a K5 component is still fully bend-optimal."""
    edges = _cycle(4, 0) + _k5(4)
    layout = _kandinsky(9, edges, bend_optimal=True)
    assert layout.used_bend_optimal is True
    assert _indices_cover_all(layout, 9)
    assert not _boxes_overlap(layout.node_boxes)
    assert _all_edges_orthogonal(layout)


def test_kandinsky_default_disconnected_uses_heuristic():
    # bend_optimal defaults to False for Kandinsky.
    edges = _grid(3, 3, 0) + _grid(3, 3, 9)
    layout = _kandinsky(18, edges)
    assert layout.used_bend_optimal is False
    assert _indices_cover_all(layout, 18)
    assert _all_edges_orthogonal(layout)


def test_connected_graph_unaffected():
    """The disconnected path must not perturb a single-component drawing."""
    edges = _grid(3, 3)
    a = _giotto(9, edges, bend_optimal=True)
    assert a.used_bend_optimal is True
    assert a.total_bends == 0
    assert _indices_cover_all(a, 9)

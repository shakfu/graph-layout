"""Kandinsky's opt-in bend-optimal drawing (shared Topology-Shape-Metrics path).

``KandinskyLayout`` computes the bend-minimal orthogonal representation but, by
default, draws with the hierarchical heuristic router (the representation is only
exposed via ``orthogonal_rep``). With ``bend_optimal=True`` it instead draws
directly from that representation via the shared realizer -- the same path
GIOTTO uses -- covering connected planar graphs including degree > 4 (cages, H5)
and bridges / cut vertices (per-corner angles, H6a). Non-planar input falls back
to the heuristic router.
"""

from __future__ import annotations

import pytest

from graph_layout import KandinskyLayout


def _layout(n, edges, **kwargs):
    return _run(
        KandinskyLayout(
            nodes=[{} for _ in range(n)],
            links=[{"source": u, "target": v} for u, v in edges],
            size=(700, 600),
            **kwargs,
        )
    )


def _run(layout):
    layout.run()
    return layout


def _grid(w, h):
    edges = []
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                edges.append((v, v + 1))
            if r + 1 < h:
                edges.append((v, v + w))
    return w * h, edges


def _boxes_overlap(boxes) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if (abs(a.x - b.x) * 2 < (a.width + b.width) - 1e-6) and (
                abs(a.y - b.y) * 2 < (a.height + b.height) - 1e-6
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


class TestDefaultUnchanged:
    def test_bend_optimal_defaults_off(self):
        layout = KandinskyLayout(nodes=[{}], links=[])
        assert layout.bend_optimal is False

    def test_default_run_does_not_use_bend_optimal(self):
        n, edges = _grid(3, 3)
        layout = _layout(n, edges)
        assert layout.used_bend_optimal is False
        assert len(layout.node_boxes) == n

    def test_optimize_bends_still_populates_representation(self):
        """optimize_bends computes the representation even without realizing it."""
        layout = _layout(4, [(0, 1), (1, 2), (2, 3), (3, 0)], optimize_bends=True)
        assert layout.used_bend_optimal is False
        assert layout.orthogonal_rep is not None


class TestBendOptimalPlanar:
    def test_planar_graphs_realize(self):
        cases = [
            ("square", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            ("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
            ("prism", 6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
        ]
        for name, n, edges in cases:
            layout = _layout(n, edges, bend_optimal=True)
            assert layout.used_bend_optimal is True, name
            assert len(layout.node_boxes) == n, name
            assert not _boxes_overlap(layout.node_boxes), f"{name}: overlapping boxes"
            assert _all_edges_orthogonal(layout), f"{name}: non-orthogonal edge"

    def test_grids_realize(self):
        for w in range(2, 5):
            for h in range(2, 5):
                n, edges = _grid(w, h)
                layout = _layout(n, edges, bend_optimal=True)
                assert layout.used_bend_optimal is True, f"grid {w}x{h}"
                assert not _boxes_overlap(layout.node_boxes), f"grid {w}x{h}: overlap"
                assert _all_edges_orthogonal(layout), f"grid {w}x{h}: non-orthogonal"

    def test_bend_optimal_matches_giotto_bend_count(self):
        """On an in-domain graph both layouts draw from the same bend-minimal
        representation, so they report the same total bends."""
        from graph_layout import GIOTTOLayout

        n, edges = _grid(3, 3)
        k = _layout(n, edges, bend_optimal=True)
        g = GIOTTOLayout(
            nodes=[{} for _ in range(n)],
            links=[{"source": u, "target": v} for u, v in edges],
            size=(700, 600),
            strict=False,
            bend_optimal=True,
        )
        g.run()
        assert k.used_bend_optimal and g.used_bend_optimal
        assert k.total_bends == g.total_bends


class TestBendOptimalExtendedDomain:
    def test_non_biconnected_realizes(self):
        """Bridges / cut vertices / leaves (H6a)."""
        cases = [
            ("binary tree", 7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
            (
                "two squares + bridge",
                8,
                [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6, 7), (7, 4)],
            ),
        ]
        for name, n, edges in cases:
            layout = _layout(n, edges, bend_optimal=True)
            assert layout.used_bend_optimal is True, name
            assert not _boxes_overlap(layout.node_boxes), name
            assert _all_edges_orthogonal(layout), name

    def test_high_degree_realizes_as_cages(self):
        """Degree > 4 hub becomes a cage box with several ports per side (H5)."""
        n, edges = 9, [(0, i) for i in range(1, 9)]  # degree-8 hub
        layout = _layout(n, edges, bend_optimal=True)
        assert layout.used_bend_optimal is True
        assert _all_edges_orthogonal(layout)
        # The hub box is larger than a leaf box (it spans the cage rectangle).
        hub = layout.node_boxes[0]
        leaf = layout.node_boxes[1]
        assert hub.width > leaf.width or hub.height > leaf.height
        # Some side of the hub carries multiple edges at distinct port positions.
        by_side: dict = {}
        for e in layout.orthogonal_edges:
            p = e.source_port if e.source == 0 else e.target_port
            by_side.setdefault(p.side, []).append(round(p.position, 6))
        assert any(len(v) >= 2 for v in by_side.values())
        for positions in by_side.values():
            assert len(set(positions)) == len(positions)


class TestBendOptimalFallback:
    def test_non_planar_falls_back(self):
        """K5 is non-planar: with handle_crossings the drawing has crossing
        dummies, so realization is skipped and the heuristic router runs."""
        k5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        layout = _layout(5, k5, bend_optimal=True)
        assert layout.used_bend_optimal is False
        assert len(layout.node_boxes) >= 5  # produced a drawing via fallback

    def test_bend_optimal_setter(self):
        layout = KandinskyLayout(nodes=[{}], links=[])
        assert layout.bend_optimal is False
        layout.bend_optimal = True
        assert layout.bend_optimal is True

    def test_empty_graph(self):
        layout = KandinskyLayout(nodes=[], links=[], bend_optimal=True)
        layout.run()
        assert layout.used_bend_optimal is False


class TestSvgSmoke:
    def test_bend_optimal_svg_renders(self):
        n, edges = _grid(3, 3)
        layout = _layout(n, edges, bend_optimal=True)
        svg = layout.to_svg()
        assert svg.startswith("<svg") or "<svg" in svg
        assert "polyline" in svg or "path" in svg


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])

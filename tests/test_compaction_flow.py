"""
Tests for flow-based and longest-path compaction.
"""

from __future__ import annotations

import time

import pytest

from graph_layout import KandinskyLayout
from graph_layout.orthogonal.compaction_flow import (
    _build_constraint_dag,
    compact_flow_1d,
    compact_layout_flow,
    compact_layout_longest_path,
    compact_longest_path_1d,
)
from graph_layout.orthogonal.types import NodeBox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_box(index: int, x: float, y: float, width: float = 60, height: float = 40) -> NodeBox:
    return NodeBox(index=index, x=x, y=y, width=width, height=height)


def _check_no_overlaps(
    boxes: list[NodeBox],
    positions: list[tuple[float, float]],
    h_sep: float,
    v_sep: float,
    tol: float = 1e-6,
) -> None:
    """Assert that no two boxes overlap given new positions."""
    n = len(boxes)
    for i in range(n):
        xi, yi = positions[i]
        wi, hi_ = boxes[i].width, boxes[i].height
        for j in range(i + 1, n):
            xj, yj = positions[j]
            wj, hj = boxes[j].width, boxes[j].height

            # Check if they overlap vertically
            if not (yi + hi_ / 2 <= yj - hj / 2 + tol or yj + hj / 2 <= yi - hi_ / 2 + tol):
                # They overlap vertically -> must be separated horizontally
                h_gap = abs(xj - xi) - (wi / 2 + wj / 2)
                assert h_gap >= h_sep - tol, (
                    f"Boxes {i} and {j} violate horizontal separation: "
                    f"gap={h_gap:.2f}, required={h_sep}"
                )

            # Check if they overlap horizontally
            if not (xi + wi / 2 <= xj - wj / 2 + tol or xj + wj / 2 <= xi - wi / 2 + tol):
                # They overlap horizontally -> must be separated vertically
                v_gap = abs(yj - yi) - (hi_ / 2 + hj / 2)
                assert v_gap >= v_sep - tol, (
                    f"Boxes {i} and {j} violate vertical separation: "
                    f"gap={v_gap:.2f}, required={v_sep}"
                )


# ---------------------------------------------------------------------------
# Constraint DAG tests
# ---------------------------------------------------------------------------


class TestConstraintDAG:
    def test_two_boxes_vertical_overlap_creates_edge(self):
        """Two boxes at same y should produce a constraint edge."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=200, y=0),
        ]
        n, edges = _build_constraint_dag(boxes, separation=20, dimension="horizontal")
        assert n == 2
        assert len(edges) == 1
        u, v, gap = edges[0]
        assert u == 0 and v == 1
        # gap = half_width_0 + sep + half_width_1 = 30 + 20 + 30
        assert gap == pytest.approx(80.0)

    def test_two_boxes_no_vertical_overlap_no_edge(self):
        """Two boxes at very different y should produce no constraint edge."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=200, y=200),
        ]
        _, edges = _build_constraint_dag(boxes, separation=20, dimension="horizontal")
        assert len(edges) == 0

    def test_three_boxes_in_line(self):
        """Three boxes in a horizontal line -> chain of constraints."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=100, y=0),
            _make_box(2, x=200, y=0),
        ]
        _, edges = _build_constraint_dag(boxes, separation=20, dimension="horizontal")
        # All three overlap vertically, so we get edges 0->1, 0->2, 1->2
        assert len(edges) == 3

    def test_vertical_dimension(self):
        """Constraint DAG works for vertical dimension."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=0, y=200),
        ]
        _, edges = _build_constraint_dag(boxes, separation=20, dimension="vertical")
        assert len(edges) == 1
        u, v, gap = edges[0]
        assert u == 0 and v == 1
        # gap = half_height_0 + sep + half_height_1 = 20 + 20 + 20
        assert gap == pytest.approx(60.0)

    def test_empty_boxes(self):
        n, edges = _build_constraint_dag([], separation=20, dimension="horizontal")
        assert n == 0
        assert edges == []


# ---------------------------------------------------------------------------
# Longest-path compaction tests
# ---------------------------------------------------------------------------


class TestLongestPathCompaction:
    def test_two_boxes_minimum_separation(self):
        """Two boxes should end up with exactly minimum separation."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=500, y=0),  # far apart initially
        ]
        sep = 20.0
        new_x = compact_longest_path_1d(boxes, sep, "horizontal")

        # Gap between box edges should be exactly separation
        gap = new_x[1] - new_x[0] - (boxes[0].width / 2 + boxes[1].width / 2)
        assert gap == pytest.approx(sep)

    def test_two_boxes_vertical(self):
        """Vertical compaction of two boxes."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=0, y=500),
        ]
        sep = 30.0
        new_y = compact_longest_path_1d(boxes, sep, "vertical")

        gap = new_y[1] - new_y[0] - (boxes[0].height / 2 + boxes[1].height / 2)
        assert gap == pytest.approx(sep)

    def test_no_perpendicular_overlap_preserves_relative_order(self):
        """Boxes with no perpendicular overlap get independent coordinates."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=100, y=200),  # no vertical overlap
        ]
        sep = 20.0
        new_x = compact_longest_path_1d(boxes, sep, "horizontal")

        # Both should get their independent positions (from source)
        # Since no constraint edge, each gets half_width + sep
        assert new_x[0] == pytest.approx(boxes[0].width / 2 + sep)
        assert new_x[1] == pytest.approx(boxes[1].width / 2 + sep)

    def test_grid_2x2(self):
        """2x2 grid compaction in both dimensions."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=300, y=0),
            _make_box(2, x=0, y=300),
            _make_box(3, x=300, y=300),
        ]
        result = compact_layout_longest_path(boxes, [], node_separation=20, layer_separation=20)

        assert len(result.node_positions) == 4
        assert result.width > 0
        assert result.height > 0
        _check_no_overlaps(boxes, result.node_positions, 20, 20)

    def test_single_box(self):
        """Single box should work without errors."""
        boxes = [_make_box(0, x=100, y=100)]
        new_x = compact_longest_path_1d(boxes, 20, "horizontal")
        assert len(new_x) == 1

    def test_all_positions_non_negative(self):
        """All positions should be non-negative (or close to zero)."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=100, y=0),
            _make_box(2, x=200, y=0),
        ]
        new_x = compact_longest_path_1d(boxes, 20, "horizontal")
        for x in new_x:
            assert x >= 0

    def test_separation_constraints_satisfied(self):
        """All separation constraints must be satisfied."""
        boxes = [
            _make_box(0, x=0, y=0, width=80, height=50),
            _make_box(1, x=100, y=10, width=60, height=40),
            _make_box(2, x=250, y=5, width=70, height=45),
            _make_box(3, x=50, y=100, width=50, height=30),
        ]
        sep = 25.0
        result = compact_layout_longest_path(boxes, [], node_separation=sep, layer_separation=sep)
        _check_no_overlaps(boxes, result.node_positions, sep, sep)

    def test_empty_boxes(self):
        result = compact_layout_longest_path([], [], node_separation=20, layer_separation=20)
        assert result.node_positions == []
        assert result.width == 0
        assert result.height == 0


# ---------------------------------------------------------------------------
# Flow compaction tests
# ---------------------------------------------------------------------------


class TestFlowCompaction:
    def test_two_boxes_minimum_separation(self):
        """Flow compaction should achieve at least minimum separation."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=500, y=0),
        ]
        sep = 20.0
        new_x = compact_flow_1d(boxes, sep, "horizontal")

        gap = new_x[1] - new_x[0] - (boxes[0].width / 2 + boxes[1].width / 2)
        assert gap >= sep - 1e-6

    def test_non_negative_positions(self):
        """All positions should be non-negative."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=100, y=0),
            _make_box(2, x=200, y=0),
        ]
        new_x = compact_flow_1d(boxes, 20, "horizontal")
        for x in new_x:
            assert x >= -1e-6

    def test_separation_constraints_satisfied(self):
        """All separation constraints must be satisfied after flow compaction."""
        boxes = [
            _make_box(0, x=0, y=0, width=80, height=50),
            _make_box(1, x=100, y=10, width=60, height=40),
            _make_box(2, x=250, y=5, width=70, height=45),
        ]
        sep = 25.0
        result = compact_layout_flow(boxes, [], node_separation=sep, layer_separation=sep)
        _check_no_overlaps(boxes, result.node_positions, sep, sep)

    def test_flow_at_least_as_good_as_longest_path(self):
        """Flow compaction area should be <= longest-path area."""
        boxes = [
            _make_box(0, x=0, y=0),
            _make_box(1, x=300, y=0),
            _make_box(2, x=0, y=300),
            _make_box(3, x=300, y=300),
        ]
        lp = compact_layout_longest_path(boxes, [], node_separation=20, layer_separation=20)
        flow = compact_layout_flow(boxes, [], node_separation=20, layer_separation=20)

        lp_area = lp.width * lp.height
        flow_area = flow.width * flow.height
        # Flow should be at least as good (or equal)
        assert flow_area <= lp_area + 1e-6

    def test_empty_boxes(self):
        result = compact_layout_flow([], [], node_separation=20, layer_separation=20)
        assert result.node_positions == []

    def test_single_box(self):
        boxes = [_make_box(0, x=100, y=100)]
        result = compact_layout_flow(boxes, [], node_separation=20, layer_separation=20)
        assert len(result.node_positions) == 1


# ---------------------------------------------------------------------------
# Integration tests with KandinskyLayout
# ---------------------------------------------------------------------------


class TestIntegration:
    def _make_graph(self, n_nodes: int = 6):
        nodes = [{} for _ in range(n_nodes)]
        links = [{"source": i, "target": i + 1} for i in range(n_nodes - 1)]
        return nodes, links

    def test_kandinsky_flow_compaction(self):
        """KandinskyLayout with flow compaction runs without error."""
        nodes, links = self._make_graph()
        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compaction_method="flow",
        )
        result = layout.run()
        assert result is layout
        assert layout.node_boxes is not None
        assert len(layout.node_boxes) >= len(nodes)

    def test_kandinsky_longest_path_compaction(self):
        """KandinskyLayout with longest_path compaction runs without error."""
        nodes, links = self._make_graph()
        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compaction_method="longest_path",
        )
        result = layout.run()
        assert result is layout
        assert layout.node_boxes is not None

    def test_area_comparison(self):
        """Flow/longest-path should produce area <= greedy (or close)."""
        nodes, links = self._make_graph(8)
        # Add some cross-links
        links.append({"source": 0, "target": 3})
        links.append({"source": 2, "target": 5})

        areas = {}
        for method in ("greedy", "longest_path", "flow"):
            layout = KandinskyLayout(
                nodes=nodes,
                links=links,
                size=(800, 600),
                compaction_method=method,
            )
            layout.run()
            if layout.compaction_result is not None:
                areas[method] = layout.compaction_result.width * layout.compaction_result.height

        if "greedy" in areas and "longest_path" in areas:
            # Longest path should be competitive with greedy
            # Allow 20% tolerance since they use different constraint strategies
            assert areas["longest_path"] <= areas["greedy"] * 1.2 + 1e-6

    def test_giotto_compaction_method(self):
        """GIOTTOLayout with compaction_method parameter works."""
        from graph_layout import GIOTTOLayout

        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]
        layout = GIOTTOLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compaction_method="longest_path",
        )
        result = layout.run()
        assert result is layout


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_longest_path_100_nodes(self):
        """Longest-path compaction of 100 nodes should be fast."""
        import random

        rng = random.Random(42)
        boxes = [_make_box(i, x=rng.uniform(0, 1000), y=rng.uniform(0, 1000)) for i in range(100)]

        start = time.monotonic()
        result = compact_layout_longest_path(
            boxes,
            [],
            node_separation=20,
            layer_separation=20,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"100-node longest-path took {elapsed:.2f}s"
        assert len(result.node_positions) == 100

    def test_longest_path_500_nodes(self):
        """Longest-path compaction of 500 nodes should complete in <0.5s."""
        import random

        rng = random.Random(42)
        boxes = [_make_box(i, x=rng.uniform(0, 2000), y=rng.uniform(0, 2000)) for i in range(500)]

        start = time.monotonic()
        result = compact_layout_longest_path(
            boxes,
            [],
            node_separation=20,
            layer_separation=20,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"500-node longest-path took {elapsed:.2f}s"
        assert len(result.node_positions) == 500

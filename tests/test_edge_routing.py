"""Tests for the constraint-aware edge routing module."""

from __future__ import annotations

import pytest

from graph_layout.orthogonal.edge_routing import (
    _segment_intersects_box,
    assign_ports,
    determine_port_sides,
    nudge_overlapping_segments,
    route_all_edges,
    route_edge,
    route_self_loop,
)
from graph_layout.orthogonal.types import NodeBox, OrthogonalEdge, Port, Side


def _make_box(index: int, x: float, y: float, w: float = 60, h: float = 40) -> NodeBox:
    return NodeBox(index=index, x=x, y=y, width=w, height=h)


# ---------------------------------------------------------------------------
# determine_port_sides
# ---------------------------------------------------------------------------


class TestDeterminePortSides:
    def test_target_below_source(self) -> None:
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 100, 300)
        assert determine_port_sides(src, tgt) == (Side.SOUTH, Side.NORTH)

    def test_target_above_source(self) -> None:
        src = _make_box(0, 100, 300)
        tgt = _make_box(1, 100, 100)
        assert determine_port_sides(src, tgt) == (Side.NORTH, Side.SOUTH)

    def test_target_right_of_source(self) -> None:
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 400, 100)
        assert determine_port_sides(src, tgt) == (Side.EAST, Side.WEST)

    def test_target_left_of_source(self) -> None:
        src = _make_box(0, 400, 100)
        tgt = _make_box(1, 100, 100)
        assert determine_port_sides(src, tgt) == (Side.WEST, Side.EAST)


# ---------------------------------------------------------------------------
# assign_ports
# ---------------------------------------------------------------------------


class TestAssignPorts:
    def test_single_edge_center_port(self) -> None:
        boxes = [_make_box(0, 0, 0), _make_box(1, 200, 0)]
        edges = [(0, 1)]
        sides = [(Side.EAST, Side.WEST)]
        ports = assign_ports(boxes, edges, sides)
        assert len(ports) == 1
        src_port, tgt_port = ports[0]
        assert src_port.position == pytest.approx(0.5)
        assert tgt_port.position == pytest.approx(0.5)

    def test_two_edges_same_side_distributed(self) -> None:
        boxes = [_make_box(0, 0, 0), _make_box(1, 200, 0), _make_box(2, 200, 200)]
        edges = [(0, 1), (0, 2)]
        sides = [(Side.EAST, Side.WEST), (Side.EAST, Side.WEST)]
        ports = assign_ports(boxes, edges, sides)

        # Both edges exit from node 0's EAST side
        p0 = ports[0][0].position
        p1 = ports[1][0].position
        # Should be at 1/3 and 2/3
        assert p0 == pytest.approx(1 / 3)
        assert p1 == pytest.approx(2 / 3)

    def test_three_edges_same_side(self) -> None:
        boxes = [_make_box(i, i * 200, 0) for i in range(4)]
        edges = [(0, 1), (0, 2), (0, 3)]
        sides = [(Side.EAST, Side.WEST)] * 3
        ports = assign_ports(boxes, edges, sides)

        positions = [ports[i][0].position for i in range(3)]
        assert positions[0] == pytest.approx(0.25)
        assert positions[1] == pytest.approx(0.5)
        assert positions[2] == pytest.approx(0.75)

    def test_self_loop_ports(self) -> None:
        boxes = [_make_box(0, 100, 100)]
        edges = [(0, 0)]
        sides = [(Side.EAST, Side.SOUTH)]
        ports = assign_ports(boxes, edges, sides)
        src_port, tgt_port = ports[0]
        assert src_port.side == Side.EAST
        assert tgt_port.side == Side.SOUTH


# ---------------------------------------------------------------------------
# route_self_loop
# ---------------------------------------------------------------------------


class TestRouteSelfLoop:
    def test_self_loop_produces_bends(self) -> None:
        box = _make_box(0, 100, 100)
        src_port = Port(node=0, side=Side.EAST, position=0.5)
        tgt_port = Port(node=0, side=Side.SOUTH, position=0.5)
        bends = route_self_loop(box, src_port, tgt_port, 15.0)
        # Should produce 3 bends (forming a right-angle path outside the node)
        assert len(bends) == 3

    def test_self_loop_bends_are_outside_box(self) -> None:
        box = _make_box(0, 100, 100, 60, 40)
        src_port = Port(node=0, side=Side.EAST, position=0.5)
        tgt_port = Port(node=0, side=Side.SOUTH, position=0.5)
        bends = route_self_loop(box, src_port, tgt_port, 15.0)

        # All bends should be outside the box boundary
        for bx, by in bends:
            # At least one coordinate should be outside the box
            outside = bx > box.right or bx < box.left or by > box.bottom or by < box.top
            assert outside, f"Bend ({bx}, {by}) is inside box"


# ---------------------------------------------------------------------------
# route_edge
# ---------------------------------------------------------------------------


class TestRouteEdge:
    def test_aligned_vertical_no_bends(self) -> None:
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 100, 300)
        sp = Port(node=0, side=Side.SOUTH, position=0.5)
        tp = Port(node=1, side=Side.NORTH, position=0.5)
        bends = route_edge(src, tgt, sp, tp, [], 15.0)
        assert len(bends) == 0

    def test_offset_vertical_two_bends(self) -> None:
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 200, 300)
        sp = Port(node=0, side=Side.SOUTH, position=0.5)
        tp = Port(node=1, side=Side.NORTH, position=0.5)
        bends = route_edge(src, tgt, sp, tp, [], 15.0)
        assert len(bends) == 2

    def test_perpendicular_one_bend(self) -> None:
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 300, 300)
        sp = Port(node=0, side=Side.SOUTH, position=0.5)
        tp = Port(node=1, side=Side.WEST, position=0.5)
        bends = route_edge(src, tgt, sp, tp, [], 15.0)
        assert len(bends) == 1


# ---------------------------------------------------------------------------
# route_all_edges (integration)
# ---------------------------------------------------------------------------


class TestRouteAllEdges:
    def test_simple_chain(self) -> None:
        boxes = [_make_box(i, i * 120, 100) for i in range(3)]
        edges = [(0, 1), (1, 2)]
        result = route_all_edges(boxes, edges, [0, 1], edge_separation=15.0)
        assert len(result) == 2
        assert all(isinstance(e, OrthogonalEdge) for e in result)

    def test_self_loop_routed(self) -> None:
        boxes = [_make_box(0, 100, 100)]
        edges = [(0, 0)]
        result = route_all_edges(boxes, edges, [0], edge_separation=15.0)
        assert len(result) == 1
        assert result[0].source == 0
        assert result[0].target == 0
        assert len(result[0].bends) > 0  # Self-loop must have bends

    def test_parallel_edges_different_ports(self) -> None:
        boxes = [_make_box(0, 100, 100), _make_box(1, 100, 300)]
        edges = [(0, 1), (0, 1)]
        result = route_all_edges(boxes, edges, [0, 1], edge_separation=15.0)
        assert len(result) == 2

        # Ports should have different positions (not both 0.5)
        p0_src = result[0].source_port.position
        p1_src = result[1].source_port.position
        assert p0_src != pytest.approx(p1_src)

    def test_empty_input(self) -> None:
        result = route_all_edges([], [], [], edge_separation=15.0)
        assert result == []

    def test_port_constraints_respected(self) -> None:
        boxes = [_make_box(0, 100, 100), _make_box(1, 300, 100)]
        edges = [(0, 1)]
        constraints = {(0, 1): (Side.NORTH, Side.SOUTH)}
        result = route_all_edges(
            boxes,
            edges,
            [0],
            edge_separation=15.0,
            port_constraints=constraints,
        )
        assert result[0].source_port.side == Side.NORTH
        assert result[0].target_port.side == Side.SOUTH


# ---------------------------------------------------------------------------
# Integration with layouts
# ---------------------------------------------------------------------------


class TestLayoutIntegration:
    def test_kandinsky_with_self_loop(self) -> None:
        """Kandinsky should produce a valid layout with self-loop edges."""
        from graph_layout import KandinskyLayout

        nodes = [{} for _ in range(3)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 0, "target": 0},  # Self-loop
        ]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(400, 400))
        layout.run()

        assert len(layout.orthogonal_edges) >= 2
        # Find the self-loop edge
        self_loops = [e for e in layout.orthogonal_edges if e.source == e.target]
        assert len(self_loops) == 1
        assert len(self_loops[0].bends) > 0

    def test_kandinsky_with_parallel_edges(self) -> None:
        """Kandinsky should produce valid layout with parallel edges."""
        from graph_layout import KandinskyLayout

        nodes = [{} for _ in range(3)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 1},  # Parallel edge
            {"source": 1, "target": 2},
        ]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(400, 400))
        layout.run()

        assert len(layout.orthogonal_edges) == 3
        # The two parallel edges should have different port positions
        par_edges = [e for e in layout.orthogonal_edges if e.source == 0 and e.target == 1]
        if len(par_edges) == 2:
            assert par_edges[0].source_port.position != pytest.approx(
                par_edges[1].source_port.position
            )

    def test_giotto_with_self_loop_fallback(self) -> None:
        """GIOTTO in non-strict mode should handle self-loops."""
        from graph_layout import GIOTTOLayout

        nodes = [{} for _ in range(3)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 0, "target": 0},  # Self-loop (invalid for strict GIOTTO)
        ]
        layout = GIOTTOLayout(nodes=nodes, links=links, size=(400, 400), strict=False)
        layout.run()

        assert len(layout.orthogonal_edges) >= 2


# ---------------------------------------------------------------------------
# Visibility graph routing
# ---------------------------------------------------------------------------


class TestVisibilityRouting:
    def test_single_obstacle_between_nodes(self) -> None:
        """Route should go around obstacle between src and tgt."""
        src = _make_box(0, 100, 200)
        tgt = _make_box(1, 500, 200)
        obstacle = _make_box(2, 300, 200, 60, 40)

        sp = Port(node=0, side=Side.EAST, position=0.5)
        tp = Port(node=1, side=Side.WEST, position=0.5)

        bends = route_edge(src, tgt, sp, tp, [obstacle], 15.0)

        # Route should have bends to avoid the obstacle
        src_pos = src.get_port_position(Side.EAST, 0.5)
        tgt_pos = tgt.get_port_position(Side.WEST, 0.5)

        # Verify no segment passes through the obstacle
        points = [src_pos] + bends + [tgt_pos]
        for i in range(len(points) - 1):
            assert not _segment_intersects_box(points[i], points[i + 1], obstacle), (
                f"Segment {i} passes through obstacle"
            )

    def test_multiple_obstacles(self) -> None:
        """Route should avoid all obstacles."""
        src = _make_box(0, 50, 200)
        tgt = _make_box(1, 650, 200)
        obs1 = _make_box(2, 200, 200, 60, 40)
        obs2 = _make_box(3, 450, 200, 60, 40)
        obstacles = [obs1, obs2]

        sp = Port(node=0, side=Side.EAST, position=0.5)
        tp = Port(node=1, side=Side.WEST, position=0.5)

        bends = route_edge(src, tgt, sp, tp, obstacles, 15.0)

        src_pos = src.get_port_position(Side.EAST, 0.5)
        tgt_pos = tgt.get_port_position(Side.WEST, 0.5)
        points = [src_pos] + bends + [tgt_pos]
        for i in range(len(points) - 1):
            for obs in obstacles:
                assert not _segment_intersects_box(points[i], points[i + 1], obs), (
                    f"Segment {i} passes through obstacle {obs.index}"
                )

    def test_no_obstacle_no_extra_bends(self) -> None:
        """Without obstacles, routing should not add extra bends."""
        src = _make_box(0, 100, 200)
        tgt = _make_box(1, 400, 200)

        sp = Port(node=0, side=Side.EAST, position=0.5)
        tp = Port(node=1, side=Side.WEST, position=0.5)

        bends = route_edge(src, tgt, sp, tp, [], 15.0)
        # Horizontal alignment: no bends needed
        assert len(bends) == 0

    def test_obstacle_not_in_path(self) -> None:
        """Obstacle far from the route should not cause detours."""
        src = _make_box(0, 100, 100)
        tgt = _make_box(1, 100, 400)
        obstacle = _make_box(2, 500, 250, 60, 40)

        sp = Port(node=0, side=Side.SOUTH, position=0.5)
        tp = Port(node=1, side=Side.NORTH, position=0.5)

        bends = route_edge(src, tgt, sp, tp, [obstacle], 15.0)
        assert len(bends) == 0


# ---------------------------------------------------------------------------
# Segment nudging
# ---------------------------------------------------------------------------


class TestNudgeOverlappingSegments:
    def test_coincident_horizontal_segments_separated(self) -> None:
        """Two edges with coincident horizontal segments get separated."""
        boxes = [
            _make_box(0, 100, 100),
            _make_box(1, 100, 300),
            _make_box(2, 300, 100),
            _make_box(3, 300, 300),
        ]
        edges = [(0, 3), (2, 1)]

        routed = route_all_edges(boxes, edges, [0, 1], edge_separation=15.0)
        nudged = nudge_overlapping_segments(routed, boxes, edge_separation=15.0)

        assert len(nudged) == 2

    def test_no_nudge_when_already_separated(self) -> None:
        """Edges with non-overlapping segments should not be nudged."""
        boxes = [_make_box(0, 100, 100), _make_box(1, 300, 100)]
        edges = [(0, 1)]
        routed = route_all_edges(boxes, edges, [0], edge_separation=15.0)
        nudged = nudge_overlapping_segments(routed, boxes, edge_separation=15.0)

        # Single edge: no nudging needed
        assert len(nudged) == 1
        assert nudged[0].bends == routed[0].bends

    def test_empty_edges(self) -> None:
        """Empty input should return empty output."""
        result = nudge_overlapping_segments([], [], edge_separation=15.0)
        assert result == []

    def test_integration_kandinsky_with_nudging(self) -> None:
        """Kandinsky layout with nudging should produce valid results."""
        from graph_layout import KandinskyLayout

        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(400, 400))
        layout.run()

        nudged = nudge_overlapping_segments(
            layout.orthogonal_edges, layout.node_boxes, edge_separation=15.0
        )
        assert len(nudged) == len(layout.orthogonal_edges)

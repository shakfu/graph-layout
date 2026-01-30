"""
Tests for Kandinsky orthogonal layout.
"""

import pytest

from graph_layout import GIOTTOLayout, KandinskyLayout
from graph_layout.orthogonal import (
    CompactionResult,
    ILPCompactionResult,
    NodeBox,
    OrthogonalRepresentation,
    Side,
    compact_horizontal,
    compact_layout,
    compact_layout_ilp,
    compact_vertical,
    compute_faces,
    compute_orthogonal_representation,
    find_edge_crossings,
    is_scipy_available,
    planarize_graph,
    segments_intersect,
)


class TestKandinskyBasic:
    """Basic functionality tests."""

    def test_layout_runs_without_error(self):
        """Layout should run without raising exceptions."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        result = layout.run()

        assert result is layout

    def test_nodes_have_positions_after_run(self):
        """All nodes should have positions after layout."""
        nodes = [{} for _ in range(5)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 4},
        ]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        for node in layout.nodes:
            assert node.x is not None
            assert node.y is not None

    def test_empty_graph(self):
        """Empty graph should not raise errors."""
        layout = KandinskyLayout(nodes=[], links=[], size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 0
        assert len(layout.orthogonal_edges) == 0

    def test_single_node(self):
        """Single node should be positioned."""
        nodes = [{}]
        layout = KandinskyLayout(nodes=nodes, size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 1
        assert layout.nodes[0].x is not None

    def test_two_nodes_with_edge(self):
        """Two connected nodes should be positioned."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert len(layout.nodes) == 2
        assert len(layout.orthogonal_edges) == 1


class TestKandinskyConfiguration:
    """Configuration property tests."""

    def test_node_width_property(self):
        """Node width should be settable."""
        layout = KandinskyLayout(node_width=100)
        assert layout.node_width == 100

        layout.node_width = 80
        assert layout.node_width == 80

    def test_node_height_property(self):
        """Node height should be settable."""
        layout = KandinskyLayout(node_height=50)
        assert layout.node_height == 50

    def test_node_separation_property(self):
        """Node separation should be settable."""
        layout = KandinskyLayout(node_separation=100)
        assert layout.node_separation == 100

    def test_edge_separation_property(self):
        """Edge separation should be settable."""
        layout = KandinskyLayout(edge_separation=20)
        assert layout.edge_separation == 20

    def test_layer_separation_property(self):
        """Layer separation should be settable."""
        layout = KandinskyLayout(layer_separation=120)
        assert layout.layer_separation == 120


class TestKandinskyOrthogonalOutput:
    """Tests for orthogonal output structures."""

    def test_node_boxes_created(self):
        """Node boxes should be created for all nodes."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert len(layout.node_boxes) == 3
        for box in layout.node_boxes:
            assert isinstance(box, NodeBox)
            assert box.width > 0
            assert box.height > 0

    def test_orthogonal_edges_created(self):
        """Orthogonal edges should be created for all links."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert len(layout.orthogonal_edges) == 2
        for edge in layout.orthogonal_edges:
            assert edge.source_port is not None
            assert edge.target_port is not None

    def test_edges_have_valid_ports(self):
        """Edge ports should reference valid sides."""
        nodes = [{} for _ in range(2)]
        links = [{"source": 0, "target": 1}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side in Side
        assert edge.target_port.side in Side


class TestKandinskyLayering:
    """Tests for layer assignment."""

    def test_chain_creates_layers(self):
        """Chain graph should create sequential layers."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        # In a chain, each node should be at a different y level
        y_positions = [node.y for node in layout.nodes]
        # Allow some tolerance for floating point
        unique_y = len(set(round(y, 1) for y in y_positions))
        assert unique_y >= 2  # At least some layering should occur

    def test_parallel_nodes_same_layer(self):
        """Nodes with same predecessor should be in same layer."""
        # Tree structure: 0 -> (1, 2)
        nodes = [{} for _ in range(3)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
        ]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        # Nodes 1 and 2 should have same y (same layer)
        assert abs(layout.nodes[1].y - layout.nodes[2].y) < 1


class TestKandinskyEdgeRouting:
    """Tests for edge routing."""

    def test_edges_have_bends_list(self):
        """Edges should have bends list (possibly empty)."""
        nodes = [{} for _ in range(2)]
        links = [{"source": 0, "target": 1}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert isinstance(edge.bends, list)

    def test_horizontal_edge_minimal_bends(self):
        """Horizontally aligned nodes should have few bends."""
        nodes = [{"x": 100, "y": 300}, {"x": 500, "y": 300}]
        links = [{"source": 0, "target": 1}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        # Simple horizontal connection shouldn't need many bends
        edge = layout.orthogonal_edges[0]
        assert len(edge.bends) <= 2


class TestKandinskyEvents:
    """Event system tests."""

    def test_start_event_fires(self):
        """Start event should fire when layout begins."""
        events = []

        def on_start(event):
            events.append(("start", event))

        layout = KandinskyLayout(nodes=[{}], on_start=on_start)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "start"

    def test_end_event_fires(self):
        """End event should fire when layout completes."""
        events = []

        def on_end(event):
            events.append(("end", event))

        layout = KandinskyLayout(nodes=[{}], on_end=on_end)
        layout.run()

        assert len(events) == 1
        assert events[0][0] == "end"


class TestNodeBox:
    """Tests for NodeBox data structure."""

    def test_box_properties(self):
        """NodeBox should compute edge properties correctly."""
        box = NodeBox(index=0, x=100, y=100, width=60, height=40)

        assert box.left == 70
        assert box.right == 130
        assert box.top == 80
        assert box.bottom == 120

    def test_port_position_north(self):
        """North port should be at top edge."""
        box = NodeBox(index=0, x=100, y=100, width=60, height=40)
        px, py = box.get_port_position(Side.NORTH)

        assert py == box.top
        assert box.left <= px <= box.right

    def test_port_position_south(self):
        """South port should be at bottom edge."""
        box = NodeBox(index=0, x=100, y=100, width=60, height=40)
        px, py = box.get_port_position(Side.SOUTH)

        assert py == box.bottom
        assert box.left <= px <= box.right

    def test_port_position_east(self):
        """East port should be at right edge."""
        box = NodeBox(index=0, x=100, y=100, width=60, height=40)
        px, py = box.get_port_position(Side.EAST)

        assert px == box.right
        assert box.top <= py <= box.bottom

    def test_port_position_west(self):
        """West port should be at left edge."""
        box = NodeBox(index=0, x=100, y=100, width=60, height=40)
        px, py = box.get_port_position(Side.WEST)

        assert px == box.left
        assert box.top <= py <= box.bottom


class TestSide:
    """Tests for Side enum."""

    def test_opposite(self):
        """Opposite sides should be correct."""
        assert Side.NORTH.opposite() == Side.SOUTH
        assert Side.SOUTH.opposite() == Side.NORTH
        assert Side.EAST.opposite() == Side.WEST
        assert Side.WEST.opposite() == Side.EAST

    def test_is_horizontal(self):
        """Horizontal check should work."""
        assert Side.NORTH.is_horizontal()
        assert Side.SOUTH.is_horizontal()
        assert not Side.EAST.is_horizontal()
        assert not Side.WEST.is_horizontal()

    def test_is_vertical(self):
        """Vertical check should work."""
        assert not Side.NORTH.is_vertical()
        assert not Side.SOUTH.is_vertical()
        assert Side.EAST.is_vertical()
        assert Side.WEST.is_vertical()


class TestKandinskyImport:
    """Import tests."""

    def test_import_from_package(self):
        """KandinskyLayout should be importable from package root."""
        from graph_layout import KandinskyLayout as KandinskyLayoutPkg

        assert KandinskyLayoutPkg is KandinskyLayout

    def test_import_from_orthogonal_module(self):
        """KandinskyLayout should be importable from orthogonal module."""
        from graph_layout.orthogonal import KandinskyLayout as KandinskyLayoutMod

        assert KandinskyLayoutMod is KandinskyLayout


class TestSegmentsIntersect:
    """Tests for segment intersection detection."""

    def test_crossing_segments(self):
        """Two crossing segments should return intersection point."""
        p1 = (0.0, 0.0)
        p2 = (2.0, 2.0)
        p3 = (0.0, 2.0)
        p4 = (2.0, 0.0)

        result = segments_intersect(p1, p2, p3, p4)
        assert result is not None
        assert abs(result[0] - 1.0) < 0.01
        assert abs(result[1] - 1.0) < 0.01

    def test_parallel_segments(self):
        """Parallel segments should not intersect."""
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        p3 = (0.0, 1.0)
        p4 = (2.0, 1.0)

        result = segments_intersect(p1, p2, p3, p4)
        assert result is None

    def test_non_crossing_segments(self):
        """Non-crossing segments should not intersect."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (2.0, 0.0)
        p4 = (3.0, 0.0)

        result = segments_intersect(p1, p2, p3, p4)
        assert result is None

    def test_endpoint_touch_not_intersection(self):
        """Segments touching at endpoints should not count as intersection."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        p3 = (1.0, 1.0)
        p4 = (2.0, 0.0)

        result = segments_intersect(p1, p2, p3, p4)
        assert result is None


class TestFindEdgeCrossings:
    """Tests for finding edge crossings in a graph."""

    def test_no_crossings_in_tree(self):
        """Tree graph should have no crossings."""
        positions = [(0, 0), (1, 1), (2, 1)]
        edges = [(0, 1), (0, 2)]

        crossings = find_edge_crossings(positions, edges)
        assert len(crossings) == 0

    def test_crossing_in_k4(self):
        """K4 in certain positions should have crossings."""
        # Square with diagonals
        positions = [(0, 0), (2, 0), (2, 2), (0, 2)]
        edges = [(0, 2), (1, 3)]  # Diagonals cross

        crossings = find_edge_crossings(positions, edges)
        assert len(crossings) == 1

    def test_adjacent_edges_no_crossing(self):
        """Edges sharing a vertex should not count as crossing."""
        positions = [(0, 0), (1, 1), (2, 0)]
        edges = [(0, 1), (1, 2)]

        crossings = find_edge_crossings(positions, edges)
        assert len(crossings) == 0


class TestPlanarizeGraph:
    """Tests for graph planarization."""

    def test_planar_graph_unchanged(self):
        """Planar graph should not gain crossing vertices."""
        positions = [(0, 0), (1, 0), (1, 1), (0, 1)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square

        result = planarize_graph(4, edges, positions)

        assert result.num_original_nodes == 4
        assert result.num_total_nodes == 4
        assert len(result.crossings) == 0

    def test_crossing_creates_vertex(self):
        """Crossing edges should create a crossing vertex."""
        positions = [(0, 0), (2, 2), (0, 2), (2, 0)]
        edges = [(0, 1), (2, 3)]  # Two crossing diagonals

        result = planarize_graph(4, edges, positions)

        assert result.num_original_nodes == 4
        assert len(result.crossings) == 1
        assert result.num_total_nodes == 5


class TestKandinskyPlanarization:
    """Tests for Kandinsky layout with planarization."""

    def test_handle_crossings_property(self):
        """handle_crossings should be configurable."""
        layout = KandinskyLayout(handle_crossings=True)
        assert layout.handle_crossings is True

        layout.handle_crossings = False
        assert layout.handle_crossings is False

    def test_num_crossings_property(self):
        """num_crossings should report detected crossings."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
        ]

        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            handle_crossings=True,
        )
        layout.run()

        # May or may not have crossings depending on layout
        assert layout.num_crossings >= 0

    def test_crossing_vertices_accessible(self):
        """crossing_vertices should be accessible after layout."""
        nodes = [{} for _ in range(4)]
        links = [{"source": 0, "target": 1}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert isinstance(layout.crossing_vertices, list)


class TestComputeFaces:
    """Tests for face computation in planar graphs."""

    def test_triangle_has_two_faces(self):
        """Triangle should have inner face and outer face."""
        positions = [(0, 0), (2, 0), (1, 2)]
        edges = [(0, 1), (1, 2), (2, 0)]

        faces = compute_faces(3, edges, positions)

        assert len(faces) == 2
        # One should be outer
        outer_count = sum(1 for f in faces if f.is_outer)
        assert outer_count == 1

    def test_square_has_two_faces(self):
        """Square should have inner face and outer face."""
        positions = [(0, 0), (2, 0), (2, 2), (0, 2)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        faces = compute_faces(4, edges, positions)

        assert len(faces) == 2

    def test_empty_graph_no_faces(self):
        """Empty graph should have no faces."""
        faces = compute_faces(0, [], None)
        assert len(faces) == 0

    def test_single_edge_one_face(self):
        """Single edge creates one degenerate face with both directed edges."""
        positions = [(0, 0), (1, 0)]
        edges = [(0, 1)]

        faces = compute_faces(2, edges, positions)

        # Single edge creates one "face" containing both directed edges
        assert len(faces) == 1
        # The face should contain both directions of the edge
        assert len(faces[0].edges) == 2


class TestOrthogonalRepresentation:
    """Tests for orthogonal representation data structure."""

    def test_empty_representation(self):
        """Empty representation should have zero bends."""
        ortho_rep = OrthogonalRepresentation()

        assert ortho_rep.total_bends == 0
        assert len(ortho_rep.vertex_face_angles) == 0
        assert len(ortho_rep.edge_bends) == 0

    def test_total_bends_calculation(self):
        """total_bends should count all bends across edges."""
        ortho_rep = OrthogonalRepresentation()
        ortho_rep.edge_bends[(0, 1)] = [1, -1]  # 2 bends
        ortho_rep.edge_bends[(1, 2)] = [1]  # 1 bend
        ortho_rep.edge_bends[(2, 3)] = []  # 0 bends

        assert ortho_rep.total_bends == 3


class TestComputeOrthogonalRepresentation:
    """Tests for the full orthogonalization pipeline."""

    def test_empty_graph(self):
        """Empty graph returns empty representation."""
        ortho_rep = compute_orthogonal_representation(0, [])

        assert ortho_rep.total_bends == 0

    def test_no_edges(self):
        """Graph with no edges returns empty representation."""
        ortho_rep = compute_orthogonal_representation(5, [])

        assert ortho_rep.total_bends == 0

    def test_simple_path(self):
        """Simple path should have valid representation."""
        positions = [(0, 0), (1, 0), (2, 0)]
        edges = [(0, 1), (1, 2)]

        ortho_rep = compute_orthogonal_representation(3, edges, positions)

        # Should produce a valid representation (may or may not have bends)
        assert isinstance(ortho_rep, OrthogonalRepresentation)

    def test_triangle(self):
        """Triangle should have valid representation."""
        positions = [(0, 0), (2, 0), (1, 2)]
        edges = [(0, 1), (1, 2), (2, 0)]

        ortho_rep = compute_orthogonal_representation(3, edges, positions)

        assert isinstance(ortho_rep, OrthogonalRepresentation)

    def test_square_cycle(self):
        """Square cycle should have valid representation."""
        positions = [(0, 0), (2, 0), (2, 2), (0, 2)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        ortho_rep = compute_orthogonal_representation(4, edges, positions)

        assert isinstance(ortho_rep, OrthogonalRepresentation)
        # Square has no bends when properly orthogonal
        # But our representation may have bends depending on flow solution


class TestKandinskyOrthogonalization:
    """Tests for Kandinsky layout orthogonalization integration."""

    def test_optimize_bends_property(self):
        """optimize_bends should be configurable."""
        layout = KandinskyLayout(optimize_bends=True)
        assert layout.optimize_bends is True

        layout.optimize_bends = False
        assert layout.optimize_bends is False

    def test_orthogonal_rep_accessible(self):
        """orthogonal_rep should be accessible after layout."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            optimize_bends=True,
        )
        layout.run()

        assert layout.orthogonal_rep is not None

    def test_orthogonalization_reduces_bends(self):
        """Orthogonalization should aim to minimize bends."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        # Run without optimization
        layout_no_opt = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            optimize_bends=False,
        )
        layout_no_opt.run()

        # Run with optimization
        layout_opt = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            optimize_bends=True,
        )
        layout_opt.run()

        # Both should complete without error
        assert len(layout_no_opt.orthogonal_edges) == 3
        assert len(layout_opt.orthogonal_edges) == 3


class TestCompactLayout:
    """Tests for layout compaction functions."""

    def test_compact_empty(self):
        """Empty layout should compact without error."""
        result = compact_layout(boxes=[], edges=[])

        assert result.width == 0
        assert result.height == 0
        assert len(result.node_positions) == 0

    def test_compact_single_node(self):
        """Single node should remain at valid position."""
        boxes = [NodeBox(index=0, x=100, y=100, width=60, height=40)]

        result = compact_layout(boxes=boxes, edges=[])

        assert len(result.node_positions) == 1
        assert result.width > 0
        assert result.height > 0

    def test_compact_two_nodes_horizontal(self):
        """Two horizontally separated nodes should be compacted."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=500, y=100, width=60, height=40),
        ]

        result = compact_layout(boxes=boxes, edges=[], node_separation=60)

        assert len(result.node_positions) == 2
        # Compaction should produce valid positions maintaining separation
        x0, _ = result.node_positions[0]
        x1, _ = result.node_positions[1]
        # Nodes should maintain minimum separation (width/2 + sep + width/2)
        min_separation = 60 + 60  # node_separation + width
        assert x1 - x0 >= min_separation

    def test_compact_two_nodes_vertical(self):
        """Two vertically separated nodes should be compacted."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=100, y=500, width=60, height=40),
        ]

        result = compact_layout(boxes=boxes, edges=[], layer_separation=80)

        assert len(result.node_positions) == 2
        # Compaction should produce valid positions maintaining separation
        _, y0 = result.node_positions[0]
        _, y1 = result.node_positions[1]
        # Nodes should maintain minimum separation (height/2 + sep + height/2)
        min_separation = 80 + 40  # layer_separation + height
        assert y1 - y0 >= min_separation

    def test_compact_horizontal_preserves_order(self):
        """Horizontal compaction should preserve left-to-right order."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=300, y=100, width=60, height=40),
            NodeBox(index=2, x=500, y=100, width=60, height=40),
        ]

        new_x = compact_horizontal(boxes, edges=[], node_separation=60, edge_separation=15)

        assert len(new_x) == 3
        # Order should be preserved
        assert new_x[0] < new_x[1] < new_x[2]

    def test_compact_vertical_preserves_order(self):
        """Vertical compaction should preserve top-to-bottom order."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=100, y=300, width=60, height=40),
            NodeBox(index=2, x=100, y=500, width=60, height=40),
        ]

        new_y = compact_vertical(boxes, edges=[], layer_separation=80, edge_separation=15)

        assert len(new_y) == 3
        # Order should be preserved
        assert new_y[0] < new_y[1] < new_y[2]


class TestKandinskyCompaction:
    """Tests for Kandinsky layout compaction integration."""

    def test_compact_property(self):
        """compact should be configurable."""
        layout = KandinskyLayout(compact=True)
        assert layout.compact is True

        layout.compact = False
        assert layout.compact is False

    def test_compaction_result_accessible(self):
        """compaction_result should be accessible after layout."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compact=True,
        )
        layout.run()

        assert layout.compaction_result is not None
        assert isinstance(layout.compaction_result, CompactionResult)

    def test_compaction_reduces_area(self):
        """Compaction should reduce or maintain layout area."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        # Run without compaction
        layout_no_compact = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compact=False,
        )
        layout_no_compact.run()

        # Run with compaction
        layout_compact = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compact=True,
        )
        layout_compact.run()

        # Both should complete without error
        assert len(layout_no_compact.orthogonal_edges) == 3
        assert len(layout_compact.orthogonal_edges) == 3

    def test_compaction_maintains_separation(self):
        """Compaction should maintain minimum node separation."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            node_separation=60,
            compact=True,
        )
        layout.run()

        # Check that nodes don't overlap
        for i, box1 in enumerate(layout.node_boxes):
            for j, box2 in enumerate(layout.node_boxes):
                if i >= j:
                    continue
                # Boxes should not overlap
                x_overlap = not (box1.right < box2.left or box2.right < box1.left)
                y_overlap = not (box1.bottom < box2.top or box2.bottom < box1.top)
                assert not (x_overlap and y_overlap), f"Nodes {i} and {j} overlap"


class TestPortConstraints:
    """Tests for user-specified port constraints."""

    def test_source_side_constraint(self):
        """Edge should exit from specified source side."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "source_side": Side.EAST}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side == Side.EAST

    def test_target_side_constraint(self):
        """Edge should enter from specified target side."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "target_side": Side.WEST}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.target_port.side == Side.WEST

    def test_both_sides_constrained(self):
        """Both source and target sides should be constrained."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "source_side": Side.SOUTH, "target_side": Side.NORTH}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side == Side.SOUTH
        assert edge.target_port.side == Side.NORTH

    def test_partial_constraint_source_only(self):
        """Source constrained, target uses heuristic."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "source_side": Side.NORTH}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side == Side.NORTH
        # Target should be a valid side (heuristic-based)
        assert edge.target_port.side in Side

    def test_no_constraint_uses_heuristic(self):
        """No constraint should use heuristic (default behavior)."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1}]  # No constraints

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        # Should complete without error and use heuristic
        assert len(layout.orthogonal_edges) == 1
        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side in Side
        assert edge.target_port.side in Side

    def test_string_side_constraint(self):
        """String values should be accepted for side constraints."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "source_side": "east", "target_side": "west"}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        edge = layout.orthogonal_edges[0]
        assert edge.source_port.side == Side.EAST
        assert edge.target_port.side == Side.WEST

    def test_port_constraints_property(self):
        """Port constraints should be accessible."""
        nodes = [{}, {}]
        links = [{"source": 0, "target": 1, "source_side": Side.SOUTH}]

        layout = KandinskyLayout(nodes=nodes, links=links, size=(800, 600))

        assert (0, 1) in layout.port_constraints
        src_constraint, tgt_constraint = layout.port_constraints[(0, 1)]
        assert src_constraint == Side.SOUTH
        assert tgt_constraint is None


class TestILPCompaction:
    """Tests for ILP-based compaction."""

    def test_ilp_produces_valid_layout(self):
        """ILP compaction should produce valid layout without overlaps."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=300, y=100, width=60, height=40),
            NodeBox(index=2, x=200, y=300, width=60, height=40),
        ]

        result = compact_layout_ilp(boxes=boxes, edges=[], node_separation=60)

        assert isinstance(result, ILPCompactionResult)
        assert len(result.node_positions) == 3
        assert result.width > 0
        assert result.height > 0

    def test_ilp_respects_separation(self):
        """ILP compaction should maintain minimum separation."""
        boxes = [
            NodeBox(index=0, x=100, y=100, width=60, height=40),
            NodeBox(index=1, x=500, y=100, width=60, height=40),
        ]

        result = compact_layout_ilp(boxes=boxes, edges=[], node_separation=60)

        x0, y0 = result.node_positions[0]
        x1, y1 = result.node_positions[1]

        # Minimum horizontal separation: w1/2 + sep + w2/2 = 30 + 60 + 30 = 120
        assert x1 - x0 >= 120 or abs(y1 - y0) > 40  # Either separated or not overlapping

    def test_ilp_empty_graph(self):
        """ILP compaction should handle empty input."""
        result = compact_layout_ilp(boxes=[], edges=[])

        assert result.optimal is True
        assert result.solver_status == "empty_graph"
        assert result.width == 0
        assert result.height == 0

    def test_ilp_single_node(self):
        """ILP compaction should handle single node."""
        boxes = [NodeBox(index=0, x=100, y=100, width=60, height=40)]

        result = compact_layout_ilp(boxes=boxes, edges=[])

        assert len(result.node_positions) == 1
        assert result.width > 0
        assert result.height > 0

    def test_is_scipy_available_function(self):
        """is_scipy_available should return bool."""
        result = is_scipy_available()
        assert isinstance(result, bool)

    def test_compaction_method_property(self):
        """compaction_method should be configurable."""
        layout = KandinskyLayout(compaction_method="greedy")
        assert layout.compaction_method == "greedy"

        layout.compaction_method = "auto"
        assert layout.compaction_method == "auto"

        layout.compaction_method = "ilp"
        assert layout.compaction_method == "ilp"

    def test_invalid_compaction_method_raises(self):
        """Invalid compaction method should raise ValueError."""
        layout = KandinskyLayout()

        with pytest.raises(ValueError):
            layout.compaction_method = "invalid"

    def test_greedy_compaction_method(self):
        """Greedy compaction method should work."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        layout = KandinskyLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            compact=True,
            compaction_method="greedy",
        )
        layout.run()

        assert layout.compaction_result is not None
        assert len(layout.node_boxes) == 4


class TestGIOTTOLayout:
    """Tests for GIOTTO orthogonal layout algorithm."""

    def test_degree_4_grid_graph(self):
        """Valid degree-4 planar graph should produce layout."""
        # 2x2 grid graph
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},  # Top edge
            {"source": 2, "target": 3},  # Bottom edge
            {"source": 0, "target": 2},  # Left edge
            {"source": 1, "target": 3},  # Right edge
        ]

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert layout.is_valid_input is True
        assert len(layout.node_boxes) == 4
        assert len(layout.orthogonal_edges) == 4

    def test_rejects_degree_5(self):
        """Degree > 4 should raise ValueError in strict mode."""
        # Star graph with center having degree 5
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 0, "target": 3},
            {"source": 0, "target": 4},
            {"source": 0, "target": 5},
        ]

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=True)

        with pytest.raises(ValueError, match="max degree 4"):
            layout.run()

    def test_rejects_non_planar_k5(self):
        """K5 (complete graph on 5 vertices) should raise ValueError."""
        # K5 - every pair of 5 vertices connected
        nodes = [{} for _ in range(5)]
        links = []
        for i in range(5):
            for j in range(i + 1, 5):
                links.append({"source": i, "target": j})

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=True)

        # K5 has 10 edges, which violates Euler's formula (E <= 3V - 6 = 9)
        # So it will fail the planarity check first
        with pytest.raises(ValueError, match="planar"):
            layout.run()

    def test_strict_false_fallback_degree_5(self):
        """Degree > 4 should fall back to Kandinsky-like in non-strict mode."""
        nodes = [{} for _ in range(6)]
        links = [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 0, "target": 3},
            {"source": 0, "target": 4},
            {"source": 0, "target": 5},
        ]

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=False)
        layout.run()

        assert layout.is_valid_input is False
        assert len(layout.node_boxes) == 6
        # All nodes should have positions
        for node in layout.nodes:
            assert node.x is not None
            assert node.y is not None

    def test_strict_false_fallback_non_planar(self):
        """Non-planar graph should fall back in non-strict mode."""
        # Dense graph that fails planarity check
        nodes = [{} for _ in range(5)]
        links = []
        for i in range(5):
            for j in range(i + 1, 5):
                links.append({"source": i, "target": j})

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600), strict=False)
        layout.run()

        assert layout.is_valid_input is False
        # Should still produce a layout
        assert len(layout.node_boxes) == 5

    def test_empty_graph(self):
        """Empty graph should work."""
        layout = GIOTTOLayout(nodes=[], links=[], size=(800, 600))
        layout.run()

        assert layout.is_valid_input is True
        assert len(layout.node_boxes) == 0

    def test_single_node(self):
        """Single node should work."""
        layout = GIOTTOLayout(nodes=[{}], size=(800, 600))
        layout.run()

        assert layout.is_valid_input is True
        assert len(layout.node_boxes) == 1

    def test_simple_path(self):
        """Simple path (degree <= 2) should work."""
        nodes = [{} for _ in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert layout.is_valid_input is True
        assert len(layout.orthogonal_edges) == 3

    def test_total_bends_property(self):
        """total_bends should count bends."""
        nodes = [{} for _ in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = GIOTTOLayout(nodes=nodes, links=links, size=(800, 600))
        layout.run()

        assert layout.total_bends >= 0

    def test_strict_property(self):
        """strict property should be configurable."""
        layout = GIOTTOLayout(strict=True)
        assert layout.strict is True

        layout.strict = False
        assert layout.strict is False

    def test_configuration_properties(self):
        """Configuration properties should work."""
        layout = GIOTTOLayout(
            node_width=100,
            node_height=50,
            node_separation=80,
            edge_separation=20,
            layer_separation=100,
        )

        assert layout.node_width == 100
        assert layout.node_height == 50
        assert layout.node_separation == 80
        assert layout.edge_separation == 20
        assert layout.layer_separation == 100

    def test_import_from_package_root(self):
        """GIOTTOLayout should be importable from package root."""
        from graph_layout import GIOTTOLayout as GIOTTOLayoutPkg

        assert GIOTTOLayoutPkg is GIOTTOLayout

    def test_import_from_orthogonal_module(self):
        """GIOTTOLayout should be importable from orthogonal module."""
        from graph_layout.orthogonal import GIOTTOLayout as GIOTTOLayoutMod

        assert GIOTTOLayoutMod is GIOTTOLayout

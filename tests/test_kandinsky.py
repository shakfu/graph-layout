"""
Tests for Kandinsky orthogonal layout.
"""

import math

import pytest

from graph_layout import KandinskyLayout, EventType
from graph_layout.orthogonal import Side, NodeBox, Port


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
        from graph_layout import KandinskyLayout as KL

        assert KL is KandinskyLayout

    def test_import_from_orthogonal_module(self):
        """KandinskyLayout should be importable from orthogonal module."""
        from graph_layout.orthogonal import KandinskyLayout as KL

        assert KL is KandinskyLayout

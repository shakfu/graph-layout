"""Tests for export functionality (SVG, DOT, GraphML)."""

import pytest

from graph_layout import CircularLayout, FruchtermanReingoldLayout
from graph_layout.export import (
    to_dot,
    to_dot_orthogonal,
    to_graphml,
    to_graphml_orthogonal,
    to_svg,
    to_svg_orthogonal,
)
from graph_layout.orthogonal.types import NodeBox, OrthogonalEdge, Port, Side

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_layout():
    """A simple circular layout for testing."""
    nodes = [{"index": i} for i in range(5)]
    links = [{"source": i, "target": (i + 1) % 5} for i in range(5)]
    return CircularLayout(nodes=nodes, links=links, size=(400, 400)).run()


@pytest.fixture
def labeled_layout():
    """A layout with labeled nodes."""
    nodes = [{"index": i, "name": f"Node_{i}"} for i in range(3)]
    links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
    return CircularLayout(nodes=nodes, links=links, size=(300, 300)).run()


@pytest.fixture
def weighted_layout():
    """A layout with weighted edges."""
    nodes = [{"index": i} for i in range(4)]
    links = [
        {"source": 0, "target": 1, "weight": 1.5},
        {"source": 1, "target": 2, "weight": 2.0},
        {"source": 2, "target": 3, "length": 100.0},
    ]
    return CircularLayout(nodes=nodes, links=links, size=(400, 400)).run()


@pytest.fixture
def orthogonal_boxes():
    """Sample orthogonal layout boxes."""
    return [
        NodeBox(index=0, x=100, y=100, width=60, height=40),
        NodeBox(index=1, x=250, y=100, width=60, height=40),
        NodeBox(index=2, x=175, y=200, width=60, height=40),
    ]


@pytest.fixture
def orthogonal_edges(orthogonal_boxes):
    """Sample orthogonal edges with bends."""
    boxes = orthogonal_boxes
    return [
        OrthogonalEdge(
            source=0,
            target=1,
            source_port=Port(node=0, side=Side.EAST),
            target_port=Port(node=1, side=Side.WEST),
            bends=[],
        ),
        OrthogonalEdge(
            source=0,
            target=2,
            source_port=Port(node=0, side=Side.SOUTH),
            target_port=Port(node=2, side=Side.NORTH),
            bends=[(100, 150)],
        ),
        OrthogonalEdge(
            source=1,
            target=2,
            source_port=Port(node=1, side=Side.SOUTH),
            target_port=Port(node=2, side=Side.NORTH),
            bends=[(250, 150), (175, 150)],
        ),
    ]


# =============================================================================
# SVG Export Tests
# =============================================================================


class TestSVGExport:
    """Tests for SVG export."""

    def test_basic_svg_export(self, simple_layout):
        """Test basic SVG export produces valid output."""
        svg = to_svg(simple_layout)

        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_svg_contains_nodes(self, simple_layout):
        """Test that SVG contains node elements."""
        svg = to_svg(simple_layout)

        assert '<g class="nodes">' in svg
        assert "<circle" in svg or "<rect" in svg

    def test_svg_contains_edges(self, simple_layout):
        """Test that SVG contains edge elements."""
        svg = to_svg(simple_layout)

        assert '<g class="edges">' in svg
        assert "<line" in svg

    def test_svg_contains_labels(self, simple_layout):
        """Test that SVG contains labels when enabled."""
        svg = to_svg(simple_layout, show_labels=True)

        assert '<g class="labels">' in svg
        assert "<text" in svg

    def test_svg_no_labels_when_disabled(self, simple_layout):
        """Test that labels are omitted when disabled."""
        svg = to_svg(simple_layout, show_labels=False)

        assert '<g class="labels">' not in svg

    def test_svg_custom_colors(self, simple_layout):
        """Test SVG with custom colors."""
        svg = to_svg(
            simple_layout,
            node_color="#ff0000",
            edge_color="#00ff00",
            label_color="#0000ff",
        )

        assert 'fill="#ff0000"' in svg
        assert 'stroke="#00ff00"' in svg
        assert 'fill="#0000ff"' in svg

    def test_svg_background(self, simple_layout):
        """Test SVG with background color."""
        svg = to_svg(simple_layout, background="#ffffff")

        assert 'fill="#ffffff"' in svg

    def test_svg_rect_nodes(self, simple_layout):
        """Test SVG with rectangular nodes."""
        svg = to_svg(simple_layout, node_shape="rect")

        assert "<rect" in svg

    def test_svg_empty_layout(self):
        """Test SVG export with empty layout."""
        layout = CircularLayout(nodes=[], links=[], size=(100, 100)).run()
        svg = to_svg(layout)

        assert "<svg" in svg
        assert "</svg>" in svg

    def test_svg_orthogonal_basic(self, orthogonal_boxes, orthogonal_edges):
        """Test orthogonal SVG export."""
        svg = to_svg_orthogonal(orthogonal_boxes, orthogonal_edges)

        assert svg.startswith("<svg")
        assert '<g class="nodes">' in svg
        assert '<g class="edges">' in svg
        assert "<rect" in svg  # Orthogonal uses rectangles
        assert "<polyline" in svg  # Edges with bends

    def test_svg_orthogonal_with_bends(self, orthogonal_boxes, orthogonal_edges):
        """Test that orthogonal SVG includes bend points."""
        svg = to_svg_orthogonal(orthogonal_boxes, orthogonal_edges)

        # The polyline should have multiple points for edges with bends
        assert "<polyline" in svg


# =============================================================================
# DOT Export Tests
# =============================================================================


class TestDOTExport:
    """Tests for DOT (Graphviz) export."""

    def test_basic_dot_export(self, simple_layout):
        """Test basic DOT export produces valid output."""
        dot = to_dot(simple_layout)

        assert dot.startswith("graph G {")
        assert dot.endswith("}")

    def test_dot_directed(self, simple_layout):
        """Test DOT export for directed graphs."""
        dot = to_dot(simple_layout, directed=True)

        assert dot.startswith("digraph G {")
        assert "->" in dot

    def test_dot_undirected(self, simple_layout):
        """Test DOT export for undirected graphs."""
        dot = to_dot(simple_layout, directed=False)

        assert dot.startswith("graph G {")
        assert "--" in dot

    def test_dot_contains_nodes(self, simple_layout):
        """Test that DOT contains node definitions."""
        dot = to_dot(simple_layout)

        # Should have node definitions
        for i in range(5):
            assert f"{i}" in dot

    def test_dot_contains_positions(self, simple_layout):
        """Test that DOT includes position attributes."""
        dot = to_dot(simple_layout, include_positions=True)

        assert "pos=" in dot

    def test_dot_no_positions(self, simple_layout):
        """Test DOT without position attributes."""
        dot = to_dot(simple_layout, include_positions=False)

        assert "pos=" not in dot

    def test_dot_custom_name(self, simple_layout):
        """Test DOT with custom graph name."""
        dot = to_dot(simple_layout, name="MyGraph")

        assert "graph MyGraph {" in dot

    def test_dot_node_attributes(self, simple_layout):
        """Test DOT with custom node attributes."""
        dot = to_dot(simple_layout, node_shape="box", node_fillcolor="#aabbcc")

        assert "shape=box" in dot
        assert "fillcolor=" in dot
        assert "style=filled" in dot

    def test_dot_edge_weights(self, weighted_layout):
        """Test that DOT includes edge weights."""
        dot = to_dot(weighted_layout)

        assert "weight=" in dot

    def test_dot_labeled_nodes(self, labeled_layout):
        """Test DOT with labeled nodes."""
        dot = to_dot(labeled_layout)

        assert "Node_0" in dot
        assert "Node_1" in dot

    def test_dot_custom_labels(self, simple_layout):
        """Test DOT with custom label function."""
        dot = to_dot(simple_layout, get_node_label=lambda n: f"Custom_{n.index}")

        assert "Custom_0" in dot
        assert "Custom_1" in dot

    def test_dot_orthogonal(self, orthogonal_boxes, orthogonal_edges):
        """Test orthogonal DOT export."""
        dot = to_dot_orthogonal(orthogonal_boxes, orthogonal_edges)

        assert "graph G {" in dot
        assert "splines=ortho" in dot
        assert "shape=box" in dot


# =============================================================================
# GraphML Export Tests
# =============================================================================


class TestGraphMLExport:
    """Tests for GraphML export."""

    def test_basic_graphml_export(self, simple_layout):
        """Test basic GraphML export produces valid XML."""
        graphml = to_graphml(simple_layout)

        assert graphml.startswith('<?xml version="1.0"')
        assert "<graphml" in graphml
        assert "</graphml>" in graphml

    def test_graphml_contains_nodes(self, simple_layout):
        """Test that GraphML contains node elements."""
        graphml = to_graphml(simple_layout)

        assert "<node id=" in graphml
        assert "</node>" in graphml

    def test_graphml_contains_edges(self, simple_layout):
        """Test that GraphML contains edge elements."""
        graphml = to_graphml(simple_layout)

        assert "<edge id=" in graphml
        assert 'source="' in graphml
        assert 'target="' in graphml

    def test_graphml_directed(self, simple_layout):
        """Test GraphML with directed edges."""
        graphml = to_graphml(simple_layout, directed=True)

        assert 'edgedefault="directed"' in graphml

    def test_graphml_undirected(self, simple_layout):
        """Test GraphML with undirected edges."""
        graphml = to_graphml(simple_layout, directed=False)

        assert 'edgedefault="undirected"' in graphml

    def test_graphml_positions(self, simple_layout):
        """Test GraphML includes position data."""
        graphml = to_graphml(simple_layout, include_positions=True)

        assert 'attr.name="x"' in graphml
        assert 'attr.name="y"' in graphml
        assert "<data key=" in graphml

    def test_graphml_no_positions(self, simple_layout):
        """Test GraphML without position data."""
        graphml = to_graphml(simple_layout, include_positions=False)

        # Should not have x/y keys
        lines = graphml.split("\n")
        position_keys = [l for l in lines if 'attr.name="x"' in l or 'attr.name="y"' in l]
        assert len(position_keys) == 0

    def test_graphml_weights(self, weighted_layout):
        """Test GraphML includes edge weights."""
        graphml = to_graphml(weighted_layout, include_weights=True)

        assert 'attr.name="weight"' in graphml

    def test_graphml_labels(self, labeled_layout):
        """Test GraphML includes node labels."""
        graphml = to_graphml(labeled_layout)

        assert 'attr.name="label"' in graphml
        assert "Node_0" in graphml

    def test_graphml_custom_graph_id(self, simple_layout):
        """Test GraphML with custom graph ID."""
        graphml = to_graphml(simple_layout, graph_id="CustomGraph")

        assert 'id="CustomGraph"' in graphml

    def test_graphml_orthogonal(self, orthogonal_boxes, orthogonal_edges):
        """Test orthogonal GraphML export."""
        graphml = to_graphml_orthogonal(orthogonal_boxes, orthogonal_edges)

        assert "<graphml" in graphml
        assert "<node id=" in graphml
        assert "<edge id=" in graphml

    def test_graphml_orthogonal_bends(self, orthogonal_boxes, orthogonal_edges):
        """Test orthogonal GraphML includes bend data."""
        graphml = to_graphml_orthogonal(orthogonal_boxes, orthogonal_edges, include_bends=True)

        assert 'attr.name="bends"' in graphml

    def test_graphml_orthogonal_port_sides(self, orthogonal_boxes, orthogonal_edges):
        """Test orthogonal GraphML includes port side data."""
        graphml = to_graphml_orthogonal(orthogonal_boxes, orthogonal_edges, include_bends=True)

        assert 'attr.name="source_side"' in graphml
        assert 'attr.name="target_side"' in graphml
        assert "east" in graphml or "west" in graphml or "north" in graphml


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestExportEdgeCases:
    """Tests for edge cases in export functions."""

    def test_empty_graph_svg(self):
        """Test SVG export with no nodes."""
        layout = CircularLayout(nodes=[], links=[], size=(100, 100)).run()
        svg = to_svg(layout)
        assert "<svg" in svg

    def test_empty_graph_dot(self):
        """Test DOT export with no nodes."""
        layout = CircularLayout(nodes=[], links=[], size=(100, 100)).run()
        dot = to_dot(layout)
        assert "graph G {" in dot

    def test_empty_graph_graphml(self):
        """Test GraphML export with no nodes."""
        layout = CircularLayout(nodes=[], links=[], size=(100, 100)).run()
        graphml = to_graphml(layout)
        assert "<graphml" in graphml

    def test_single_node_svg(self):
        """Test SVG export with single node."""
        layout = CircularLayout(nodes=[{"index": 0}], links=[], size=(100, 100)).run()
        svg = to_svg(layout)
        assert "<circle" in svg or "<rect" in svg

    def test_single_node_dot(self):
        """Test DOT export with single node."""
        layout = CircularLayout(nodes=[{"index": 0}], links=[], size=(100, 100)).run()
        dot = to_dot(layout)
        assert "0" in dot

    def test_self_loop_svg(self):
        """Test SVG export with self-loop edge."""
        layout = CircularLayout(
            nodes=[{"index": 0}, {"index": 1}],
            links=[{"source": 0, "target": 0}],
            size=(100, 100),
        ).run()
        svg = to_svg(layout)
        # Self-loops should still produce valid SVG
        assert "<svg" in svg

    def test_special_chars_in_labels(self):
        """Test export handles special characters in labels."""
        nodes = [{"index": 0, "name": "<Node & 'Test'>"}]
        layout = CircularLayout(nodes=nodes, links=[], size=(100, 100)).run()

        svg = to_svg(layout)
        # Should be escaped
        assert "&lt;" in svg or "&amp;" in svg

        graphml = to_graphml(layout)
        assert "&lt;" in graphml or "&amp;" in graphml

    def test_large_coordinates_svg(self):
        """Test SVG handles large coordinate values."""
        nodes = [{"index": 0, "x": 10000, "y": 10000}]
        layout = FruchtermanReingoldLayout(nodes=nodes, links=[], size=(20000, 20000)).run()
        svg = to_svg(layout)
        assert "<svg" in svg


# =============================================================================
# Integration Tests
# =============================================================================


class TestExportIntegration:
    """Integration tests for export functionality."""

    def test_force_directed_layout_export(self):
        """Test export of force-directed layout."""
        nodes = [{"index": i} for i in range(10)]
        links = [{"source": i, "target": (i + 1) % 10} for i in range(10)]
        layout = FruchtermanReingoldLayout(
            nodes=nodes, links=links, size=(500, 500), iterations=50
        ).run()

        svg = to_svg(layout)
        dot = to_dot(layout)
        graphml = to_graphml(layout)

        assert "<svg" in svg
        assert "graph G {" in dot
        assert "<graphml" in graphml

    def test_all_formats_same_graph(self, simple_layout):
        """Test that all formats export the same graph structure."""
        svg = to_svg(simple_layout)
        dot = to_dot(simple_layout)
        graphml = to_graphml(simple_layout)

        # All should have 5 nodes
        assert svg.count("<circle") == 5 or svg.count("<rect") >= 5
        # DOT has 5 node definitions and 5 edges
        assert dot.count("--") == 5
        # GraphML has 5 nodes and 5 edges
        assert graphml.count("<node id=") == 5
        assert graphml.count("<edge id=") == 5


# =============================================================================
# Method-Based Export Tests (layout.to_svg(), layout.to_dot(), etc.)
# =============================================================================


class TestMethodBasedExports:
    """Tests for method-based export API on layout objects."""

    def test_to_svg_method(self, simple_layout):
        """Test layout.to_svg() method."""
        svg = simple_layout.to_svg()

        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "<circle" in svg or "<rect" in svg

    def test_to_dot_method(self, simple_layout):
        """Test layout.to_dot() method."""
        dot = simple_layout.to_dot()

        assert dot.startswith("graph G {")
        assert "--" in dot

    def test_to_graphml_method(self, simple_layout):
        """Test layout.to_graphml() method."""
        graphml = simple_layout.to_graphml()

        assert '<?xml version="1.0"' in graphml
        assert "<graphml" in graphml
        assert "<node id=" in graphml

    def test_methods_with_kwargs(self, simple_layout):
        """Test that methods pass kwargs correctly."""
        svg = simple_layout.to_svg(node_color="#ff0000", show_labels=False)
        assert 'fill="#ff0000"' in svg
        assert '<g class="labels">' not in svg

        dot = simple_layout.to_dot(directed=True, name="TestGraph")
        assert "digraph TestGraph {" in dot
        assert "->" in dot

        graphml = simple_layout.to_graphml(graph_id="MyGraph", include_positions=False)
        assert 'id="MyGraph"' in graphml

    def test_method_equivalence(self, simple_layout):
        """Test that method and function produce equivalent output."""
        # SVG
        svg_method = simple_layout.to_svg()
        svg_func = to_svg(simple_layout)
        assert svg_method == svg_func

        # DOT
        dot_method = simple_layout.to_dot()
        dot_func = to_dot(simple_layout)
        assert dot_method == dot_func

        # GraphML
        graphml_method = simple_layout.to_graphml()
        graphml_func = to_graphml(simple_layout)
        assert graphml_method == graphml_func


class TestOrthogonalMethodExports:
    """Tests for method-based exports on orthogonal layouts."""

    def test_kandinsky_to_svg(self):
        """Test KandinskyLayout.to_svg() method."""
        from graph_layout import KandinskyLayout

        nodes = [{"index": i} for i in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(400, 300)).run()

        svg = layout.to_svg()

        assert "<svg" in svg
        assert "<rect" in svg  # Orthogonal uses rectangles
        assert "<polyline" in svg  # Edges with potential bends

    def test_kandinsky_to_dot(self):
        """Test KandinskyLayout.to_dot() method."""
        from graph_layout import KandinskyLayout

        nodes = [{"index": i} for i in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(300, 200)).run()

        dot = layout.to_dot()

        assert "graph G {" in dot
        assert "splines=ortho" in dot
        assert "shape=box" in dot

    def test_kandinsky_to_graphml(self):
        """Test KandinskyLayout.to_graphml() method."""
        from graph_layout import KandinskyLayout

        nodes = [{"index": i} for i in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
        layout = KandinskyLayout(nodes=nodes, links=links, size=(300, 200)).run()

        graphml = layout.to_graphml()

        assert "<graphml" in graphml
        assert 'attr.name="bends"' in graphml
        assert 'attr.name="source_side"' in graphml

    def test_giotto_to_svg(self):
        """Test GIOTTOLayout.to_svg() method."""
        from graph_layout import GIOTTOLayout

        # Simple path graph (degree <= 2, trivially planar)
        nodes = [{"index": i} for i in range(4)]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ]
        layout = GIOTTOLayout(nodes=nodes, links=links, size=(400, 300)).run()

        svg = layout.to_svg()

        assert "<svg" in svg
        assert "<rect" in svg

    def test_giotto_to_dot(self):
        """Test GIOTTOLayout.to_dot() method."""
        from graph_layout import GIOTTOLayout

        nodes = [{"index": i} for i in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
        layout = GIOTTOLayout(nodes=nodes, links=links, size=(300, 200)).run()

        dot = layout.to_dot()

        assert "graph G {" in dot
        assert "splines=ortho" in dot

    def test_giotto_to_graphml(self):
        """Test GIOTTOLayout.to_graphml() method."""
        from graph_layout import GIOTTOLayout

        nodes = [{"index": i} for i in range(3)]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
        layout = GIOTTOLayout(nodes=nodes, links=links, size=(300, 200)).run()

        graphml = layout.to_graphml()

        assert "<graphml" in graphml
        assert "<node id=" in graphml

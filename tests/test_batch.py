"""Tests for batch module."""

from graph_layout.cola.batch import power_graph_grid_layout
from graph_layout.cola.layout import Link, Node


class TestPowerGraphGridLayout:
    """Test power_graph_grid_layout function."""

    def test_simple_graph(self):
        """Test with simple graph."""
        # Create simple graph
        nodes = [
            Node(x=0, y=0, width=10, height=10),
            Node(x=20, y=0, width=10, height=10),
            Node(x=10, y=20, width=10, height=10),
        ]

        links = [Link(source=0, target=1), Link(source=1, target=2)]

        graph = {"nodes": nodes, "links": links}

        # Run power graph grid layout
        result = power_graph_grid_layout(graph, [100, 100], 10)

        # Should return dict with cola and powerGraph
        assert "cola" in result
        assert "powerGraph" in result

        # Cola should be a Layout instance
        assert hasattr(result["cola"], "nodes")
        assert hasattr(result["cola"], "start")

        # Nodes should have been positioned
        for node in nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_with_custom_size(self):
        """Test with custom canvas size."""
        nodes = [Node(x=0, y=0, width=10, height=10), Node(x=50, y=50, width=10, height=10)]

        links = [Link(source=0, target=1)]

        graph = {"nodes": nodes, "links": links}

        result = power_graph_grid_layout(graph, [200, 150], 5)

        assert "cola" in result
        assert "powerGraph" in result

    def test_with_padding(self):
        """Test with group padding."""
        nodes = [
            Node(x=0, y=0, width=10, height=10),
            Node(x=20, y=0, width=10, height=10),
            Node(x=0, y=20, width=10, height=10),
        ]

        links = [Link(source=0, target=1), Link(source=1, target=2), Link(source=2, target=0)]

        graph = {"nodes": nodes, "links": links}

        # Try with larger padding
        result = power_graph_grid_layout(graph, [100, 100], 20)

        assert "cola" in result
        assert "powerGraph" in result

    def test_result_structure(self):
        """Test that result has correct structure."""
        nodes = [Node(x=0, y=0, width=10, height=10), Node(x=10, y=10, width=10, height=10)]
        links = [Link(source=0, target=1)]
        graph = {"nodes": nodes, "links": links}

        result = power_graph_grid_layout(graph, [100, 100], 10)

        # Check cola
        assert result["cola"] is not None
        assert len(result["cola"].nodes()) == 2

        # Check powerGraph
        pg = result["powerGraph"]
        assert hasattr(pg, "groups")
        assert hasattr(pg, "powerEdges")

    def test_node_indices_assigned(self):
        """Test that node indices are properly assigned."""
        nodes = [
            Node(x=0, y=0, width=10, height=10),
            Node(x=10, y=0, width=10, height=10),
            Node(x=20, y=0, width=10, height=10),
        ]

        links = [Link(source=0, target=1), Link(source=1, target=2)]

        graph = {"nodes": nodes, "links": links}

        result = power_graph_grid_layout(graph, [100, 100], 10)

        # All nodes should have indices
        for i, node in enumerate(nodes):
            assert hasattr(node, "index")
            assert node.index == i

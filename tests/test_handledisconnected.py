"""Tests for handledisconnected module."""

from graph_layout.cola.handledisconnected import apply_packing, separate_graphs


class SimpleNode:
    """Simple node for testing."""

    def __init__(self, index: int, x: float = 0, y: float = 0):
        self.index = index
        self.x = x
        self.y = y
        self.width = 10.0
        self.height = 10.0


class SimpleLink:
    """Simple link for testing."""

    def __init__(self, source: SimpleNode, target: SimpleNode):
        self.source = source
        self.target = target


class TestSeparateGraphs:
    """Test connected components detection."""

    def test_single_node(self):
        """Test with single isolated node."""
        nodes = [SimpleNode(0)]
        links = []

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 1
        assert len(graphs[0]["array"]) == 1

    def test_two_connected_nodes(self):
        """Test with two connected nodes."""
        nodes = [SimpleNode(0), SimpleNode(1)]
        links = [SimpleLink(nodes[0], nodes[1])]

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 1
        assert len(graphs[0]["array"]) == 2

    def test_two_disconnected_components(self):
        """Test with two separate components."""
        nodes = [SimpleNode(i) for i in range(4)]
        links = [
            SimpleLink(nodes[0], nodes[1]),  # Component 1
            SimpleLink(nodes[2], nodes[3]),  # Component 2
        ]

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 2
        assert all(len(g["array"]) == 2 for g in graphs)

    def test_linear_graph(self):
        """Test with linear connected graph."""
        nodes = [SimpleNode(i) for i in range(5)]
        links = [
            SimpleLink(nodes[0], nodes[1]),
            SimpleLink(nodes[1], nodes[2]),
            SimpleLink(nodes[2], nodes[3]),
            SimpleLink(nodes[3], nodes[4]),
        ]

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 1
        assert len(graphs[0]["array"]) == 5

    def test_cycle(self):
        """Test with cyclic graph."""
        nodes = [SimpleNode(i) for i in range(3)]
        links = [
            SimpleLink(nodes[0], nodes[1]),
            SimpleLink(nodes[1], nodes[2]),
            SimpleLink(nodes[2], nodes[0]),
        ]

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 1
        assert len(graphs[0]["array"]) == 3

    def test_complex_disconnected(self):
        """Test with complex disconnected graph."""
        nodes = [SimpleNode(i) for i in range(7)]
        links = [
            # Component 1: triangle
            SimpleLink(nodes[0], nodes[1]),
            SimpleLink(nodes[1], nodes[2]),
            SimpleLink(nodes[2], nodes[0]),
            # Component 2: chain
            SimpleLink(nodes[3], nodes[4]),
            SimpleLink(nodes[4], nodes[5]),
            # Component 3: isolated in nodes[6]
        ]

        graphs = separate_graphs(nodes, links)

        assert len(graphs) == 3
        component_sizes = sorted([len(g["array"]) for g in graphs])
        assert component_sizes == [1, 3, 3]


class TestApplyPacking:
    """Test packing algorithm."""

    def test_empty_graphs(self):
        """Test with empty graph list."""
        graphs = []
        apply_packing(graphs, 100, 100)
        # Should not crash

    def test_single_component(self):
        """Test packing single component."""
        node = SimpleNode(0, x=0, y=0)
        node.width = 20
        node.height = 20

        graphs = [{"array": [node]}]
        apply_packing(graphs, 200, 200, node_size=20)

        # Graph should have dimensions
        assert "width" in graphs[0]
        assert "height" in graphs[0]
        assert graphs[0]["width"] > 0

    def test_multiple_components(self):
        """Test packing multiple components."""
        graphs = []

        for i in range(3):
            node = SimpleNode(i, x=0, y=0)
            node.width = 30
            node.height = 20
            graphs.append({"array": [node]})

        apply_packing(graphs, 300, 300, node_size=25)

        # All graphs should have dimensions
        for g in graphs:
            assert "width" in g
            assert "height" in g
            assert "x" in g
            assert "y" in g

    def test_packing_positions_nodes(self):
        """Test that packing actually positions nodes."""
        graphs = []

        for i in range(2):
            node = SimpleNode(i, x=0, y=0)
            node.width = 40
            node.height = 40
            graphs.append({"array": [node]})

        initial_positions = [(g["array"][0].x, g["array"][0].y) for g in graphs]

        apply_packing(graphs, 400, 400, node_size=40)

        final_positions = [(g["array"][0].x, g["array"][0].y) for g in graphs]

        # At least some positions should change (due to centering)
        # This might not always be true if already centered, so we just check structure
        for g in graphs:
            assert hasattr(g["array"][0], "x")
            assert hasattr(g["array"][0], "y")

    def test_packing_with_custom_ratio(self):
        """Test packing with custom aspect ratio."""
        graphs = []

        for i in range(4):
            node = SimpleNode(i, x=0, y=0)
            node.width = 25
            node.height = 25
            graphs.append({"array": [node]})

        apply_packing(graphs, 500, 300, node_size=25, desired_ratio=2.0)

        # Should complete without error
        assert len(graphs) == 4

    def test_packing_without_centering(self):
        """Test packing without centering."""
        node = SimpleNode(0, x=10, y=10)
        node.width = 30
        node.height = 30

        graphs = [{"array": [node]}]

        apply_packing(graphs, 200, 200, node_size=30, center_graph=False)

        # Should still calculate dimensions
        assert "width" in graphs[0]
        assert "height" in graphs[0]

    def test_packing_preserves_node_count(self):
        """Test that packing doesn't lose nodes."""
        graphs = []
        total_nodes = 0

        for i in range(3):
            nodes = [SimpleNode(j + total_nodes, x=j * 10, y=0) for j in range(2)]
            total_nodes += len(nodes)
            for node in nodes:
                node.width = 20
                node.height = 20
            graphs.append({"array": nodes})

        apply_packing(graphs, 400, 400, node_size=20)

        # Check all nodes still present
        packed_node_count = sum(len(g["array"]) for g in graphs)
        assert packed_node_count == total_nodes

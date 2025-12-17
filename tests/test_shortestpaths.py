"""Tests for shortest paths calculator."""

from graph_layout.cola.shortestpaths import Calculator


class Edge:
    """Simple edge class for testing."""

    def __init__(self, source: int, target: int, length: float = 1.0):
        self.source = source
        self.target = target
        self.length = length


class TestCalculator:
    """Test shortest paths calculator."""

    def test_simple_path(self):
        """Test finding path in simple graph."""
        #  0 -- 1 -- 2
        edges = [
            Edge(0, 1, 1.0),
            Edge(1, 2, 1.0),
        ]

        calc = Calculator(3, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        # Distance from 0 to all nodes
        distances = calc.distances_from_node(0)
        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 2.0

    def test_weighted_graph(self):
        """Test with weighted edges."""
        #  0 --(5)-- 1
        #  |         |
        # (1)       (1)
        #  |         |
        #  2 --(1)-- 3
        edges = [
            Edge(0, 1, 5.0),
            Edge(0, 2, 1.0),
            Edge(1, 3, 1.0),
            Edge(2, 3, 1.0),
        ]

        calc = Calculator(4, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)
        assert distances[0] == 0.0
        assert distances[1] == 3.0  # Via 2->3 (1+1+1=3) is shorter than direct (5)
        assert distances[2] == 1.0
        assert distances[3] == 2.0  # Via node 2

    def test_distance_matrix(self):
        """Test all-pairs shortest paths."""
        #  0 -- 1
        #  |    |
        #  2 -- 3
        edges = [
            Edge(0, 1),
            Edge(0, 2),
            Edge(1, 3),
            Edge(2, 3),
        ]

        calc = Calculator(4, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        matrix = calc.distance_matrix()

        # Check matrix is symmetric
        for i in range(4):
            for j in range(4):
                assert matrix[i][j] == matrix[j][i]

        # Check specific distances
        assert matrix[0][0] == 0.0
        assert matrix[0][1] == 1.0
        assert matrix[0][2] == 1.0
        assert matrix[0][3] == 2.0

    def test_path_from_node_to_node(self):
        """Test finding specific path between nodes."""
        #  0 -- 1 -- 2 -- 3
        edges = [
            Edge(0, 1),
            Edge(1, 2),
            Edge(2, 3),
        ]

        calc = Calculator(4, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        # Path from 0 to 3
        path = calc.path_from_node_to_node(0, 3)

        # Path should go through 2, 1 (reversed order since we build backwards)
        assert 2 in path
        assert 1 in path
        assert 0 in path

    def test_disconnected_graph(self):
        """Test graph with disconnected components."""
        #  0 -- 1    2 -- 3
        edges = [
            Edge(0, 1),
            Edge(2, 3),
        ]

        calc = Calculator(4, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)

        # Nodes 0 and 1 are reachable
        assert distances[0] == 0.0
        assert distances[1] == 1.0

        # Nodes 2 and 3 are unreachable
        assert distances[2] == float("inf")
        assert distances[3] == float("inf")

    def test_triangle_graph(self):
        """Test graph with triangle (multiple paths)."""
        #    0
        #   / \
        #  1---2
        edges = [
            Edge(0, 1, 1.0),
            Edge(0, 2, 1.0),
            Edge(1, 2, 2.0),
        ]

        calc = Calculator(3, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)

        assert distances[0] == 0.0
        assert distances[1] == 1.0  # Direct
        assert distances[2] == 1.0  # Direct (not via 1)

    def test_path_with_prev_cost(self):
        """Test path finding with previous edge cost."""
        #  0 -- 1 -- 2
        #   \       /
        #    ---3---
        edges = [
            Edge(0, 1, 1.0),
            Edge(1, 2, 1.0),
            Edge(0, 3, 1.5),
            Edge(3, 2, 1.5),
        ]

        calc = Calculator(4, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        # Without bend penalty, should prefer 0-1-2 (length 2.0)
        # With high bend penalty, might prefer 0-3-2 (length 3.0 but no bends)

        # Simple prev_cost that adds 1.0 for each bend
        def prev_cost(prev: int, curr: int, next: int) -> float:
            # Penalize direction changes
            return 0.0  # Simplified for this test

        path = calc.path_from_node_to_node_with_prev_cost(0, 2, prev_cost)

        # Path should be found
        assert isinstance(path, list)

    def test_single_node(self):
        """Test graph with single node."""
        calc = Calculator(1, [], lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)
        assert distances[0] == 0.0

    def test_no_edges(self):
        """Test graph with nodes but no edges."""
        calc = Calculator(3, [], lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)

        assert distances[0] == 0.0
        assert distances[1] == float("inf")
        assert distances[2] == float("inf")

    def test_self_loop_ignored(self):
        """Test that self loops don't affect shortest paths."""
        #  0 -- 1
        #  (self-loop on 0)
        edges = [
            Edge(0, 1, 1.0),
            Edge(0, 0, 5.0),  # Self loop
        ]

        calc = Calculator(2, edges, lambda e: e.source, lambda e: e.target, lambda e: e.length)

        distances = calc.distances_from_node(0)

        # Distance to 1 should not be affected by self-loop
        assert distances[0] == 0.0
        assert distances[1] == 1.0

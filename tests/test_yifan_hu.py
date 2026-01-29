"""
Tests for Yifan Hu Multilevel layout algorithm.
"""

import math

from graph_layout import Link, Node
from graph_layout.force import YifanHuLayout

# =============================================================================
# Test Fixtures
# =============================================================================


def create_simple_graph():
    """Create a simple graph with 3 nodes in a triangle."""
    nodes = [
        {"x": 0, "y": 0},
        {"x": 100, "y": 0},
        {"x": 50, "y": 100},
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 0},
    ]
    return nodes, links


def create_linear_graph(n=5):
    """Create a linear chain of n nodes."""
    nodes = [{"x": i * 10, "y": 0} for i in range(n)]
    links = [{"source": i, "target": i + 1} for i in range(n - 1)]
    return nodes, links


def create_star_graph(n=6):
    """Create a star graph with center node 0 and n-1 peripheral nodes."""
    nodes = [{"x": 0, "y": 0}]  # Center (hub)
    nodes.extend([{"x": i * 10, "y": i * 10} for i in range(1, n)])
    links = [{"source": 0, "target": i} for i in range(1, n)]
    return nodes, links


def create_medium_graph(n=50):
    """Create a medium-sized graph for testing multilevel coarsening."""
    nodes = [{"x": i * 10, "y": (i % 5) * 10} for i in range(n)]
    links = []
    # Create a connected graph with some structure
    for i in range(n - 1):
        links.append({"source": i, "target": i + 1})
    # Add some cross-links
    for i in range(0, n - 5, 5):
        links.append({"source": i, "target": i + 5})
    return nodes, links


def create_two_clusters():
    """Create two densely connected clusters with a weak bridge."""
    # Cluster 1: nodes 0-3
    nodes = [
        {"x": 0, "y": 0},
        {"x": 10, "y": 0},
        {"x": 0, "y": 10},
        {"x": 10, "y": 10},
    ]
    # Cluster 2: nodes 4-7
    nodes.extend(
        [
            {"x": 100, "y": 100},
            {"x": 110, "y": 100},
            {"x": 100, "y": 110},
            {"x": 110, "y": 110},
        ]
    )
    # Dense connections within clusters
    links = [
        # Cluster 1 (complete graph)
        {"source": 0, "target": 1},
        {"source": 0, "target": 2},
        {"source": 0, "target": 3},
        {"source": 1, "target": 2},
        {"source": 1, "target": 3},
        {"source": 2, "target": 3},
        # Cluster 2 (complete graph)
        {"source": 4, "target": 5},
        {"source": 4, "target": 6},
        {"source": 4, "target": 7},
        {"source": 5, "target": 6},
        {"source": 5, "target": 7},
        {"source": 6, "target": 7},
        # Single bridge between clusters
        {"source": 3, "target": 4},
    ]
    return nodes, links


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestYifanHuBasic:
    """Basic functionality tests for YifanHu layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=50,
        )
        layout.run()

        assert len(layout.nodes) == 3
        for node in layout.nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_nodes_move(self):
        """Test that nodes actually move during layout."""
        nodes, links = create_simple_graph()
        initial_positions = [(n["x"], n["y"]) for n in nodes]

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        moved = False
        for i, node in enumerate(layout.nodes):
            if (
                abs(node.x - initial_positions[i][0]) > 1
                or abs(node.y - initial_positions[i][1]) > 1
            ):
                moved = True
                break
        assert moved, "Nodes should move during layout"

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = YifanHuLayout(
            nodes=[],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 0

    def test_single_node(self):
        """Test layout with single node."""
        layout = YifanHuLayout(
            nodes=[{"x": 100, "y": 100}],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 1


# =============================================================================
# Configuration Tests
# =============================================================================


class TestYifanHuConfiguration:
    """Tests for YifanHu configuration properties."""

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = YifanHuLayout(
            optimal_distance=50.0,
            relative_strength=0.3,
            step_ratio=0.85,
            convergence_tolerance=0.02,
            use_barnes_hut=False,
            barnes_hut_theta=1.5,
            coarsening_threshold=0.8,
            min_coarsest_size=15,
            level_iterations=60,
            iterations=200,
        )

        assert layout.optimal_distance == 50.0
        assert layout.relative_strength == 0.3
        assert layout.step_ratio == 0.85
        assert layout.convergence_tolerance == 0.02
        assert layout.use_barnes_hut is False
        assert layout.barnes_hut_theta == 1.5
        assert layout.coarsening_threshold == 0.8
        assert layout.min_coarsest_size == 15
        assert layout.level_iterations == 60
        assert layout.iterations == 200

    def test_property_setters(self):
        """Test property setters work correctly."""
        layout = YifanHuLayout()

        layout.optimal_distance = 75.0
        assert layout.optimal_distance == 75.0

        layout.relative_strength = 0.5
        assert layout.relative_strength == 0.5

        layout.step_ratio = 0.8
        assert layout.step_ratio == 0.8

        layout.convergence_tolerance = 0.05
        assert layout.convergence_tolerance == 0.05

        layout.use_barnes_hut = False
        assert layout.use_barnes_hut is False

        layout.barnes_hut_theta = 2.0
        assert layout.barnes_hut_theta == 2.0

        layout.coarsening_threshold = 0.85
        assert layout.coarsening_threshold == 0.85

        layout.min_coarsest_size = 20
        assert layout.min_coarsest_size == 20

        layout.level_iterations = 80
        assert layout.level_iterations == 80

    def test_step_ratio_clamped(self):
        """Test step_ratio is clamped to [0.1, 0.99]."""
        layout = YifanHuLayout(step_ratio=2.0)
        assert layout.step_ratio == 0.99

        layout.step_ratio = 0.01
        assert layout.step_ratio == 0.1

    def test_coarsening_threshold_clamped(self):
        """Test coarsening_threshold is clamped to [0.5, 0.99]."""
        layout = YifanHuLayout(coarsening_threshold=1.5)
        assert layout.coarsening_threshold == 0.99

        layout.coarsening_threshold = 0.3
        assert layout.coarsening_threshold == 0.5


# =============================================================================
# Fixed Nodes Tests
# =============================================================================


class TestYifanHuFixedNodes:
    """Tests for fixed node behavior."""

    def test_fixed_nodes(self):
        """Test that fixed nodes don't move during layout."""
        nodes = [
            {"x": 100, "y": 100, "fixed": 1},
            {"x": 200, "y": 100},
            {"x": 150, "y": 200},
        ]
        links = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ]

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(400, 400),
            iterations=100,
        )
        layout.run(center_graph=False)

        # Fixed node should stay at its original position
        assert abs(layout.nodes[0].x - 100) < 0.1
        assert abs(layout.nodes[0].y - 100) < 0.1


# =============================================================================
# Event Tests
# =============================================================================


class TestYifanHuEvents:
    """Tests for event system."""

    def test_events_fired(self):
        """Test that events are fired during layout."""
        nodes, links = create_simple_graph()
        events = []

        def on_start(e):
            events.append(("start", e))

        def on_tick(e):
            events.append(("tick", e))

        def on_end(e):
            events.append(("end", e))

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
            on_start=on_start,
            on_tick=on_tick,
            on_end=on_end,
        )
        layout.run()

        start_events = [e for e in events if e[0] == "start"]
        tick_events = [e for e in events if e[0] == "tick"]
        end_events = [e for e in events if e[0] == "end"]

        assert len(start_events) == 1
        assert len(tick_events) > 0
        assert len(end_events) == 1

    def test_events_via_on_method(self):
        """Test that events work via the on() method."""
        nodes, links = create_simple_graph()
        events = []

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )

        layout.on("start", lambda e: events.append(("start", e)))
        layout.on("tick", lambda e: events.append(("tick", e)))
        layout.on("end", lambda e: events.append(("end", e)))

        layout.run()

        start_events = [e for e in events if e[0] == "start"]
        tick_events = [e for e in events if e[0] == "tick"]
        end_events = [e for e in events if e[0] == "end"]

        assert len(start_events) == 1
        assert len(tick_events) > 0
        assert len(end_events) == 1


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestYifanHuReproducibility:
    """Tests for reproducibility with random seed."""

    def test_random_seed(self):
        """Test that random seed produces deterministic results."""
        nodes, links = create_simple_graph()

        layout1 = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            random_seed=42,
            iterations=50,
        )
        layout1.run()
        pos1 = [(n.x, n.y) for n in layout1.nodes]

        # Create new graph with same seed
        nodes2 = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 50, "y": 100}]
        links2 = [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ]

        layout2 = YifanHuLayout(
            nodes=nodes2,
            links=links2,
            size=(500, 500),
            random_seed=42,
            iterations=50,
        )
        layout2.run()
        pos2 = [(n.x, n.y) for n in layout2.nodes]

        for p1, p2 in zip(pos1, pos2):
            assert abs(p1[0] - p2[0]) < 0.01
            assert abs(p1[1] - p2[1]) < 0.01


# =============================================================================
# Algorithm-Specific Tests
# =============================================================================


class TestYifanHuAlgorithm:
    """Tests for YifanHu-specific algorithm behavior."""

    def test_multilevel_coarsening(self):
        """Test that multilevel coarsening works for larger graphs."""
        nodes, links = create_medium_graph(50)

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            min_coarsest_size=5,
            coarsening_threshold=0.8,
            random_seed=42,
            iterations=100,
        )
        layout.run()

        # All nodes should have valid positions
        for node in layout.nodes:
            assert math.isfinite(node.x)
            assert math.isfinite(node.y)

    def test_adaptive_step_control(self):
        """Test that adaptive step control produces reasonable results."""
        nodes, links = create_simple_graph()

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            step_ratio=0.9,
            random_seed=42,
            iterations=100,
        )
        layout.run()

        # Check that nodes are positioned reasonably
        for node in layout.nodes:
            assert math.isfinite(node.x)
            assert math.isfinite(node.y)

    def test_barnes_hut_similar_to_naive(self):
        """Test that Barnes-Hut produces similar results to naive algorithm."""
        nodes, links = create_simple_graph()

        # Run with naive (no Barnes-Hut)
        layout_naive = YifanHuLayout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(500, 500),
            use_barnes_hut=False,
            random_seed=42,
            iterations=50,
        )
        layout_naive.run()
        naive_pos = [(n.x, n.y) for n in layout_naive.nodes]

        # Run with Barnes-Hut (but note: BH only activates for n > 50)
        # For small graphs, results should be identical
        layout_bh = YifanHuLayout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(500, 500),
            use_barnes_hut=True,
            random_seed=42,
            iterations=50,
        )
        layout_bh.run()
        bh_pos = [(n.x, n.y) for n in layout_bh.nodes]

        # For small graphs (< 50 nodes), both should use naive and be identical
        for p1, p2 in zip(naive_pos, bh_pos):
            assert abs(p1[0] - p2[0]) < 0.01
            assert abs(p1[1] - p2[1]) < 0.01

    def test_clusters_separate(self):
        """Test that well-defined clusters stay separated."""
        nodes, links = create_two_clusters()

        layout = YifanHuLayout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(500, 500),
            random_seed=42,
            iterations=100,
        )
        layout.run()

        # Calculate center of each cluster
        def cluster_center(layout, indices):
            cx = sum(layout.nodes[i].x for i in indices) / len(indices)
            cy = sum(layout.nodes[i].y for i in indices) / len(indices)
            return cx, cy

        c1 = cluster_center(layout, [0, 1, 2, 3])
        c2 = cluster_center(layout, [4, 5, 6, 7])

        # Clusters should be separated
        dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        assert dist > 10, f"Clusters should be separated (dist: {dist:.1f})"


# =============================================================================
# Node Object Support Tests
# =============================================================================


class TestYifanHuNodeObjects:
    """Tests for Node and Link object support."""

    def test_node_objects_supported(self):
        """Test that Node objects are supported directly."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=100)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )
        layout.run()

        assert len(layout.nodes) == 3


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================


class TestYifanHuCompatibility:
    """Tests for compatibility with other force layouts."""

    def test_pythonic_api(self):
        """Test that YifanHu supports the standard Pythonic API."""
        nodes, links = create_simple_graph()
        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )
        # Properties work
        assert len(layout.nodes) == 3
        assert len(layout.links) == 3
        assert layout.size == (500, 500)

    def test_linear_chain(self):
        """Test layout of linear chain."""
        nodes, links = create_linear_graph(5)
        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(800, 200),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 5

    def test_star_graph(self):
        """Test layout of star graph."""
        nodes, links = create_star_graph(6)
        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 6

    def test_medium_graph(self):
        """Test layout of medium-sized graph."""
        nodes, links = create_medium_graph(100)
        layout = YifanHuLayout(
            nodes=nodes,
            links=links,
            size=(800, 800),
            iterations=150,
            random_seed=42,
        )
        layout.run()

        assert len(layout.nodes) == 100
        # Verify all positions are valid
        for node in layout.nodes:
            assert math.isfinite(node.x)
            assert math.isfinite(node.y)

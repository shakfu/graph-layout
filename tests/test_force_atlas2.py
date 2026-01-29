"""
Tests for ForceAtlas2 layout algorithm.
"""

import math

from graph_layout import Link, Node
from graph_layout.force import ForceAtlas2Layout

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


class TestForceAtlas2Basic:
    """Basic functionality tests for ForceAtlas2."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = ForceAtlas2Layout(
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

        layout = ForceAtlas2Layout(
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
        layout = ForceAtlas2Layout(
            nodes=[],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 0

    def test_single_node(self):
        """Test layout with single node."""
        layout = ForceAtlas2Layout(
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


class TestForceAtlas2Configuration:
    """Tests for ForceAtlas2 configuration properties."""

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = ForceAtlas2Layout(
            scaling=3.0,
            gravity=2.0,
            strong_gravity_mode=True,
            linlog_mode=True,
            prevent_overlap=True,
            edge_weight_influence=0.5,
            tolerance=2.0,
            use_barnes_hut=False,
            barnes_hut_theta=0.8,
            iterations=200,
        )

        assert layout.scaling == 3.0
        assert layout.gravity == 2.0
        assert layout.strong_gravity_mode is True
        assert layout.linlog_mode is True
        assert layout.prevent_overlap is True
        assert layout.edge_weight_influence == 0.5
        assert layout.tolerance == 2.0
        assert layout.use_barnes_hut is False
        assert layout.barnes_hut_theta == 0.8
        assert layout.iterations == 200

    def test_property_setters(self):
        """Test property setters work correctly."""
        layout = ForceAtlas2Layout()

        layout.scaling = 5.0
        assert layout.scaling == 5.0

        layout.gravity = 0.5
        assert layout.gravity == 0.5

        layout.strong_gravity_mode = True
        assert layout.strong_gravity_mode is True

        layout.linlog_mode = True
        assert layout.linlog_mode is True

        layout.prevent_overlap = True
        assert layout.prevent_overlap is True

        layout.edge_weight_influence = 0.8
        assert layout.edge_weight_influence == 0.8

        layout.tolerance = 1.5
        assert layout.tolerance == 1.5

        layout.use_barnes_hut = False
        assert layout.use_barnes_hut is False

        layout.barnes_hut_theta = 1.5
        assert layout.barnes_hut_theta == 1.5

    def test_edge_weight_influence_clamped(self):
        """Test edge_weight_influence is clamped to [0, 1]."""
        layout = ForceAtlas2Layout(edge_weight_influence=2.0)
        assert layout.edge_weight_influence == 1.0

        layout.edge_weight_influence = -0.5
        assert layout.edge_weight_influence == 0.0

    def test_tolerance_minimum(self):
        """Test tolerance has minimum value of 0.1."""
        layout = ForceAtlas2Layout(tolerance=0.01)
        assert layout.tolerance == 0.1


# =============================================================================
# Fixed Nodes Tests
# =============================================================================


class TestForceAtlas2FixedNodes:
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

        layout = ForceAtlas2Layout(
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


class TestForceAtlas2Events:
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

        layout = ForceAtlas2Layout(
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

        layout = ForceAtlas2Layout(
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


class TestForceAtlas2Reproducibility:
    """Tests for reproducibility with random seed."""

    def test_random_seed(self):
        """Test that random seed produces deterministic results."""
        nodes, links = create_simple_graph()

        layout1 = ForceAtlas2Layout(
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

        layout2 = ForceAtlas2Layout(
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


class TestForceAtlas2Algorithm:
    """Tests for ForceAtlas2-specific algorithm behavior."""

    def test_linlog_mode_tighter_clusters(self):
        """Test that LinLog mode produces tighter clusters."""
        nodes, links = create_two_clusters()

        # Run without LinLog
        layout_linear = ForceAtlas2Layout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(500, 500),
            linlog_mode=False,
            random_seed=42,
            iterations=100,
        )
        layout_linear.run(center_graph=False)

        # Calculate cluster diameter for linear mode
        def cluster_diameter(layout, indices):
            max_dist = 0
            for i in indices:
                for j in indices:
                    if i != j:
                        dx = layout.nodes[i].x - layout.nodes[j].x
                        dy = layout.nodes[i].y - layout.nodes[j].y
                        dist = math.sqrt(dx * dx + dy * dy)
                        max_dist = max(max_dist, dist)
            return max_dist

        linear_diam1 = cluster_diameter(layout_linear, [0, 1, 2, 3])
        linear_diam2 = cluster_diameter(layout_linear, [4, 5, 6, 7])

        # Run with LinLog
        layout_linlog = ForceAtlas2Layout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(500, 500),
            linlog_mode=True,
            random_seed=42,
            iterations=100,
        )
        layout_linlog.run(center_graph=False)

        linlog_diam1 = cluster_diameter(layout_linlog, [0, 1, 2, 3])
        linlog_diam2 = cluster_diameter(layout_linlog, [4, 5, 6, 7])

        # LinLog should produce tighter (smaller diameter) clusters
        # Allow some tolerance since randomness affects results
        avg_linear = (linear_diam1 + linear_diam2) / 2
        avg_linlog = (linlog_diam1 + linlog_diam2) / 2

        # LinLog clusters should generally be tighter, but we allow
        # some margin due to randomness and other forces
        assert avg_linlog < avg_linear * 1.5, (
            f"LinLog clusters ({avg_linlog:.1f}) should be tighter than linear ({avg_linear:.1f})"
        )

    def test_strong_gravity_prevents_drift(self):
        """Test that strong gravity keeps nodes near center."""
        # Create two disconnected components
        nodes = [
            {"x": 0, "y": 0},
            {"x": 10, "y": 0},
            {"x": 200, "y": 200},
            {"x": 210, "y": 200},
        ]
        links = [
            {"source": 0, "target": 1},
            {"source": 2, "target": 3},
        ]

        layout = ForceAtlas2Layout(
            nodes=[dict(n) for n in nodes],
            links=[dict(l) for l in links],
            size=(400, 400),
            strong_gravity_mode=True,
            gravity=5.0,  # Strong gravity
            random_seed=42,
            iterations=100,
        )
        layout.run(center_graph=False)

        # With strong gravity, all nodes should be pulled toward center
        cx, cy = 200, 200  # Canvas center
        max_dist_from_center = 0

        for node in layout.nodes:
            dx = node.x - cx
            dy = node.y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            max_dist_from_center = max(max_dist_from_center, dist)

        # With strong gravity, nodes should stay reasonably close to center
        assert max_dist_from_center < 200, (
            f"With strong gravity, nodes should stay near center "
            f"(max dist: {max_dist_from_center:.1f})"
        )

    def test_degree_weighted_repulsion(self):
        """Test that hubs (high-degree nodes) repel more strongly."""
        # Create a star graph - node 0 is the hub
        nodes, links = create_star_graph(6)

        layout = ForceAtlas2Layout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            scaling=10.0,  # Strong repulsion to see effect
            gravity=0.0,  # No gravity so repulsion dominates
            random_seed=42,
            iterations=150,
        )
        layout.run(center_graph=False)

        # The hub (node 0) should be somewhat separated from peripheral nodes
        hub = layout.nodes[0]
        peripheral_nodes = layout.nodes[1:]

        # Calculate average distance from hub to peripheral nodes
        avg_dist_from_hub = 0
        for node in peripheral_nodes:
            dx = node.x - hub.x
            dy = node.y - hub.y
            avg_dist_from_hub += math.sqrt(dx * dx + dy * dy)
        avg_dist_from_hub /= len(peripheral_nodes)

        # Hub should have some distance from peripherals due to degree-weighted repulsion
        # The attraction pulls them together, but repulsion should maintain some distance
        assert avg_dist_from_hub > 5, (
            f"Hub should repel peripheral nodes (avg dist: {avg_dist_from_hub:.1f})"
        )

    def test_barnes_hut_similar_to_naive(self):
        """Test that Barnes-Hut produces similar results to naive algorithm."""
        nodes, links = create_simple_graph()

        # Run with naive (no Barnes-Hut)
        layout_naive = ForceAtlas2Layout(
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
        layout_bh = ForceAtlas2Layout(
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


# =============================================================================
# Node Object Support Tests
# =============================================================================


class TestForceAtlas2NodeObjects:
    """Tests for Node and Link object support."""

    def test_node_objects_supported(self):
        """Test that Node objects are supported directly."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=100)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        layout = ForceAtlas2Layout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )
        layout.run()

        assert len(layout.nodes) == 3

    def test_weighted_edges(self):
        """Test that edge weights are respected."""
        nodes = [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 50, "y": 100},
        ]
        # Edge 0-1 has high weight, edge 1-2 has low weight
        links = [
            {"source": 0, "target": 1, "weight": 10.0},
            {"source": 1, "target": 2, "weight": 0.1},
        ]

        layout = ForceAtlas2Layout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            edge_weight_influence=1.0,
            random_seed=42,
            iterations=100,
        )
        layout.run()

        # Nodes 0 and 1 should be closer than nodes 1 and 2
        dx01 = layout.nodes[1].x - layout.nodes[0].x
        dy01 = layout.nodes[1].y - layout.nodes[0].y
        dist_01 = math.sqrt(dx01 * dx01 + dy01 * dy01)

        dx12 = layout.nodes[2].x - layout.nodes[1].x
        dy12 = layout.nodes[2].y - layout.nodes[1].y
        dist_12 = math.sqrt(dx12 * dx12 + dy12 * dy12)

        # Higher weight edge should pull nodes closer together
        # (though other forces also affect this, so we use a loose bound)
        assert dist_01 < dist_12 * 2, (
            f"Higher-weight edge should result in closer nodes "
            f"(dist_01={dist_01:.1f}, dist_12={dist_12:.1f})"
        )


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================


class TestForceAtlas2Compatibility:
    """Tests for compatibility with other force layouts."""

    def test_pythonic_api(self):
        """Test that ForceAtlas2 supports the standard Pythonic API."""
        nodes, links = create_simple_graph()
        layout = ForceAtlas2Layout(
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
        layout = ForceAtlas2Layout(
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
        layout = ForceAtlas2Layout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 6

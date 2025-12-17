"""
Tests for force-directed layout algorithms.
"""

import math

import pytest

from graph_layout import Link, Node
from graph_layout.force import (
    FruchtermanReingoldLayout,
    KamadaKawaiLayout,
    SpringLayout,
)

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
    nodes = [{"x": 0, "y": 0}]  # Center
    nodes.extend([{"x": i * 10, "y": i * 10} for i in range(1, n)])
    links = [{"source": 0, "target": i} for i in range(1, n)]
    return nodes, links


def create_disconnected_graph():
    """Create two disconnected components."""
    nodes = [
        {"x": 0, "y": 0},
        {"x": 10, "y": 0},
        {"x": 100, "y": 100},
        {"x": 110, "y": 100},
    ]
    links = [
        {"source": 0, "target": 1},
        {"source": 2, "target": 3},
    ]
    return nodes, links


# =============================================================================
# Fruchterman-Reingold Tests
# =============================================================================


class TestFruchtermanReingoldLayout:
    """Tests for Fruchterman-Reingold algorithm."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = FruchtermanReingoldLayout(
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

        layout = FruchtermanReingoldLayout(
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

    def test_linear_graph(self):
        """Test layout of a linear chain."""
        nodes, links = create_linear_graph(5)
        layout = FruchtermanReingoldLayout(
            nodes=nodes,
            links=links,
            size=(800, 200),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 5

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = FruchtermanReingoldLayout(
            optimal_distance=50,
            temperature=100,
            cooling_factor=0.9,
            gravity=0.2,
            center_gravity=True,
            iterations=200,
        )

        assert layout.optimal_distance == 50
        assert layout.temperature == 100
        assert layout.cooling_factor == 0.9
        assert layout.gravity == 0.2
        assert layout.center_gravity is True
        assert layout.iterations == 200

        # Test property setters
        layout.temperature = 50
        assert layout.temperature == 50

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

        layout = FruchtermanReingoldLayout(
            nodes=nodes,
            links=links,
            size=(400, 400),
            iterations=100,
        )
        # Disable centering since we want to test fixed absolute positions
        layout.run(center_graph=False)

        # Fixed node should stay at its original position
        assert abs(layout.nodes[0].x - 100) < 0.1
        assert abs(layout.nodes[0].y - 100) < 0.1

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

        layout = FruchtermanReingoldLayout(
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

        layout = FruchtermanReingoldLayout(
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

    def test_random_seed(self):
        """Test that random seed produces deterministic results."""
        nodes, links = create_simple_graph()

        layout1 = FruchtermanReingoldLayout(
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

        layout2 = FruchtermanReingoldLayout(
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

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = FruchtermanReingoldLayout(
            nodes=[],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 0

    def test_single_node(self):
        """Test layout with single node."""
        layout = FruchtermanReingoldLayout(
            nodes=[{"x": 100, "y": 100}],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 1


# =============================================================================
# Spring Layout Tests
# =============================================================================


class TestSpringLayout:
    """Tests for Spring layout algorithm."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = SpringLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=50,
        )
        layout.run()

        assert len(layout.nodes) == 3

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = SpringLayout(
            spring_constant=0.2,
            spring_length=150,
            repulsion=20000,
            damping=0.6,
            gravity=0.1,
        )

        assert layout.spring_constant == 0.2
        assert layout.spring_length == 150
        assert layout.repulsion == 20000
        assert layout.damping == 0.6
        assert layout.gravity == 0.1

    def test_spring_length_respected(self):
        """Test that connected nodes tend toward spring length."""
        nodes = [{"x": 0, "y": 0}, {"x": 10, "y": 0}]
        links = [{"source": 0, "target": 1}]

        layout = SpringLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            spring_length=100,
            iterations=200,
        )
        layout.run()

        dx = layout.nodes[1].x - layout.nodes[0].x
        dy = layout.nodes[1].y - layout.nodes[0].y
        dist = math.sqrt(dx * dx + dy * dy)

        # Distance should be reasonably close to spring length
        assert abs(dist - 100) < 50, f"Distance {dist} not close to spring length 100"

    def test_linear_chain(self):
        """Test layout of linear chain."""
        nodes, links = create_linear_graph(4)
        layout = SpringLayout(
            nodes=nodes,
            links=links,
            size=(800, 200),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 4

    def test_star_graph(self):
        """Test layout of star graph."""
        nodes, links = create_star_graph(6)
        layout = SpringLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 6


# =============================================================================
# Kamada-Kawai Tests
# =============================================================================


class TestKamadaKawaiLayout:
    """Tests for Kamada-Kawai algorithm."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = KamadaKawaiLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 3

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = KamadaKawaiLayout(
            edge_length=80,
            epsilon=0.001,
            disconnected_distance=500,
        )

        assert layout.edge_length == 80
        assert layout.epsilon == 0.001
        assert layout.disconnected_distance == 500

    def test_graph_theoretic_distance(self):
        """Test that layout respects graph-theoretic distances."""
        # Create a path graph: 0 -- 1 -- 2
        nodes = [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 100, "y": 0}]
        links = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        layout = KamadaKawaiLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            edge_length=100,
            iterations=200,
        )
        layout.run()

        # Distance 0-1 should be ~100 (1 hop)
        dx01 = layout.nodes[1].x - layout.nodes[0].x
        dy01 = layout.nodes[1].y - layout.nodes[0].y
        dist_01 = math.sqrt(dx01 * dx01 + dy01 * dy01)

        # Distance 0-2 should be ~200 (2 hops)
        dx02 = layout.nodes[2].x - layout.nodes[0].x
        dy02 = layout.nodes[2].y - layout.nodes[0].y
        dist_02 = math.sqrt(dx02 * dx02 + dy02 * dy02)

        # The ratio of distances should be approximately 2:1
        ratio = dist_02 / dist_01 if dist_01 > 0 else 0
        assert 1.5 < ratio < 2.5, f"Distance ratio {ratio} not close to 2:1"

    def test_disconnected_components(self):
        """Test layout with disconnected components."""
        nodes, links = create_disconnected_graph()
        layout = KamadaKawaiLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=100,
        )
        layout.run()

        assert len(layout.nodes) == 4

        # Components should be separated
        comp1_center = (
            (layout.nodes[0].x + layout.nodes[1].x) / 2,
            (layout.nodes[0].y + layout.nodes[1].y) / 2,
        )
        comp2_center = (
            (layout.nodes[2].x + layout.nodes[3].x) / 2,
            (layout.nodes[2].y + layout.nodes[3].y) / 2,
        )

        dx = comp2_center[0] - comp1_center[0]
        dy = comp2_center[1] - comp1_center[1]
        center_dist = math.sqrt(dx * dx + dy * dy)

        assert center_dist > 50, "Disconnected components should be separated"

    def test_single_node(self):
        """Test layout with single node."""
        layout = KamadaKawaiLayout(
            nodes=[{"x": 100, "y": 100}],
            links=[],
            size=(500, 500),
            iterations=10,
        )
        layout.run()
        assert len(layout.nodes) == 1

    def test_two_connected_nodes(self):
        """Test layout with two connected nodes."""
        nodes = [{"x": 0, "y": 0}, {"x": 10, "y": 0}]
        links = [{"source": 0, "target": 1}]

        layout = KamadaKawaiLayout(
            nodes=nodes,
            links=links,
            size=(500, 500),
            edge_length=100,
            iterations=100,
        )
        layout.run()

        dx = layout.nodes[1].x - layout.nodes[0].x
        dy = layout.nodes[1].y - layout.nodes[0].y
        dist = math.sqrt(dx * dx + dy * dy)

        assert abs(dist - 100) < 30, f"Distance {dist} not close to edge_length 100"


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================


class TestAllForceLayouts:
    """Tests that apply to all force-directed algorithms."""

    @pytest.mark.parametrize(
        "LayoutClass",
        [
            FruchtermanReingoldLayout,
            SpringLayout,
            KamadaKawaiLayout,
        ],
    )
    def test_empty_graph(self, LayoutClass):
        """All algorithms should handle empty graphs."""
        layout = LayoutClass(nodes=[], links=[], size=(500, 500))
        layout.run()
        assert len(layout.nodes) == 0

    @pytest.mark.parametrize(
        "LayoutClass",
        [
            FruchtermanReingoldLayout,
            SpringLayout,
            KamadaKawaiLayout,
        ],
    )
    def test_single_node(self, LayoutClass):
        """All algorithms should handle single node."""
        layout = LayoutClass(
            nodes=[{"x": 100, "y": 100}],
            links=[],
            size=(500, 500),
        )
        layout.run()
        assert len(layout.nodes) == 1

    @pytest.mark.parametrize(
        "LayoutClass",
        [
            FruchtermanReingoldLayout,
            SpringLayout,
            KamadaKawaiLayout,
        ],
    )
    def test_pythonic_api(self, LayoutClass):
        """All algorithms support Pythonic API."""
        nodes, links = create_simple_graph()
        layout = LayoutClass(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )
        # Properties work
        assert len(layout.nodes) == 3
        assert len(layout.links) == 3
        assert layout.size == (500, 500)

    @pytest.mark.parametrize(
        "LayoutClass",
        [
            FruchtermanReingoldLayout,
            SpringLayout,
            KamadaKawaiLayout,
        ],
    )
    def test_node_objects_supported(self, LayoutClass):
        """All algorithms support Node objects directly."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=100)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        layout = LayoutClass(
            nodes=nodes,
            links=links,
            size=(500, 500),
            iterations=10,
        )
        layout.run()

        assert len(layout.nodes) == 3

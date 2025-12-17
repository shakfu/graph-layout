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
        {'x': 0, 'y': 0},
        {'x': 100, 'y': 0},
        {'x': 50, 'y': 100},
    ]
    links = [
        {'source': 0, 'target': 1},
        {'source': 1, 'target': 2},
        {'source': 2, 'target': 0},
    ]
    return nodes, links


def create_linear_graph(n=5):
    """Create a linear chain of n nodes."""
    nodes = [{'x': i * 10, 'y': 0} for i in range(n)]
    links = [{'source': i, 'target': i + 1} for i in range(n - 1)]
    return nodes, links


def create_star_graph(n=6):
    """Create a star graph with center node 0 and n-1 peripheral nodes."""
    nodes = [{'x': 0, 'y': 0}]  # Center
    nodes.extend([{'x': i * 10, 'y': i * 10} for i in range(1, n)])
    links = [{'source': 0, 'target': i} for i in range(1, n)]
    return nodes, links


def create_disconnected_graph():
    """Create two disconnected components."""
    nodes = [
        {'x': 0, 'y': 0},
        {'x': 10, 'y': 0},
        {'x': 100, 'y': 100},
        {'x': 110, 'y': 100},
    ]
    links = [
        {'source': 0, 'target': 1},
        {'source': 2, 'target': 3},
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
        layout = FruchtermanReingoldLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=50)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 3
        for node in result_nodes:
            assert hasattr(node, 'x')
            assert hasattr(node, 'y')

    def test_nodes_move(self):
        """Test that nodes actually move during layout."""
        nodes, links = create_simple_graph()
        layout = FruchtermanReingoldLayout()
        layout.nodes(nodes).links(links).size([500, 500])

        initial_positions = [(n['x'], n['y']) for n in nodes]
        layout.start(iterations=100)
        result_nodes = layout.nodes()

        moved = False
        for i, node in enumerate(result_nodes):
            if abs(node.x - initial_positions[i][0]) > 1 or \
               abs(node.y - initial_positions[i][1]) > 1:
                moved = True
                break
        assert moved, "Nodes should move during layout"

    def test_linear_graph(self):
        """Test layout of a linear chain."""
        nodes, links = create_linear_graph(5)
        layout = FruchtermanReingoldLayout()
        layout.nodes(nodes).links(links).size([800, 200])
        layout.start(iterations=100)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 5

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = FruchtermanReingoldLayout()

        # Test chaining
        result = (layout
            .optimal_distance(50)
            .temperature(100)
            .cooling_factor(0.9)
            .gravity(0.2)
            .center_gravity(True)
            .iterations(200))

        assert result is layout
        assert layout.optimal_distance() == 50
        assert layout.temperature() == 100
        assert layout.cooling_factor() == 0.9
        assert layout.gravity() == 0.2
        assert layout.center_gravity() is True
        assert layout.iterations() == 200

    def test_fixed_nodes(self):
        """Test that fixed nodes don't move during layout."""
        nodes = [
            {'x': 100, 'y': 100, 'fixed': 1},
            {'x': 200, 'y': 100},
            {'x': 150, 'y': 200},
        ]
        links = [
            {'source': 0, 'target': 1},
            {'source': 1, 'target': 2},
        ]

        layout = FruchtermanReingoldLayout()
        layout.nodes(nodes).links(links).size([400, 400])
        # Disable centering since we want to test fixed absolute positions
        layout.start(iterations=100, center_graph=False)

        result_nodes = layout.nodes()
        # Fixed node should stay at its original position
        assert abs(result_nodes[0].x - 100) < 0.1
        assert abs(result_nodes[0].y - 100) < 0.1

    def test_events_fired(self):
        """Test that events are fired during layout."""
        nodes, links = create_simple_graph()
        layout = FruchtermanReingoldLayout()
        layout.nodes(nodes).links(links).size([500, 500])

        events = []

        def on_start(e):
            events.append(('start', e))

        def on_tick(e):
            events.append(('tick', e))

        def on_end(e):
            events.append(('end', e))

        layout.on('start', on_start)
        layout.on('tick', on_tick)
        layout.on('end', on_end)

        layout.start(iterations=10)

        start_events = [e for e in events if e[0] == 'start']
        tick_events = [e for e in events if e[0] == 'tick']
        end_events = [e for e in events if e[0] == 'end']

        assert len(start_events) == 1
        assert len(tick_events) > 0
        assert len(end_events) == 1

    def test_random_seed(self):
        """Test that random seed produces deterministic results."""
        nodes, links = create_simple_graph()

        layout1 = FruchtermanReingoldLayout()
        layout1.nodes(nodes).links(links).size([500, 500]).random_seed(42)
        layout1.start(iterations=50)
        pos1 = [(n.x, n.y) for n in layout1.nodes()]

        # Create new graph with same seed
        nodes2 = [{'x': 0, 'y': 0}, {'x': 100, 'y': 0}, {'x': 50, 'y': 100}]
        links2 = [
            {'source': 0, 'target': 1},
            {'source': 1, 'target': 2},
            {'source': 2, 'target': 0}
        ]

        layout2 = FruchtermanReingoldLayout()
        layout2.nodes(nodes2).links(links2).size([500, 500]).random_seed(42)
        layout2.start(iterations=50)
        pos2 = [(n.x, n.y) for n in layout2.nodes()]

        for p1, p2 in zip(pos1, pos2):
            assert abs(p1[0] - p2[0]) < 0.01
            assert abs(p1[1] - p2[1]) < 0.01

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = FruchtermanReingoldLayout()
        layout.nodes([]).links([]).size([500, 500])
        layout.start(iterations=10)
        assert len(layout.nodes()) == 0

    def test_single_node(self):
        """Test layout with single node."""
        layout = FruchtermanReingoldLayout()
        layout.nodes([{'x': 100, 'y': 100}]).links([]).size([500, 500])
        layout.start(iterations=10)
        assert len(layout.nodes()) == 1


# =============================================================================
# Spring Layout Tests
# =============================================================================

class TestSpringLayout:
    """Tests for Spring layout algorithm."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = SpringLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=50)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 3

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = SpringLayout()

        result = (layout
            .spring_constant(0.2)
            .spring_length(150)
            .repulsion(20000)
            .damping(0.6)
            .gravity(0.1))

        assert result is layout
        assert layout.spring_constant() == 0.2
        assert layout.spring_length() == 150
        assert layout.repulsion() == 20000
        assert layout.damping() == 0.6
        assert layout.gravity() == 0.1

    def test_spring_length_respected(self):
        """Test that connected nodes tend toward spring length."""
        nodes = [{'x': 0, 'y': 0}, {'x': 10, 'y': 0}]
        links = [{'source': 0, 'target': 1}]

        layout = SpringLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.spring_length(100).iterations(200)
        layout.start()

        result_nodes = layout.nodes()
        dx = result_nodes[1].x - result_nodes[0].x
        dy = result_nodes[1].y - result_nodes[0].y
        dist = math.sqrt(dx * dx + dy * dy)

        # Distance should be reasonably close to spring length
        assert abs(dist - 100) < 50, f"Distance {dist} not close to spring length 100"

    def test_linear_chain(self):
        """Test layout of linear chain."""
        nodes, links = create_linear_graph(4)
        layout = SpringLayout()
        layout.nodes(nodes).links(links).size([800, 200])
        layout.start(iterations=100)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 4

    def test_star_graph(self):
        """Test layout of star graph."""
        nodes, links = create_star_graph(6)
        layout = SpringLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=100)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 6


# =============================================================================
# Kamada-Kawai Tests
# =============================================================================

class TestKamadaKawaiLayout:
    """Tests for Kamada-Kawai algorithm."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = KamadaKawaiLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=100)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 3

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = KamadaKawaiLayout()

        result = (layout
            .edge_length(80)
            .epsilon(0.001)
            .disconnected_distance(500))

        assert result is layout
        assert layout.edge_length() == 80
        assert layout.epsilon() == 0.001
        assert layout.disconnected_distance() == 500

    def test_graph_theoretic_distance(self):
        """Test that layout respects graph-theoretic distances."""
        # Create a path graph: 0 -- 1 -- 2
        nodes = [{'x': 0, 'y': 0}, {'x': 50, 'y': 0}, {'x': 100, 'y': 0}]
        links = [{'source': 0, 'target': 1}, {'source': 1, 'target': 2}]

        layout = KamadaKawaiLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.edge_length(100).start(iterations=200)

        result_nodes = layout.nodes()

        # Distance 0-1 should be ~100 (1 hop)
        dx01 = result_nodes[1].x - result_nodes[0].x
        dy01 = result_nodes[1].y - result_nodes[0].y
        dist_01 = math.sqrt(dx01 * dx01 + dy01 * dy01)

        # Distance 0-2 should be ~200 (2 hops)
        dx02 = result_nodes[2].x - result_nodes[0].x
        dy02 = result_nodes[2].y - result_nodes[0].y
        dist_02 = math.sqrt(dx02 * dx02 + dy02 * dy02)

        # The ratio of distances should be approximately 2:1
        ratio = dist_02 / dist_01 if dist_01 > 0 else 0
        assert 1.5 < ratio < 2.5, f"Distance ratio {ratio} not close to 2:1"

    def test_disconnected_components(self):
        """Test layout with disconnected components."""
        nodes, links = create_disconnected_graph()
        layout = KamadaKawaiLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=100)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 4

        # Components should be separated
        comp1_center = (
            (result_nodes[0].x + result_nodes[1].x) / 2,
            (result_nodes[0].y + result_nodes[1].y) / 2
        )
        comp2_center = (
            (result_nodes[2].x + result_nodes[3].x) / 2,
            (result_nodes[2].y + result_nodes[3].y) / 2
        )

        dx = comp2_center[0] - comp1_center[0]
        dy = comp2_center[1] - comp1_center[1]
        center_dist = math.sqrt(dx * dx + dy * dy)

        assert center_dist > 50, "Disconnected components should be separated"

    def test_single_node(self):
        """Test layout with single node."""
        layout = KamadaKawaiLayout()
        layout.nodes([{'x': 100, 'y': 100}]).links([]).size([500, 500])
        layout.start(iterations=10)
        assert len(layout.nodes()) == 1

    def test_two_connected_nodes(self):
        """Test layout with two connected nodes."""
        nodes = [{'x': 0, 'y': 0}, {'x': 10, 'y': 0}]
        links = [{'source': 0, 'target': 1}]

        layout = KamadaKawaiLayout()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.edge_length(100).start(iterations=100)

        result_nodes = layout.nodes()
        dx = result_nodes[1].x - result_nodes[0].x
        dy = result_nodes[1].y - result_nodes[0].y
        dist = math.sqrt(dx * dx + dy * dy)

        assert abs(dist - 100) < 30, f"Distance {dist} not close to edge_length 100"


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================

class TestAllForceLayouts:
    """Tests that apply to all force-directed algorithms."""

    @pytest.mark.parametrize("LayoutClass", [
        FruchtermanReingoldLayout,
        SpringLayout,
        KamadaKawaiLayout,
    ])
    def test_empty_graph(self, LayoutClass):
        """All algorithms should handle empty graphs."""
        layout = LayoutClass()
        layout.nodes([]).links([]).size([500, 500])
        layout.start()
        assert len(layout.nodes()) == 0

    @pytest.mark.parametrize("LayoutClass", [
        FruchtermanReingoldLayout,
        SpringLayout,
        KamadaKawaiLayout,
    ])
    def test_single_node(self, LayoutClass):
        """All algorithms should handle single node."""
        layout = LayoutClass()
        layout.nodes([{'x': 100, 'y': 100}]).links([]).size([500, 500])
        layout.start()
        assert len(layout.nodes()) == 1

    @pytest.mark.parametrize("LayoutClass", [
        FruchtermanReingoldLayout,
        SpringLayout,
        KamadaKawaiLayout,
    ])
    def test_fluent_api(self, LayoutClass):
        """All algorithms support fluent API."""
        nodes, links = create_simple_graph()
        layout = LayoutClass()
        result = (layout
            .nodes(nodes)
            .links(links)
            .size([500, 500])
            .iterations(10))
        assert result is layout

    @pytest.mark.parametrize("LayoutClass", [
        FruchtermanReingoldLayout,
        SpringLayout,
        KamadaKawaiLayout,
    ])
    def test_node_objects_supported(self, LayoutClass):
        """All algorithms support Node objects directly."""
        nodes = [Node(x=0, y=0), Node(x=100, y=0), Node(x=50, y=100)]
        links = [Link(0, 1), Link(1, 2), Link(2, 0)]

        layout = LayoutClass()
        layout.nodes(nodes).links(links).size([500, 500])
        layout.start(iterations=10)

        result_nodes = layout.nodes()
        assert len(result_nodes) == 3

"""
Tests for circular layout algorithms.
"""

import math

import pytest

from graph_layout.circular import CircularLayout, ShellLayout

# =============================================================================
# Test Fixtures
# =============================================================================


def create_simple_graph():
    """Create a simple graph with 5 nodes."""
    nodes = [{} for _ in range(5)]
    links = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 0},
    ]
    return nodes, links


def create_star_graph():
    """Create a star graph with center at 0."""
    nodes = [{} for _ in range(6)]
    links = [
        {"source": 0, "target": 1},
        {"source": 0, "target": 2},
        {"source": 0, "target": 3},
        {"source": 0, "target": 4},
        {"source": 0, "target": 5},
    ]
    return nodes, links


# =============================================================================
# Circular Layout Tests
# =============================================================================


class TestCircularLayout:
    """Tests for Circular layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = CircularLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 5

    def test_nodes_on_circle(self):
        """Test that all nodes are on a circle."""
        nodes, links = create_simple_graph()
        layout = CircularLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run(center_graph=False)

        cx, cy = 400, 300

        # All nodes should be at same distance from center
        distances = []
        for node in layout.nodes:
            dist = math.sqrt((node.x - cx) ** 2 + (node.y - cy) ** 2)
            distances.append(dist)

        # Check all distances are equal (within tolerance)
        for dist in distances:
            assert abs(dist - distances[0]) < 1.0

    def test_even_spacing(self):
        """Test that nodes are evenly spaced."""
        nodes = [{} for _ in range(4)]
        layout = CircularLayout(
            nodes=nodes,
            links=[],
            size=(800, 600),
        )
        layout.run(center_graph=False)

        cx, cy = 400, 300

        # Calculate angles
        angles = []
        for node in layout.nodes:
            dx = node.x - cx
            dy = node.y - cy
            angle = math.atan2(dy, dx)
            angles.append(angle)

        # Sort and compute differences
        angles.sort()
        expected_diff = 2 * math.pi / 4  # 90 degrees for 4 nodes

        for i in range(len(angles)):
            next_idx = (i + 1) % len(angles)
            diff = angles[next_idx] - angles[i]
            if diff < 0:
                diff += 2 * math.pi
            assert abs(diff - expected_diff) < 0.1

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = CircularLayout(
            radius=200,
            start_angle=math.pi / 4,
            sort_by="degree",
        )

        assert layout.radius == 200
        assert layout.start_angle == math.pi / 4
        assert layout.sort_by == "degree"

    def test_sort_by_degree(self):
        """Test sorting by degree."""
        nodes, links = create_star_graph()
        layout = CircularLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            sort_by="degree",
        )
        layout.run()

        # Node 0 has highest degree (5), should be positioned
        assert len(layout.nodes) == 6

    def test_custom_sort_function(self):
        """Test custom sort function."""
        nodes = [{"priority": i} for i in range(5)]
        layout = CircularLayout(
            nodes=nodes,
            links=[],
            size=(800, 600),
            sort_by=lambda n: -getattr(n, "priority", 0),
        )
        layout.run()

        assert len(layout.nodes) == 5

    def test_single_node(self):
        """Test layout with single node."""
        layout = CircularLayout(
            nodes=[{}],
            links=[],
            size=(800, 600),
        )
        layout.run()
        assert len(layout.nodes) == 1

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = CircularLayout(
            nodes=[],
            links=[],
            size=(800, 600),
        )
        layout.run()
        assert len(layout.nodes) == 0


# =============================================================================
# Shell Layout Tests
# =============================================================================


class TestShellLayout:
    """Tests for Shell layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_simple_graph()
        layout = ShellLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        layout.run()

        assert len(layout.nodes) == 5

    def test_explicit_shells(self):
        """Test explicit shell assignment."""
        nodes = [{} for _ in range(6)]
        layout = ShellLayout(
            nodes=nodes,
            links=[],
            size=(800, 600),
            # Center node, inner ring, outer ring
            shells=[[0], [1, 2], [3, 4, 5]],
        )
        layout.run(center_graph=False)

        cx, cy = 400, 300

        # Calculate distances from center
        dist_0 = math.sqrt((layout.nodes[0].x - cx) ** 2 + (layout.nodes[0].y - cy) ** 2)
        dist_1 = math.sqrt((layout.nodes[1].x - cx) ** 2 + (layout.nodes[1].y - cy) ** 2)
        dist_3 = math.sqrt((layout.nodes[3].x - cx) ** 2 + (layout.nodes[3].y - cy) ** 2)

        # Shell 0 (node 0) should be at center
        assert dist_0 < 1.0

        # Shell 1 should be closer than shell 2
        assert dist_1 < dist_3

    def test_auto_shells_by_degree(self):
        """Test automatic shell grouping by degree."""
        nodes, links = create_star_graph()
        layout = ShellLayout(
            nodes=nodes,
            links=links,
            size=(800, 600),
            auto_shells=2,
        )
        layout.run(center_graph=False)

        cx, cy = 400, 300

        # Node 0 (high degree) should be in inner shell
        dist_0 = math.sqrt((layout.nodes[0].x - cx) ** 2 + (layout.nodes[0].y - cy) ** 2)
        dist_1 = math.sqrt((layout.nodes[1].x - cx) ** 2 + (layout.nodes[1].y - cy) ** 2)

        # High degree node should be closer to center
        assert dist_0 < dist_1

    def test_configuration_properties(self):
        """Test configuration via constructor and properties."""
        layout = ShellLayout(
            shells=[[0], [1, 2, 3]],
            radius_step=80,
            start_angle=math.pi / 6,
        )

        assert layout.shells == [[0], [1, 2, 3]]
        assert layout.radius_step == 80
        assert layout.start_angle == math.pi / 6

    def test_nodes_on_concentric_circles(self):
        """Test that nodes in same shell are on same circle."""
        nodes = [{} for _ in range(6)]
        layout = ShellLayout(
            nodes=nodes,
            links=[],
            size=(800, 600),
            shells=[[0, 1], [2, 3], [4, 5]],
        )
        layout.run(center_graph=False)

        cx, cy = 400, 300

        # Nodes in same shell should have same distance from center
        for shell in [[0, 1], [2, 3], [4, 5]]:
            distances = [
                math.sqrt((layout.nodes[i].x - cx) ** 2 + (layout.nodes[i].y - cy) ** 2)
                for i in shell
            ]
            assert abs(distances[0] - distances[1]) < 1.0

    def test_single_node(self):
        """Test layout with single node."""
        layout = ShellLayout(
            nodes=[{}],
            links=[],
            size=(800, 600),
        )
        layout.run()
        assert len(layout.nodes) == 1

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = ShellLayout(
            nodes=[],
            links=[],
            size=(800, 600),
        )
        layout.run()
        assert len(layout.nodes) == 0


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================


class TestAllCircularLayouts:
    """Tests that apply to all circular algorithms."""

    @pytest.mark.parametrize("LayoutClass", [CircularLayout, ShellLayout])
    def test_empty_graph(self, LayoutClass):
        """All algorithms should handle empty graphs."""
        layout = LayoutClass(nodes=[], links=[], size=(800, 600))
        layout.run()
        assert len(layout.nodes) == 0

    @pytest.mark.parametrize("LayoutClass", [CircularLayout, ShellLayout])
    def test_single_node(self, LayoutClass):
        """All algorithms should handle single node."""
        layout = LayoutClass(nodes=[{}], links=[], size=(800, 600))
        layout.run()
        assert len(layout.nodes) == 1

    @pytest.mark.parametrize("LayoutClass", [CircularLayout, ShellLayout])
    def test_pythonic_api(self, LayoutClass):
        """All algorithms support Pythonic API."""
        nodes, links = create_simple_graph()
        layout = LayoutClass(
            nodes=nodes,
            links=links,
            size=(800, 600),
        )
        # Properties work
        assert len(layout.nodes) == 5
        assert len(layout.links) == 5
        assert layout.size == (800, 600)

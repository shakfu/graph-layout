"""
Tests for hierarchical layout algorithms.
"""

import math

import pytest

from graph_layout.hierarchical import (
    RadialTreeLayout,
    ReingoldTilfordLayout,
    SugiyamaLayout,
)

# =============================================================================
# Test Fixtures
# =============================================================================

def create_binary_tree():
    """Create a binary tree with 7 nodes."""
    #        0
    #       / \
    #      1   2
    #     / \ / \
    #    3  4 5  6
    nodes = [{} for _ in range(7)]
    links = [
        {'source': 0, 'target': 1},
        {'source': 0, 'target': 2},
        {'source': 1, 'target': 3},
        {'source': 1, 'target': 4},
        {'source': 2, 'target': 5},
        {'source': 2, 'target': 6},
    ]
    return nodes, links


def create_linear_tree():
    """Create a linear tree (linked list)."""
    nodes = [{} for _ in range(4)]
    links = [
        {'source': 0, 'target': 1},
        {'source': 1, 'target': 2},
        {'source': 2, 'target': 3},
    ]
    return nodes, links


def create_wide_tree():
    """Create a wide tree with one parent and many children."""
    nodes = [{} for _ in range(6)]
    links = [
        {'source': 0, 'target': 1},
        {'source': 0, 'target': 2},
        {'source': 0, 'target': 3},
        {'source': 0, 'target': 4},
        {'source': 0, 'target': 5},
    ]
    return nodes, links


def create_dag():
    """Create a simple DAG (not a tree)."""
    #     0
    #    / \
    #   1   2
    #    \ /
    #     3
    #     |
    #     4
    nodes = [{} for _ in range(5)]
    links = [
        {'source': 0, 'target': 1},
        {'source': 0, 'target': 2},
        {'source': 1, 'target': 3},
        {'source': 2, 'target': 3},
        {'source': 3, 'target': 4},
    ]
    return nodes, links


# =============================================================================
# Reingold-Tilford Tests
# =============================================================================

class TestReingoldTilfordLayout:
    """Tests for Reingold-Tilford tree layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_binary_tree()
        layout = ReingoldTilfordLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start()

        result_nodes = layout.nodes()
        assert len(result_nodes) == 7
        for node in result_nodes:
            assert hasattr(node, 'x')
            assert hasattr(node, 'y')

    def test_root_at_top(self):
        """Test that root is at top in top-to-bottom orientation."""
        nodes, links = create_binary_tree()
        layout = ReingoldTilfordLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.orientation('top-to-bottom').start()

        result_nodes = layout.nodes()
        root = result_nodes[0]

        # Root should be above all other nodes
        for i, node in enumerate(result_nodes[1:], 1):
            assert root.y < node.y, f"Root should be above node {i}"

    def test_parent_centered_over_children(self):
        """Test that parent is horizontally centered over children."""
        nodes, links = create_binary_tree()
        layout = ReingoldTilfordLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.orientation('top-to-bottom').start()

        result_nodes = layout.nodes()
        root = result_nodes[0]
        left_child = result_nodes[1]
        right_child = result_nodes[2]

        # Root should be centered between children
        children_center = (left_child.x + right_child.x) / 2
        assert abs(root.x - children_center) < 1.0

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = ReingoldTilfordLayout()

        result = (layout
            .root(0)
            .node_separation(2.0)
            .level_separation(1.5)
            .orientation('left-to-right'))

        assert result is layout
        assert layout.root() == 0
        assert layout.node_separation() == 2.0
        assert layout.level_separation() == 1.5
        assert layout.orientation() == 'left-to-right'

    def test_linear_tree(self):
        """Test layout of linear tree."""
        nodes, links = create_linear_tree()
        layout = ReingoldTilfordLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.orientation('top-to-bottom').start()

        result_nodes = layout.nodes()
        # Nodes should be arranged vertically
        for i in range(len(result_nodes) - 1):
            assert result_nodes[i].y < result_nodes[i + 1].y

    def test_wide_tree(self):
        """Test layout of wide tree."""
        nodes, links = create_wide_tree()
        layout = ReingoldTilfordLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start()

        result_nodes = layout.nodes()
        assert len(result_nodes) == 6

        # Children should be arranged horizontally
        children = result_nodes[1:6]
        x_positions = sorted([c.x for c in children])
        # Check all different
        for i in range(len(x_positions) - 1):
            assert x_positions[i] < x_positions[i + 1]

    def test_different_orientations(self):
        """Test all orientation options."""
        nodes, links = create_binary_tree()

        for orient in ['top-to-bottom', 'bottom-to-top', 'left-to-right', 'right-to-left']:
            layout = ReingoldTilfordLayout()
            layout.nodes(nodes).links(links).size([800, 600])
            layout.orientation(orient).start()

            result_nodes = layout.nodes()
            assert len(result_nodes) == 7

    def test_single_node(self):
        """Test layout with single node."""
        layout = ReingoldTilfordLayout()
        layout.nodes([{}]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 1

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = ReingoldTilfordLayout()
        layout.nodes([]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 0


# =============================================================================
# Radial Tree Tests
# =============================================================================

class TestRadialTreeLayout:
    """Tests for Radial tree layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_binary_tree()
        layout = RadialTreeLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start()

        result_nodes = layout.nodes()
        assert len(result_nodes) == 7

    def test_root_at_center(self):
        """Test that root is at center of canvas."""
        nodes, links = create_binary_tree()
        layout = RadialTreeLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start(center_graph=False)

        result_nodes = layout.nodes()
        root = result_nodes[0]

        # Root should be at center
        assert abs(root.x - 400) < 1.0
        assert abs(root.y - 300) < 1.0

    def test_children_further_from_center(self):
        """Test that children are further from center than parent."""
        nodes, links = create_binary_tree()
        layout = RadialTreeLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start(center_graph=False)

        result_nodes = layout.nodes()
        cx, cy = 400, 300

        # Root distance
        root_dist = math.sqrt(
            (result_nodes[0].x - cx) ** 2 +
            (result_nodes[0].y - cy) ** 2
        )

        # Children should be further from center
        for i in [1, 2]:
            child_dist = math.sqrt(
                (result_nodes[i].x - cx) ** 2 +
                (result_nodes[i].y - cy) ** 2
            )
            assert child_dist > root_dist

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = RadialTreeLayout()

        result = (layout
            .root(0)
            .level_radius(80)
            .start_angle(0.5)
            .sweep_angle(math.pi))

        assert result is layout
        assert layout.root() == 0
        assert layout.level_radius() == 80
        assert layout.start_angle() == 0.5
        assert layout.sweep_angle() == math.pi

    def test_wide_tree_spread(self):
        """Test that children spread around the circle."""
        nodes, links = create_wide_tree()
        layout = RadialTreeLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start(center_graph=False)

        result_nodes = layout.nodes()
        cx, cy = 400, 300

        # Calculate angles of children
        angles = []
        for i in range(1, 6):
            dx = result_nodes[i].x - cx
            dy = result_nodes[i].y - cy
            angle = math.atan2(dy, dx)
            angles.append(angle)

        # Angles should be spread out
        angles.sort()
        for i in range(len(angles) - 1):
            # Each child should be at a different angle
            assert abs(angles[i + 1] - angles[i]) > 0.1

    def test_single_node(self):
        """Test layout with single node."""
        layout = RadialTreeLayout()
        layout.nodes([{}]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 1


# =============================================================================
# Sugiyama Tests
# =============================================================================

class TestSugiyamaLayout:
    """Tests for Sugiyama DAG layout."""

    def test_basic_layout(self):
        """Test basic layout runs without error."""
        nodes, links = create_dag()
        layout = SugiyamaLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start()

        result_nodes = layout.nodes()
        assert len(result_nodes) == 5

    def test_layered_structure(self):
        """Test that DAG is laid out in layers."""
        nodes, links = create_dag()
        layout = SugiyamaLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.orientation('top-to-bottom').start()

        result_nodes = layout.nodes()

        # Node 0 should be topmost
        # Nodes 1, 2 should be at same level (middle)
        # Node 3 should be below 1, 2
        # Node 4 should be bottommost

        assert result_nodes[0].y < result_nodes[1].y
        assert result_nodes[0].y < result_nodes[2].y
        # 1 and 2 at same level
        assert abs(result_nodes[1].y - result_nodes[2].y) < 1.0
        # 3 below 1 and 2
        assert result_nodes[3].y > result_nodes[1].y
        assert result_nodes[3].y > result_nodes[2].y
        # 4 at bottom
        assert result_nodes[4].y > result_nodes[3].y

    def test_configuration_methods(self):
        """Test fluent configuration methods."""
        layout = SugiyamaLayout()

        result = (layout
            .layer_separation(120)
            .node_separation(60)
            .orientation('left-to-right')
            .crossing_iterations(30))

        assert result is layout
        assert layout.layer_separation() == 120
        assert layout.node_separation() == 60
        assert layout.orientation() == 'left-to-right'
        assert layout.crossing_iterations() == 30

    def test_tree_as_dag(self):
        """Test that trees work as DAGs."""
        nodes, links = create_binary_tree()
        layout = SugiyamaLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.start()

        result_nodes = layout.nodes()
        assert len(result_nodes) == 7

    def test_different_orientations(self):
        """Test all orientation options."""
        nodes, links = create_dag()

        for orient in ['top-to-bottom', 'bottom-to-top', 'left-to-right', 'right-to-left']:
            layout = SugiyamaLayout()
            layout.nodes(nodes).links(links).size([800, 600])
            layout.orientation(orient).start()

            result_nodes = layout.nodes()
            assert len(result_nodes) == 5

    def test_single_node(self):
        """Test layout with single node."""
        layout = SugiyamaLayout()
        layout.nodes([{}]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 1

    def test_empty_graph(self):
        """Test layout with no nodes."""
        layout = SugiyamaLayout()
        layout.nodes([]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 0

    def test_linear_dag(self):
        """Test linear DAG layout."""
        nodes = [{} for _ in range(4)]
        links = [
            {'source': 0, 'target': 1},
            {'source': 1, 'target': 2},
            {'source': 2, 'target': 3},
        ]

        layout = SugiyamaLayout()
        layout.nodes(nodes).links(links).size([800, 600])
        layout.orientation('top-to-bottom').start()

        result_nodes = layout.nodes()

        # Should be vertically ordered
        for i in range(len(result_nodes) - 1):
            assert result_nodes[i].y < result_nodes[i + 1].y


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================

class TestAllHierarchicalLayouts:
    """Tests that apply to all hierarchical algorithms."""

    @pytest.mark.parametrize("LayoutClass", [
        ReingoldTilfordLayout,
        RadialTreeLayout,
        SugiyamaLayout,
    ])
    def test_empty_graph(self, LayoutClass):
        """All algorithms should handle empty graphs."""
        layout = LayoutClass()
        layout.nodes([]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 0

    @pytest.mark.parametrize("LayoutClass", [
        ReingoldTilfordLayout,
        RadialTreeLayout,
        SugiyamaLayout,
    ])
    def test_single_node(self, LayoutClass):
        """All algorithms should handle single node."""
        layout = LayoutClass()
        layout.nodes([{}]).links([]).size([800, 600])
        layout.start()
        assert len(layout.nodes()) == 1

    @pytest.mark.parametrize("LayoutClass", [
        ReingoldTilfordLayout,
        RadialTreeLayout,
        SugiyamaLayout,
    ])
    def test_fluent_api(self, LayoutClass):
        """All algorithms support fluent API."""
        nodes, links = create_binary_tree()
        layout = LayoutClass()
        result = (layout
            .nodes(nodes)
            .links(links)
            .size([800, 600]))
        assert result is layout

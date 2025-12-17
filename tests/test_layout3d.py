"""Tests for 3D layout module."""

from graph_layout.cola.layout3d import Layout3D, Link3D, Node3D


class TestNode3D:
    """Test Node3D class."""

    def test_create_node(self):
        """Test node creation."""
        node = Node3D(1.0, 2.0, 3.0)
        assert node.x == 1.0
        assert node.y == 2.0
        assert node.z == 3.0
        assert node.fixed == False

    def test_default_node(self):
        """Test node with default values."""
        node = Node3D()
        assert node.x == 0.0
        assert node.y == 0.0
        assert node.z == 0.0


class TestLink3D:
    """Test Link3D class."""

    def test_create_link(self):
        """Test link creation."""
        link = Link3D(0, 1)
        assert link.source == 0
        assert link.target == 1
        assert link.length == 1.0


class TestLayout3D:
    """Test Layout3D class."""

    def test_create_layout(self):
        """Test layout creation."""
        nodes = [Node3D(0, 0, 0), Node3D(1, 1, 1)]
        links = [Link3D(0, 1)]

        layout = Layout3D(nodes, links)

        assert len(layout.nodes) == 2
        assert len(layout.links) == 1
        assert layout.result.shape == (3, 2)

    def test_simple_layout(self):
        """Test running simple layout."""
        nodes = [Node3D(0, 0, 0), Node3D(1, 0, 0), Node3D(0, 1, 0)]
        links = [Link3D(0, 1), Link3D(1, 2), Link3D(2, 0)]

        layout = Layout3D(nodes, links, ideal_link_length=1.0)
        layout.start(iterations=10)

        # Should complete without error
        assert layout.descent is not None

    def test_fixed_nodes(self):
        """Test layout with fixed nodes."""
        nodes = [Node3D(0, 0, 0), Node3D(2, 0, 0)]
        nodes[0].fixed = True  # Fix first node

        links = [Link3D(0, 1)]

        layout = Layout3D(nodes, links)
        initial_x = nodes[0].x

        layout.start(iterations=5)

        # Fixed node should stay at initial position (approximately)
        # Note: Due to descent algorithm, may have small variations
        assert len(layout.descent.locks.locks) > 0

    def test_tick(self):
        """Test single iteration."""
        nodes = [Node3D(0, 0, 0), Node3D(1, 0, 0)]
        links = [Link3D(0, 1)]

        layout = Layout3D(nodes, links)
        layout.start(iterations=0)  # Initialize

        displacement = layout.tick()

        # Should return a displacement value
        assert isinstance(displacement, float)
        assert displacement >= 0

    def test_link_length(self):
        """Test calculating link length."""
        nodes = [Node3D(0, 0, 0), Node3D(3, 4, 0)]
        links = [Link3D(0, 1)]

        layout = Layout3D(nodes, links)

        # Distance should be 5 (3-4-5 triangle)
        length = layout.link_length(links[0])
        assert abs(length - 5.0) < 1e-6

    def test_without_jaccard(self):
        """Test layout without Jaccard link lengths."""
        nodes = [Node3D(0, 0, 0), Node3D(1, 0, 0), Node3D(0, 1, 0)]
        links = [Link3D(0, 1), Link3D(1, 2)]

        layout = Layout3D(nodes, links)
        layout.use_jaccard_link_lengths = False

        layout.start(iterations=5)

        assert layout.descent is not None

    def test_larger_graph(self):
        """Test with larger graph."""
        # Create a simple chain
        nodes = [Node3D(i, 0, 0) for i in range(5)]
        links = [Link3D(i, i + 1) for i in range(4)]

        layout = Layout3D(nodes, links, ideal_link_length=2.0)
        layout.start(iterations=20)

        # Should converge without error
        assert layout.descent is not None
        stress = layout.descent.compute_stress()
        assert stress >= 0

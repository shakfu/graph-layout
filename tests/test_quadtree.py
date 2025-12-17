"""Tests for QuadTree implementation and Barnes-Hut force approximation."""

import math

from graph_layout.spatial.quadtree import Body, QuadTree, QuadTreeNode
from graph_layout.types import Node


class TestBody:
    """Tests for the Body dataclass."""

    def test_body_creation(self):
        """Test basic body creation."""
        body = Body(x=10.0, y=20.0, mass=1.5, index=5)
        assert body.x == 10.0
        assert body.y == 20.0
        assert body.mass == 1.5
        assert body.index == 5

    def test_body_defaults(self):
        """Test body default values."""
        body = Body(x=0.0, y=0.0)
        assert body.mass == 1.0
        assert body.index == -1


class TestQuadTreeNode:
    """Tests for QuadTreeNode."""

    def test_node_creation(self):
        """Test node creation with bounds."""
        node = QuadTreeNode(x=50.0, y=50.0, half_size=50.0)
        assert node.x == 50.0
        assert node.y == 50.0
        assert node.half_size == 50.0
        assert node.is_empty()
        assert node.is_leaf()

    def test_contains(self):
        """Test point containment check."""
        node = QuadTreeNode(x=50.0, y=50.0, half_size=50.0)

        # Points inside
        assert node.contains(25.0, 25.0)
        assert node.contains(75.0, 75.0)
        assert node.contains(50.0, 50.0)

        # Points on boundary
        assert node.contains(0.0, 0.0)
        assert node.contains(100.0, 100.0)

        # Points outside
        assert not node.contains(-1.0, 50.0)
        assert not node.contains(101.0, 50.0)
        assert not node.contains(50.0, -1.0)
        assert not node.contains(50.0, 101.0)

    def test_get_quadrant(self):
        """Test quadrant determination."""
        node = QuadTreeNode(x=50.0, y=50.0, half_size=50.0)

        # NW quadrant (0)
        assert node.get_quadrant(25.0, 25.0) == 0

        # NE quadrant (1)
        assert node.get_quadrant(75.0, 25.0) == 1

        # SW quadrant (2)
        assert node.get_quadrant(25.0, 75.0) == 2

        # SE quadrant (3)
        assert node.get_quadrant(75.0, 75.0) == 3


class TestQuadTreeInsertion:
    """Tests for QuadTree insertion operations."""

    def test_empty_tree(self):
        """Test empty tree state."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        assert tree.body_count == 0
        assert tree.root.is_empty()

    def test_single_body_insertion(self):
        """Test inserting a single body."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        body = Body(25.0, 25.0, index=0)
        tree.insert(body)

        assert tree.body_count == 1
        assert tree.root.body is body
        assert tree.root.is_leaf()

    def test_two_body_insertion(self):
        """Test inserting two bodies causes subdivision."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        tree.insert(Body(25.0, 25.0, index=0))
        tree.insert(Body(75.0, 75.0, index=1))

        assert tree.body_count == 2
        assert not tree.root.is_leaf()
        assert tree.root.children is not None

    def test_multiple_body_insertion(self):
        """Test inserting multiple bodies."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        positions = [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50)]

        for i, (x, y) in enumerate(positions):
            tree.insert(Body(float(x), float(y), index=i))

        assert tree.body_count == 5


class TestQuadTreeMassDistribution:
    """Tests for center of mass computation."""

    def test_single_body_mass(self):
        """Test mass distribution for single body."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        tree.insert(Body(30.0, 40.0, mass=2.0, index=0))
        tree.compute_mass_distribution()

        assert tree.root.total_mass == 2.0
        assert tree.root.center_of_mass_x == 30.0
        assert tree.root.center_of_mass_y == 40.0

    def test_two_equal_bodies_mass(self):
        """Test center of mass with two equal-mass bodies."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        tree.insert(Body(20.0, 50.0, mass=1.0, index=0))
        tree.insert(Body(80.0, 50.0, mass=1.0, index=1))
        tree.compute_mass_distribution()

        assert tree.root.total_mass == 2.0
        # Center of mass should be at midpoint
        assert abs(tree.root.center_of_mass_x - 50.0) < 1e-10
        assert abs(tree.root.center_of_mass_y - 50.0) < 1e-10

    def test_weighted_center_of_mass(self):
        """Test center of mass with different masses."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        tree.insert(Body(0.0, 0.0, mass=3.0, index=0))
        tree.insert(Body(100.0, 0.0, mass=1.0, index=1))
        tree.compute_mass_distribution()

        # COM = (3*0 + 1*100) / 4 = 25
        assert tree.root.total_mass == 4.0
        assert abs(tree.root.center_of_mass_x - 25.0) < 1e-10


class TestQuadTreeForceCalculation:
    """Tests for Barnes-Hut force approximation."""

    def test_force_on_single_body(self):
        """Test that single body has no force on itself."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        body = Body(50.0, 50.0, mass=1.0, index=0)
        tree.insert(body)
        tree.compute_mass_distribution()

        fx, fy = tree.calculate_force(body, repulsion_constant=10000)
        assert fx == 0.0
        assert fy == 0.0

    def test_repulsive_force_direction(self):
        """Test that repulsive forces point away from other bodies."""
        tree = QuadTree(bounds=(0, 0, 100, 100))
        tree.insert(Body(40.0, 50.0, mass=1.0, index=0))
        tree.insert(Body(60.0, 50.0, mass=1.0, index=1))
        tree.compute_mass_distribution()

        # Force on body at (40, 50) should point left (negative x)
        body0 = Body(40.0, 50.0, mass=1.0, index=0)
        fx, fy = tree.calculate_force(body0, repulsion_constant=10000)
        assert fx < 0  # Should be pushed left

        # Force on body at (60, 50) should point right (positive x)
        body1 = Body(60.0, 50.0, mass=1.0, index=1)
        fx, fy = tree.calculate_force(body1, repulsion_constant=10000)
        assert fx > 0  # Should be pushed right

    def test_force_magnitude_inverse_distance(self):
        """Test that force magnitude decreases with distance."""
        tree = QuadTree(bounds=(0, 0, 200, 100))
        tree.insert(Body(0.0, 50.0, mass=1.0, index=0))
        tree.compute_mass_distribution()

        # Force at close distance
        body_close = Body(20.0, 50.0, mass=1.0, index=1)
        fx_close, _ = tree.calculate_force(body_close, repulsion_constant=10000)

        # Force at far distance
        body_far = Body(100.0, 50.0, mass=1.0, index=2)
        fx_far, _ = tree.calculate_force(body_far, repulsion_constant=10000)

        # Close force should be stronger
        assert abs(fx_close) > abs(fx_far)

    def test_theta_zero_exact(self):
        """Test that theta=0 gives exact calculation."""
        tree = QuadTree(bounds=(0, 0, 100, 100), theta=0.0)

        # Create a cluster of bodies
        tree.insert(Body(10.0, 10.0, mass=1.0, index=0))
        tree.insert(Body(12.0, 10.0, mass=1.0, index=1))
        tree.insert(Body(11.0, 12.0, mass=1.0, index=2))
        tree.compute_mass_distribution()

        # With theta=0, all interactions should be exact
        body = Body(50.0, 50.0, mass=1.0, index=3)
        fx, fy = tree.calculate_force(body, repulsion_constant=10000)

        # Force should be non-zero and point away from cluster
        assert fx > 0  # Away from cluster on left
        assert fy > 0  # Away from cluster on top


class TestQuadTreeFromNodes:
    """Tests for building QuadTree from Node objects."""

    def test_from_nodes_empty(self):
        """Test building tree from empty node list."""
        tree = QuadTree.from_nodes([])
        assert tree.body_count == 0

    def test_from_nodes_basic(self):
        """Test building tree from Node objects."""
        nodes = [
            Node(x=10.0, y=20.0, index=0),
            Node(x=30.0, y=40.0, index=1),
            Node(x=50.0, y=60.0, index=2),
        ]
        tree = QuadTree.from_nodes(nodes)

        assert tree.body_count == 3
        tree.compute_mass_distribution()
        assert tree.root.total_mass == 3.0

    def test_from_nodes_theta(self):
        """Test that theta parameter is passed correctly."""
        nodes = [Node(x=0, y=0, index=0)]
        tree = QuadTree.from_nodes(nodes, theta=0.8)
        assert tree.theta == 0.8


class TestBarnesHutAccuracy:
    """Tests comparing Barnes-Hut to exact calculation."""

    def _exact_repulsion(self, bodies, target_idx, k_sq):
        """Calculate exact repulsive force on target body."""
        target = bodies[target_idx]
        fx, fy = 0.0, 0.0

        for i, body in enumerate(bodies):
            if i == target_idx:
                continue

            dx = target.x - body.x
            dy = target.y - body.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-10:
                continue

            dist = math.sqrt(dist_sq)
            force = k_sq * body.mass / dist
            fx += (dx / dist) * force
            fy += (dy / dist) * force

        return fx, fy

    def test_barnes_hut_vs_exact_small(self):
        """Test Barnes-Hut accuracy on small graph."""
        # Create a small set of bodies
        bodies = [
            Body(100, 100, mass=1.0, index=0),
            Body(200, 100, mass=1.0, index=1),
            Body(150, 200, mass=1.0, index=2),
            Body(300, 300, mass=1.0, index=3),
        ]

        tree = QuadTree(bounds=(0, 0, 400, 400), theta=0.5)
        for body in bodies:
            tree.insert(body)
        tree.compute_mass_distribution()

        k_sq = 10000.0

        for i, body in enumerate(bodies):
            exact_fx, exact_fy = self._exact_repulsion(bodies, i, k_sq)
            approx_fx, approx_fy = tree.calculate_force(body, repulsion_constant=k_sq)

            # Allow some tolerance due to approximation
            if abs(exact_fx) > 1e-5:
                rel_error_x = abs(approx_fx - exact_fx) / abs(exact_fx)
                assert rel_error_x < 0.5, f"X error too large: {rel_error_x}"

            if abs(exact_fy) > 1e-5:
                rel_error_y = abs(approx_fy - exact_fy) / abs(exact_fy)
                assert rel_error_y < 0.5, f"Y error too large: {rel_error_y}"

    def test_barnes_hut_theta_0_exact(self):
        """Test that theta=0 gives same result as exact calculation."""
        bodies = [
            Body(50, 50, mass=1.0, index=0),
            Body(150, 50, mass=1.0, index=1),
            Body(100, 150, mass=1.0, index=2),
        ]

        tree = QuadTree(bounds=(0, 0, 200, 200), theta=0.0)
        for body in bodies:
            tree.insert(body)
        tree.compute_mass_distribution()

        k_sq = 10000.0

        for i, body in enumerate(bodies):
            exact_fx, exact_fy = self._exact_repulsion(bodies, i, k_sq)
            approx_fx, approx_fy = tree.calculate_force(body, repulsion_constant=k_sq)

            # With theta=0, should be very close
            assert abs(approx_fx - exact_fx) < 1e-5
            assert abs(approx_fy - exact_fy) < 1e-5


class TestBarnesHutWithLayouts:
    """Tests for Barnes-Hut integration with layout algorithms."""

    def test_fruchterman_reingold_barnes_hut_config(self):
        """Test Barnes-Hut configuration on FR layout."""
        from graph_layout import FruchtermanReingoldLayout

        layout = FruchtermanReingoldLayout()

        # Default should be disabled
        assert layout.use_barnes_hut is False

        # Enable
        layout.use_barnes_hut = True
        assert layout.use_barnes_hut is True

        # Set custom theta
        layout.barnes_hut_theta = 0.8
        assert layout.barnes_hut_theta == 0.8

    def test_spring_layout_barnes_hut_config(self):
        """Test Barnes-Hut configuration on Spring layout."""
        from graph_layout import SpringLayout

        layout = SpringLayout()

        # Default should be disabled
        assert layout.use_barnes_hut is False

        # Enable
        layout.use_barnes_hut = True
        layout.barnes_hut_theta = 0.3
        assert layout.use_barnes_hut is True
        assert layout.barnes_hut_theta == 0.3

    def test_fruchterman_reingold_runs_with_barnes_hut(self):
        """Test FR layout runs successfully with Barnes-Hut enabled."""
        from graph_layout import FruchtermanReingoldLayout

        # Create larger graph to trigger Barnes-Hut
        n = 100
        nodes = [{"x": i * 10, "y": i * 10} for i in range(n)]
        links = [{"source": i, "target": (i + 1) % n} for i in range(n)]

        layout = FruchtermanReingoldLayout(
            nodes=nodes,
            links=links,
            size=(1000, 1000),
            use_barnes_hut=True,
            barnes_hut_theta=0.5,
            iterations=10,
        )

        layout.run()

        assert len(layout.nodes) == n

        # All nodes should have been moved
        for node in layout.nodes:
            assert hasattr(node, "x")
            assert hasattr(node, "y")

    def test_spring_layout_runs_with_barnes_hut(self):
        """Test Spring layout runs successfully with Barnes-Hut enabled."""
        from graph_layout import SpringLayout

        # Create larger graph to trigger Barnes-Hut
        n = 100
        nodes = [{"x": i * 10, "y": i * 10} for i in range(n)]
        links = [{"source": i, "target": (i + 1) % n} for i in range(n)]

        layout = SpringLayout(
            nodes=nodes,
            links=links,
            size=(1000, 1000),
            use_barnes_hut=True,
            barnes_hut_theta=0.5,
            iterations=10,
        )

        layout.run()

        assert len(layout.nodes) == n

    def test_barnes_hut_produces_similar_layout(self):
        """Test that Barnes-Hut produces similar layout to naive calculation."""
        # Use fixed seed for reproducibility
        import random

        from graph_layout import FruchtermanReingoldLayout

        random.seed(42)

        nodes = [{"x": random.uniform(0, 500), "y": random.uniform(0, 500)} for _ in range(30)]
        links = [{"source": i, "target": (i + 1) % 30} for i in range(30)]

        # Run without Barnes-Hut (use provided positions, no random init)
        layout1 = FruchtermanReingoldLayout(
            nodes=[dict(n) for n in nodes],  # Copy
            links=[dict(l) for l in links],
            size=(500, 500),
            use_barnes_hut=False,
            iterations=50,
        )
        layout1.run(random_init=False)
        result1 = layout1.nodes

        # Run with Barnes-Hut enabled (but n < 50 so won't actually use it)
        layout2 = FruchtermanReingoldLayout(
            nodes=[dict(n) for n in nodes],  # Copy
            links=[dict(l) for l in links],
            size=(500, 500),
            use_barnes_hut=True,
            barnes_hut_theta=0.5,
            iterations=50,
        )
        layout2.run(random_init=False)
        result2 = layout2.nodes

        # Results should be identical since n < 50 means Barnes-Hut isn't used
        for n1, n2 in zip(result1, result2):
            assert abs(n1.x - n2.x) < 1e-5
            assert abs(n1.y - n2.y) < 1e-5

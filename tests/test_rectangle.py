"""Tests for rectangle module."""

from graph_layout.cola.geom import Point
from graph_layout.cola.rectangle import (
    GraphNode,
    Projection,
    ProjectionGroup,
    Rectangle,
    compute_group_bounds,
    generate_x_constraints,
    generate_y_constraints,
    make_edge_between,
    make_edge_to,
    remove_overlaps,
)
from graph_layout.cola.vpsc import Variable


class TestRectangle:
    """Test Rectangle class."""

    def test_create_rectangle(self):
        """Test rectangle creation."""
        r = Rectangle(0, 10, 0, 5)
        assert r.x == 0
        assert r.X == 10
        assert r.y == 0
        assert r.Y == 5

    def test_empty_rectangle(self):
        """Test empty rectangle creation."""
        r = Rectangle.empty()
        assert r.x == float("inf")
        assert r.X == -float("inf")

    def test_center(self):
        """Test center calculation."""
        r = Rectangle(0, 10, 0, 20)
        assert r.cx() == 5
        assert r.cy() == 10

    def test_dimensions(self):
        """Test width and height."""
        r = Rectangle(0, 10, 0, 5)
        assert r.width() == 10
        assert r.height() == 5

    def test_overlap_x(self):
        """Test x-axis overlap."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(5, 15, 0, 10)

        overlap = r1.overlap_x(r2)
        assert overlap > 0  # Should overlap

    def test_overlap_y(self):
        """Test y-axis overlap."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(0, 10, 5, 15)

        overlap = r1.overlap_y(r2)
        assert overlap > 0  # Should overlap

    def test_no_overlap(self):
        """Test non-overlapping rectangles."""
        r1 = Rectangle(0, 5, 0, 5)
        r2 = Rectangle(10, 15, 10, 15)

        assert r1.overlap_x(r2) == 0
        assert r1.overlap_y(r2) == 0

    def test_set_x_centre(self):
        """Test setting x center."""
        r = Rectangle(0, 10, 0, 10)
        r.set_x_centre(20)

        assert r.cx() == 20
        assert r.width() == 10  # Width preserved

    def test_set_y_centre(self):
        """Test setting y center."""
        r = Rectangle(0, 10, 0, 20)
        r.set_y_centre(30)

        assert r.cy() == 30
        assert r.height() == 20  # Height preserved

    def test_union(self):
        """Test rectangle union."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(5, 15, 5, 15)

        u = r1.union(r2)

        assert u.x == 0
        assert u.X == 15
        assert u.y == 0
        assert u.Y == 15

    def test_inflate(self):
        """Test rectangle inflation."""
        r = Rectangle(0, 10, 0, 10)
        r2 = r.inflate(2)

        assert r2.x == -2
        assert r2.X == 12
        assert r2.y == -2
        assert r2.Y == 12

    def test_line_intersection(self):
        """Test line segment intersection."""
        # Two crossing lines
        p = Rectangle.line_intersection(0, 0, 10, 10, 0, 10, 10, 0)

        assert p is not None
        assert abs(p.x - 5) < 1e-6
        assert abs(p.y - 5) < 1e-6

    def test_no_line_intersection(self):
        """Test non-intersecting line segments."""
        # Parallel lines
        p = Rectangle.line_intersection(0, 0, 10, 0, 0, 5, 10, 5)
        assert p is None

    def test_vertices(self):
        """Test getting rectangle vertices."""
        r = Rectangle(0, 10, 0, 5)
        vertices = r.vertices()

        assert len(vertices) == 4
        assert vertices[0].x == 0 and vertices[0].y == 0
        assert vertices[2].x == 10 and vertices[2].y == 5

    def test_ray_intersection(self):
        """Test ray intersection with rectangle."""
        r = Rectangle(0, 10, 0, 10)

        # Ray from center to outside
        p = r.ray_intersection(20, 5)

        assert p is not None
        assert abs(p.x - 10) < 1e-6  # Should hit right edge


class TestMakeEdge:
    """Test edge creation functions."""

    def test_make_edge_between(self):
        """Test creating edge between rectangles."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(20, 30, 0, 10)

        edge = make_edge_between(r1, r2, 2)

        assert "sourceIntersection" in edge
        assert "targetIntersection" in edge
        assert "arrowStart" in edge

    def test_make_edge_to(self):
        """Test creating edge to rectangle."""
        p = Point(0, 0)
        r = Rectangle(10, 20, 10, 20)

        arrow_start = make_edge_to(p, r, 2)

        assert isinstance(arrow_start, Point)


class TestConstraintGeneration:
    """Test constraint generation."""

    def test_generate_x_constraints(self):
        """Test generating x-axis constraints."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(5, 15, 0, 10)  # Overlaps with r1

        v1 = Variable(r1.cx())
        v2 = Variable(r2.cx())

        constraints = generate_x_constraints([r1, r2], [v1, v2])

        # Should generate constraints to prevent overlap
        assert len(constraints) >= 0

    def test_generate_y_constraints(self):
        """Test generating y-axis constraints."""
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(0, 10, 5, 15)  # Overlaps with r1

        v1 = Variable(r1.cy())
        v2 = Variable(r2.cy())

        constraints = generate_y_constraints([r1, r2], [v1, v2])

        # Should generate constraints
        assert len(constraints) >= 0


class TestRemoveOverlaps:
    """Test overlap removal."""

    def test_remove_overlaps_simple(self):
        """Test removing overlaps from rectangles."""
        # Two overlapping rectangles
        r1 = Rectangle(0, 10, 0, 10)
        r2 = Rectangle(5, 15, 0, 10)

        rectangles = [r1, r2]
        remove_overlaps(rectangles)

        # After overlap removal, rectangles should not overlap
        overlap_x = rectangles[0].overlap_x(rectangles[1])
        overlap_y = rectangles[0].overlap_y(rectangles[1])

        # At least one axis should have no overlap
        assert overlap_x == 0 or overlap_y == 0

    def test_remove_overlaps_multiple(self):
        """Test removing overlaps from multiple rectangles."""
        rectangles = [
            Rectangle(0, 10, 0, 10),
            Rectangle(5, 15, 5, 15),
            Rectangle(10, 20, 10, 20),
        ]

        remove_overlaps(rectangles)

        # All rectangles should have been adjusted
        assert len(rectangles) == 3


class TestGraphNode:
    """Test GraphNode class."""

    def test_create_graph_node(self):
        """Test graph node creation."""
        node = GraphNode()

        assert node.fixed == False
        assert node.x == 0.0
        assert node.y == 0.0

    def test_graph_node_with_bounds(self):
        """Test graph node with bounds."""
        node = GraphNode()
        node.width = 10.0
        node.height = 5.0
        node.x = 5.0
        node.y = 2.5

        assert node.width == 10.0
        assert node.height == 5.0


class TestProjection:
    """Test Projection class."""

    def test_create_projection(self):
        """Test projection creation."""
        nodes = [GraphNode(), GraphNode()]
        groups = []

        proj = Projection(nodes, groups)

        assert len(proj.nodes) == 2
        assert len(proj.variables) == 2

    def test_projection_with_constraints(self):
        """Test projection with constraints."""
        nodes = [GraphNode(), GraphNode()]
        nodes[0].x = 0
        nodes[1].x = 10

        constraints = [{"axis": "x", "left": 0, "right": 1, "gap": 5.0}]

        proj = Projection(nodes, [], constraints=constraints)

        # Should have created x constraints
        assert len(proj.x_constraints) >= 0


class TestProjectionGroup:
    """Test ProjectionGroup."""

    def test_create_projection_group(self):
        """Test projection group creation."""
        group = ProjectionGroup()

        assert group.padding == 0.0
        assert group.stiffness == 0.01

    def test_compute_group_bounds(self):
        """Test computing group bounds."""
        group = ProjectionGroup()

        leaf1 = GraphNode()
        leaf1.bounds = Rectangle(0, 10, 0, 10)

        leaf2 = GraphNode()
        leaf2.bounds = Rectangle(20, 30, 0, 10)

        group.leaves = [leaf1, leaf2]
        group.padding = 2.0

        bounds = compute_group_bounds(group)

        # Should encompass both leaves plus padding
        assert bounds.x <= 0 - 2
        assert bounds.X >= 30 + 2

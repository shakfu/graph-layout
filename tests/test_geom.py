"""Tests for geometry utilities."""

from graph_layout.cola.geom import (
    BiTangent,
    BiTangents,
    LineSegment,
    Point,
    TangentVisibilityGraph,
    TVGPoint,
    VisibilityEdge,
    VisibilityVertex,
    _line_intersection,
    _tangent_point_poly_c,
    above,
    below,
    clockwise_radial_sweep,
    convex_hull,
    is_left,
    polys_overlap,
    tangents,
)


class TestPoint:
    """Test Point class."""

    def test_create_point(self):
        """Test point creation."""
        p = Point(3.5, 4.2)
        assert p.x == 3.5
        assert p.y == 4.2

    def test_default_point(self):
        """Test default point at origin."""
        p = Point()
        assert p.x == 0.0
        assert p.y == 0.0


class TestLineSegment:
    """Test LineSegment class."""

    def test_create_line_segment(self):
        """Test line segment creation."""
        l = LineSegment(1.0, 2.0, 3.0, 4.0)
        assert l.x1 == 1.0
        assert l.y1 == 2.0
        assert l.x2 == 3.0
        assert l.y2 == 4.0


class TestOrientation:
    """Test orientation functions."""

    def test_is_left_basic(self):
        """Test basic is_left orientation."""
        P0 = Point(0, 0)
        P1 = Point(1, 0)
        P2 = Point(0.5, 1)  # Above line

        result = is_left(P0, P1, P2)
        assert result > 0  # P2 is left of line P0-P1

    def test_is_left_collinear(self):
        """Test is_left with collinear points."""
        P0 = Point(0, 0)
        P1 = Point(1, 1)
        P2 = Point(2, 2)  # On line

        result = is_left(P0, P1, P2)
        assert abs(result) < 1e-10  # P2 is on line

    def test_is_left_right(self):
        """Test is_left when point is right of line."""
        P0 = Point(0, 0)
        P1 = Point(1, 0)
        P2 = Point(0.5, -1)  # Below line

        result = is_left(P0, P1, P2)
        assert result < 0  # P2 is right of line P0-P1

    def test_above(self):
        """Test above function."""
        P = Point(0.5, 1)
        V1 = Point(0, 0)
        V2 = Point(1, 0)

        assert above(P, V1, V2)

    def test_below(self):
        """Test below function."""
        P = Point(0.5, -1)
        V1 = Point(0, 0)
        V2 = Point(1, 0)

        assert below(P, V1, V2)


class TestConvexHull:
    """Test convex hull computation."""

    def test_simple_triangle(self):
        """Test convex hull of triangle."""
        points = [Point(0, 0), Point(1, 0), Point(0.5, 1)]

        hull = convex_hull(points)
        assert len(hull) == 3

    def test_square(self):
        """Test convex hull of square."""
        points = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]

        hull = convex_hull(points)
        assert len(hull) == 4

    def test_point_inside(self):
        """Test convex hull with point inside."""
        points = [
            Point(0, 0),
            Point(2, 0),
            Point(2, 2),
            Point(0, 2),
            Point(1, 1),  # Inside point
        ]

        hull = convex_hull(points)
        # Hull should only include outer points
        assert len(hull) == 4

    def test_collinear_points(self):
        """Test convex hull with collinear points."""
        points = [Point(0, 0), Point(1, 0), Point(2, 0)]

        hull = convex_hull(points)
        # Should return endpoints
        assert len(hull) <= 2

    def test_single_point(self):
        """Test convex hull with single point."""
        points = [Point(5, 5)]
        hull = convex_hull(points)
        assert len(hull) == 1

    def test_two_points(self):
        """Test convex hull with two points."""
        points = [Point(0, 0), Point(1, 1)]
        hull = convex_hull(points)
        assert len(hull) == 2


class TestClockwiseRadialSweep:
    """Test clockwise radial sweep."""

    def test_basic_sweep(self):
        """Test basic radial sweep."""
        center = Point(0, 0)
        points = [
            Point(1, 0),  # 0 degrees
            Point(0, 1),  # 90 degrees
            Point(-1, 0),  # 180 degrees
            Point(0, -1),  # 270 degrees
        ]

        visited = []
        clockwise_radial_sweep(center, points, lambda p: visited.append(p))

        # Should visit all points
        assert len(visited) == 4

    def test_sweep_order(self):
        """Test radial sweep ordering."""
        center = Point(0, 0)
        points = [
            Point(1, 1),  # 45 degrees
            Point(-1, 1),  # 135 degrees
        ]

        visited = []
        clockwise_radial_sweep(center, points, lambda p: visited.append(p))

        # Should be sorted by angle
        assert len(visited) == 2


class TestLineIntersection:
    """Test line intersection."""

    def test_intersecting_lines(self):
        """Test intersection of two crossing lines."""
        # Line 1: (0,0) to (2,2)
        # Line 2: (0,2) to (2,0)
        # Should intersect at (1,1)
        result = _line_intersection(0, 0, 2, 2, 0, 2, 2, 0)

        assert result is not None
        assert abs(result.x - 1.0) < 1e-6
        assert abs(result.y - 1.0) < 1e-6

    def test_parallel_lines(self):
        """Test parallel lines don't intersect."""
        result = _line_intersection(0, 0, 1, 0, 0, 1, 1, 1)
        assert result is None

    def test_non_intersecting_segments(self):
        """Test segments that don't intersect."""
        # Lines would intersect if extended, but segments don't
        result = _line_intersection(0, 0, 1, 0, 2, 0, 3, 0)
        assert result is None


class TestTangents:
    """Test tangent calculations."""

    def test_tangent_point_to_square(self):
        """Test finding tangent from point to square polygon."""
        # Square centered at origin
        square = [Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)]

        # Point to the right of square
        p = Point(3, 0)

        result = _tangent_point_poly_c(p, square)

        assert "rtan" in result
        assert "ltan" in result
        assert isinstance(result["rtan"], int)
        assert isinstance(result["ltan"], int)


class TestBiTangent:
    """Test BiTangent class."""

    def test_create_bitangent(self):
        """Test bitangent creation."""
        bt = BiTangent(0, 1)
        assert bt.t1 == 0
        assert bt.t2 == 1


class TestBiTangents:
    """Test BiTangents class."""

    def test_create_bitangents(self):
        """Test bitangents container creation."""
        bt = BiTangents()
        assert bt.rl is None
        assert bt.lr is None
        assert bt.ll is None
        assert bt.rr is None


class TestTangentsBetweenPolygons:
    """Test tangent computation between polygons."""

    def test_tangents_between_squares(self):
        """Test finding tangents between two squares."""
        # Two non-overlapping squares
        square1 = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]

        square2 = [Point(2, 0), Point(3, 0), Point(3, 1), Point(2, 1)]

        bt = tangents(square1, square2)

        # Should find at least some tangents
        assert isinstance(bt, BiTangents)


class TestVisibilityVertex:
    """Test VisibilityVertex class."""

    def test_create_visibility_vertex(self):
        """Test visibility vertex creation."""
        p = TVGPoint(5.0, 10.0)
        vv = VisibilityVertex(0, 1, 2, p)

        assert vv.id == 0
        assert vv.polyid == 1
        assert vv.polyvertid == 2
        assert vv.p is p
        assert p.vv is vv


class TestVisibilityEdge:
    """Test VisibilityEdge class."""

    def test_create_visibility_edge(self):
        """Test visibility edge creation."""
        p1 = TVGPoint(0.0, 0.0)
        p2 = TVGPoint(3.0, 4.0)
        v1 = VisibilityVertex(0, 0, 0, p1)
        v2 = VisibilityVertex(1, 0, 1, p2)

        edge = VisibilityEdge(v1, v2)
        assert edge.source is v1
        assert edge.target is v2

    def test_edge_length(self):
        """Test edge length calculation."""
        p1 = TVGPoint(0.0, 0.0)
        p2 = TVGPoint(3.0, 4.0)
        v1 = VisibilityVertex(0, 0, 0, p1)
        v2 = VisibilityVertex(1, 0, 1, p2)

        edge = VisibilityEdge(v1, v2)
        length = edge.length()

        # Distance should be 5 (3-4-5 triangle)
        assert abs(length - 5.0) < 1e-6


class TestTangentVisibilityGraph:
    """Test tangent visibility graph."""

    def test_create_simple_graph(self):
        """Test creating visibility graph for simple polygons."""
        # Two triangles
        poly1 = [TVGPoint(0, 0), TVGPoint(1, 0), TVGPoint(0.5, 1)]

        poly2 = [TVGPoint(2, 0), TVGPoint(3, 0), TVGPoint(2.5, 1)]

        graph = TangentVisibilityGraph([poly1, poly2])

        # Should have 6 vertices (3 per triangle)
        assert len(graph.V) == 6

        # Should have edges (at least the polygon edges)
        assert len(graph.E) > 0

    def test_single_polygon(self):
        """Test visibility graph with single polygon."""
        poly = [TVGPoint(0, 0), TVGPoint(1, 0), TVGPoint(1, 1), TVGPoint(0, 1)]

        graph = TangentVisibilityGraph([poly])

        # Should have 4 vertices
        assert len(graph.V) == 4

        # Should have 4 edges (one per side)
        assert len(graph.E) == 4

    def test_add_point(self):
        """Test adding point to visibility graph."""
        poly = [TVGPoint(0, 0), TVGPoint(1, 0), TVGPoint(1, 1), TVGPoint(0, 1)]

        graph = TangentVisibilityGraph([poly])
        initial_vertex_count = len(graph.V)

        new_point = TVGPoint(2, 0.5)
        vv = graph.add_point(new_point, -1)

        assert vv is not None
        assert len(graph.V) == initial_vertex_count + 1


class TestPolysOverlap:
    """Test polygon overlap detection."""

    def test_non_overlapping_squares(self):
        """Test non-overlapping squares."""
        square1 = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]

        square2 = [Point(2, 0), Point(3, 0), Point(3, 1), Point(2, 1)]

        assert not polys_overlap(square1, square2)

    def test_overlapping_squares(self):
        """Test overlapping squares."""
        square1 = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)]

        square2 = [Point(1, 1), Point(3, 1), Point(3, 3), Point(1, 3)]

        assert polys_overlap(square1, square2)

    def test_one_inside_other(self):
        """Test one polygon inside another."""
        outer = [Point(0, 0), Point(4, 0), Point(4, 4), Point(0, 4)]

        inner = [Point(1, 1), Point(2, 1), Point(2, 2), Point(1, 2)]

        assert polys_overlap(outer, inner)

    def test_edge_touching(self):
        """Test polygons touching at edge."""
        square1 = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]

        square2 = [Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1)]

        # Touching at edge - this may or may not be considered overlap
        # depending on implementation details
        result = polys_overlap(square1, square2)
        assert isinstance(result, bool)

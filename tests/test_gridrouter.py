"""Tests for gridrouter module."""

from graph_layout.cola.geom import Point
from graph_layout.cola.gridrouter import (
    GridLine,
    GridRouter,
    LongestCommonSubsequence,
    NodeWrapper,
    Vert,
)
from graph_layout.cola.rectangle import Rectangle


class SimpleNode:
    """Simple node for testing."""

    def __init__(self, x: float, y: float, w: float, h: float, children=None):
        self.bounds = Rectangle(x, y, x + w, y + h)
        self.children = children if children is not None else []


class SimpleNodeAccessor:
    """Simple node accessor."""

    def get_children(self, v: SimpleNode) -> list[int]:
        return v.children

    def get_bounds(self, v: SimpleNode) -> Rectangle:
        return v.bounds


class TestNodeWrapper:
    """Test NodeWrapper class."""

    def test_create_leaf_node(self):
        """Test creating a leaf node."""
        rect = Rectangle(0, 0, 10, 10)
        node = NodeWrapper(0, rect)

        assert node.id == 0
        assert node.rect == rect
        assert node.leaf is True
        assert len(node.children) == 0

    def test_create_group_node(self):
        """Test creating a group node."""
        rect = Rectangle(0, 0, 100, 100)
        node = NodeWrapper(0, rect, [1, 2, 3])

        assert node.leaf is False
        assert len(node.children) == 3


class TestVert:
    """Test Vert class."""

    def test_create_vert(self):
        """Test vertex creation."""
        v = Vert(0, 5.0, 10.0)

        assert v.id == 0
        assert v.x == 5.0
        assert v.y == 10.0
        assert v.node is None

    def test_vert_with_node(self):
        """Test vertex with node."""
        rect = Rectangle(0, 0, 10, 10)
        node = NodeWrapper(0, rect)
        v = Vert(0, 5.0, 5.0, node)

        assert v.node == node


class TestLongestCommonSubsequence:
    """Test LCS algorithm."""

    def test_identical_sequences(self):
        """Test with identical sequences."""
        s = [1, 2, 3, 4]
        t = [1, 2, 3, 4]
        lcs = LongestCommonSubsequence(s, t)

        assert lcs.length == 4
        assert lcs.get_sequence() == [1, 2, 3, 4]

    def test_partial_match(self):
        """Test with partial match."""
        s = [1, 2, 3, 4, 5]
        t = [0, 2, 3, 4, 6]
        lcs = LongestCommonSubsequence(s, t)

        assert lcs.length == 3
        assert lcs.get_sequence() == [2, 3, 4]

    def test_no_match(self):
        """Test with no match."""
        s = [1, 2, 3]
        t = [4, 5, 6]
        lcs = LongestCommonSubsequence(s, t)

        assert lcs.length == 0
        assert lcs.get_sequence() == []

    def test_reversed_match(self):
        """Test with reversed sequence."""
        s = [1, 2, 3, 4]
        t = [4, 3, 2, 1]
        lcs = LongestCommonSubsequence(s, t)

        # Should find best match considering reversal
        assert lcs.length > 0


class TestGridLine:
    """Test GridLine class."""

    def test_create_gridline(self):
        """Test grid line creation."""
        rect = Rectangle(0, 0, 10, 10)
        nodes = [NodeWrapper(0, rect)]
        line = GridLine(nodes, 5.0)

        assert len(line.nodes) == 1
        assert line.pos == 5.0


class TestGridRouter:
    """Test GridRouter class."""

    def test_simple_grid(self):
        """Test simple grid creation."""
        nodes = [SimpleNode(0, 0, 10, 10), SimpleNode(20, 0, 10, 10), SimpleNode(0, 20, 10, 10)]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        assert len(router.nodes) == 3
        assert len(router.leaves) == 3
        assert len(router.groups) == 0
        assert len(router.verts) > 0
        assert len(router.edges) > 0

    def test_grid_with_groups(self):
        """Test grid with group hierarchy."""
        nodes = [
            SimpleNode(0, 0, 10, 10),  # 0: leaf
            SimpleNode(20, 0, 10, 10),  # 1: leaf
            SimpleNode(0, 0, 50, 50, [0, 1]),  # 2: group
        ]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        assert len(router.leaves) == 2
        assert len(router.groups) == 1
        assert router.nodes[0].parent == router.nodes[2]
        assert router.nodes[1].parent == router.nodes[2]

    def test_depth_calculation(self):
        """Test depth calculation."""
        nodes = [
            SimpleNode(0, 0, 10, 10),  # 0: leaf
            SimpleNode(0, 0, 20, 20, [0]),  # 1: group
            SimpleNode(0, 0, 30, 30, [1]),  # 2: group
        ]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        assert router._get_depth(router.nodes[0]) == 2
        assert router._get_depth(router.nodes[1]) == 1
        assert router._get_depth(router.nodes[2]) == 0

    def test_route_between_nodes(self):
        """Test routing between two nodes."""
        nodes = [SimpleNode(0, 0, 10, 10), SimpleNode(50, 50, 10, 10)]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        path = router.route(0, 1)

        # Should have at least one point in the path
        assert len(path) >= 1
        # Path should be list of Points or Verts
        assert all(hasattr(p, "x") and hasattr(p, "y") for p in path)

    def test_sibling_obstacles(self):
        """Test sibling obstacle detection."""
        nodes = [
            SimpleNode(0, 0, 10, 10),  # 0
            SimpleNode(20, 0, 10, 10),  # 1
            SimpleNode(40, 0, 10, 10),  # 2
            SimpleNode(0, 0, 70, 20, [0, 1, 2]),  # 3: parent
        ]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        # When routing from 0 to 1, node 2 is sibling obstacle
        obstacles = router.sibling_obstacles(router.nodes[0], router.nodes[1])
        obstacle_ids = [o.id for o in obstacles]

        assert 2 in obstacle_ids

    def test_make_segments(self):
        """Test segment creation from path."""
        path = [
            Point(0, 0),
            Point(10, 0),
            Point(20, 0),  # straight continuation
            Point(20, 10),
        ]

        segments = GridRouter.make_segments(path)

        # Should merge first two straight segments
        assert len(segments) == 2
        # First segment should go from (0,0) to (20,0)
        assert segments[0][0].x == 0
        assert segments[0][1].x == 20

    def test_is_left(self):
        """Test left turn detection."""
        a = Point(0, 0)
        b = Point(10, 0)
        c = Point(10, -10)  # up in screen coords (Y down)

        # In screen coordinates (Y down), this is a left turn
        assert GridRouter.is_left(a, b, c) is True

        c2 = Point(10, 10)  # down in screen coords
        # This is a right turn in screen coordinates
        assert GridRouter.is_left(a, b, c2) is False

    def test_angle_between_lines(self):
        """Test angle calculation."""
        line1 = [Point(0, 0), Point(10, 0)]
        line2 = [Point(10, 0), Point(10, 10)]

        angle = GridRouter.angle_between_2_lines(line1, line2)

        # Angle should be non-zero
        # The actual angle depends on coordinate system and direction
        assert angle != 0

    def test_get_route_path_simple(self):
        """Test SVG path generation."""
        route = [[Point(0, 0), Point(10, 0)], [Point(10, 0), Point(10, 10)]]

        result = GridRouter.get_route_path(route, 2.0, 3.0, 5.0)

        assert "routepath" in result
        assert "arrowpath" in result
        assert result["routepath"].startswith("M")
        assert "L" in result["routepath"]

    def test_get_route_path_single_segment(self):
        """Test path generation for single segment."""
        route = [[Point(0, 0), Point(10, 0)]]

        result = GridRouter.get_route_path(route, 0, 2.0, 5.0)

        assert "M" in result["routepath"]
        assert "L" in result["routepath"]

    def test_order_edges(self):
        """Test edge ordering."""
        # Two paths with common section
        path1 = [Point(0, 0), Point(10, 0), Point(10, 10)]
        path2 = [Point(0, 5), Point(10, 5), Point(10, 15)]

        order_fn = GridRouter.order_edges([path1, path2])

        # Should return a callable
        assert callable(order_fn)

    def test_get_segment_sets(self):
        """Test segment grouping."""
        # Segments are lists of two points (dicts with x, y)
        routes = [
            [  # route 0
                [{"x": 0, "y": 0}, {"x": 0, "y": 10}]  # vertical segment
            ],
            [  # route 1
                [{"x": 0, "y": 5}, {"x": 0, "y": 15}]  # vertical segment
            ],
        ]

        sets = GridRouter.get_segment_sets(routes, "x", "y")

        # Should group segments at same x position
        assert len(sets) >= 1
        assert all("pos" in s and "segments" in s for s in sets)

    def test_mid_points_single(self):
        """Test midpoint calculation with single value."""
        nodes = [SimpleNode(0, 0, 10, 10)]
        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        mids = router._mid_points([5.0])
        assert len(mids) == 1
        assert mids[0] == 5.0

    def test_mid_points_multiple(self):
        """Test midpoint calculation with multiple values."""
        nodes = [SimpleNode(0, 0, 10, 10)]
        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        mids = router._mid_points([0.0, 10.0, 20.0])
        assert len(mids) == 4
        assert mids[0] < 0.0  # boundary before first
        assert mids[-1] > 20.0  # boundary after last

    def test_grid_lines_overlapping(self):
        """Test grid line detection with overlapping nodes."""
        # Two nodes overlapping in x
        nodes = [SimpleNode(0, 0, 10, 10), SimpleNode(5, 20, 10, 10)]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        # Should create grid lines
        assert len(router.cols) >= 1
        assert len(router.rows) >= 1


class TestRouteEdges:
    """Test complete edge routing."""

    def test_route_edges_simple(self):
        """Test routing multiple edges."""
        nodes = [SimpleNode(0, 0, 10, 10), SimpleNode(50, 0, 10, 10), SimpleNode(100, 0, 10, 10)]

        accessor = SimpleNodeAccessor()
        router = GridRouter(nodes, accessor)

        edges = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]

        routes = router.route_edges(edges, 2.0, lambda e: e["source"], lambda e: e["target"])

        assert len(routes) == 2
        # Each route should be list of segments
        for route in routes:
            assert isinstance(route, list)
            if len(route) > 0:
                assert isinstance(route[0], list)

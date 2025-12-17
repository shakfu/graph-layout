"""
Rectangle operations and projection for graph layout.

This module provides rectangle geometry, overlap removal, and projection
operations for constrained graph layout.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from .geom import Point
from .rbtree import RBTree
from .vpsc import Constraint, Solver, Variable


class Leaf:
    """Leaf node with bounds and variable."""

    def __init__(self) -> None:
        self.bounds: Optional[Rectangle] = None
        self.variable: Optional[Variable] = None


class ProjectionGroup:
    """Group of nodes for hierarchical projection."""

    def __init__(self) -> None:
        self.bounds: Optional[Rectangle] = None
        self.padding: float = 0.0
        self.stiffness: float = 0.01
        self.leaves: Optional[list[Leaf]] = None
        self.groups: Optional[list[ProjectionGroup]] = None
        self.min_var: Optional[Variable] = None
        self.max_var: Optional[Variable] = None


def compute_group_bounds(g: ProjectionGroup) -> Rectangle:
    """
    Compute bounds for a projection group.

    Args:
        g: Projection group

    Returns:
        Bounding rectangle
    """
    if g.leaves is not None:
        g.bounds = Rectangle.empty()
        for leaf in g.leaves:
            if leaf.bounds is not None:
                g.bounds = leaf.bounds.union(g.bounds)
    else:
        g.bounds = Rectangle.empty()

    if g.groups is not None:
        for group in g.groups:
            g.bounds = compute_group_bounds(group).union(g.bounds)

    g.bounds = g.bounds.inflate(g.padding)
    return g.bounds


class Rectangle:
    """Axis-aligned rectangle."""

    def __init__(self, x: float, X: float, y: float, Y: float):
        """
        Initialize rectangle.

        Args:
            x: Left edge
            X: Right edge
            y: Bottom edge
            Y: Top edge
        """
        self.x = x
        self.X = X
        self.y = y
        self.Y = Y

    @staticmethod
    def empty() -> Rectangle:
        """Create an empty rectangle."""
        inf = float("inf")
        return Rectangle(inf, -inf, inf, -inf)

    def cx(self) -> float:
        """Get x center."""
        return (self.x + self.X) / 2.0

    def cy(self) -> float:
        """Get y center."""
        return (self.y + self.Y) / 2.0

    def overlap_x(self, r: Rectangle) -> float:
        """Get x-axis overlap with another rectangle."""
        ux = self.cx()
        vx = r.cx()
        if ux <= vx and r.x < self.X:
            return self.X - r.x
        if vx <= ux and self.x < r.X:
            return r.X - self.x
        return 0.0

    def overlap_y(self, r: Rectangle) -> float:
        """Get y-axis overlap with another rectangle."""
        uy = self.cy()
        vy = r.cy()
        if uy <= vy and r.y < self.Y:
            return self.Y - r.y
        if vy <= uy and self.y < r.Y:
            return r.Y - self.y
        return 0.0

    def set_x_centre(self, cx: float) -> None:
        """Set x center position."""
        dx = cx - self.cx()
        self.x += dx
        self.X += dx

    def set_y_centre(self, cy: float) -> None:
        """Set y center position."""
        dy = cy - self.cy()
        self.y += dy
        self.Y += dy

    def width(self) -> float:
        """Get width."""
        return self.X - self.x

    def height(self) -> float:
        """Get height."""
        return self.Y - self.y

    def union(self, r: Rectangle) -> Rectangle:
        """Get union with another rectangle."""
        return Rectangle(min(self.x, r.x), max(self.X, r.X), min(self.y, r.y), max(self.Y, r.Y))

    def line_intersections(self, x1: float, y1: float, x2: float, y2: float) -> list[Point]:
        """
        Find intersections between a line and rectangle sides.

        Args:
            x1, y1: First point of line
            x2, y2: Second point of line

        Returns:
            List of intersection points
        """
        sides = [
            [self.x, self.y, self.X, self.y],
            [self.X, self.y, self.X, self.Y],
            [self.X, self.Y, self.x, self.Y],
            [self.x, self.Y, self.x, self.y],
        ]
        intersections = []
        for side in sides:
            r = Rectangle.line_intersection(x1, y1, x2, y2, side[0], side[1], side[2], side[3])
            if r is not None:
                intersections.append(Point(r.x, r.y))
        return intersections

    def ray_intersection(self, x2: float, y2: float) -> Optional[Point]:
        """
        Find intersection between ray from center to point and rectangle sides.

        Args:
            x2, y2: Target point

        Returns:
            First intersection point or None
        """
        ints = self.line_intersections(self.cx(), self.cy(), x2, y2)
        return ints[0] if len(ints) > 0 else None

    def vertices(self) -> list[Point]:
        """Get rectangle vertices."""
        return [
            Point(self.x, self.y),
            Point(self.X, self.y),
            Point(self.X, self.Y),
            Point(self.x, self.Y),
        ]

    @staticmethod
    def line_intersection(
        x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float
    ) -> Optional[Point]:
        """
        Find intersection of two line segments.

        Returns:
            Intersection point or None
        """
        dx12 = x2 - x1
        dx34 = x4 - x3
        dy12 = y2 - y1
        dy34 = y4 - y3
        denominator = dy34 * dx12 - dx34 * dy12

        if denominator == 0:
            return None

        dx31 = x1 - x3
        dy31 = y1 - y3
        numa = dx34 * dy31 - dy34 * dx31
        a = numa / denominator
        numb = dx12 * dy31 - dy12 * dx31
        b = numb / denominator

        if 0 <= a <= 1 and 0 <= b <= 1:
            return Point(x1 + a * dx12, y1 + a * dy12)

        return None

    def inflate(self, pad: float) -> Rectangle:
        """
        Inflate rectangle by padding.

        Args:
            pad: Padding amount

        Returns:
            Inflated rectangle
        """
        return Rectangle(self.x - pad, self.X + pad, self.y - pad, self.Y + pad)


def make_edge_between(source: Rectangle, target: Rectangle, ah: float) -> dict[str, Point]:
    """
    Create edge between two rectangles.

    Args:
        source: Source rectangle
        target: Target rectangle
        ah: Arrow head size

    Returns:
        Dict with sourceIntersection, targetIntersection, arrowStart
    """
    si = source.ray_intersection(target.cx(), target.cy())
    if si is None:
        si = Point(source.cx(), source.cy())

    ti = target.ray_intersection(source.cx(), source.cy())
    if ti is None:
        ti = Point(target.cx(), target.cy())

    dx = ti.x - si.x
    dy = ti.y - si.y
    l = math.sqrt(dx * dx + dy * dy)
    al = l - ah

    return {
        "sourceIntersection": si,
        "targetIntersection": ti,
        "arrowStart": Point(si.x + al * dx / l, si.y + al * dy / l),
    }


def make_edge_to(s: Point, target: Rectangle, ah: float) -> Point:
    """
    Create edge from point to rectangle.

    Args:
        s: Source point
        target: Target rectangle
        ah: Arrow head size

    Returns:
        Arrow start point
    """
    ti = target.ray_intersection(s.x, s.y)
    if ti is None:
        ti = Point(target.cx(), target.cy())

    dx = ti.x - s.x
    dy = ti.y - s.y
    l = math.sqrt(dx * dx + dy * dy)

    return Point(ti.x - ah * dx / l, ti.y - ah * dy / l)


class _Node:
    """Internal node for sweep line algorithm."""

    def __init__(self, v: Variable, r: Rectangle, pos: float):
        self.v = v
        self.r = r
        self.pos = pos
        self.prev: RBTree[_Node] = RBTree(lambda a, b: a.pos - b.pos)
        self.next: RBTree[_Node] = RBTree(lambda a, b: a.pos - b.pos)


class _Event:
    """Event for sweep line algorithm."""

    def __init__(self, is_open: bool, v: _Node, pos: float):
        self.is_open = is_open
        self.v = v
        self.pos = pos


def _compare_events(a: _Event, b: _Event) -> int:
    """Compare events for sorting."""
    if a.pos > b.pos:
        return 1
    if a.pos < b.pos:
        return -1
    if a.is_open:
        return -1
    if b.is_open:
        return 1
    return 0


class _RectAccessors:
    """Accessors for rectangle properties in different orientations."""

    def __init__(
        self,
        get_centre: Callable[[Rectangle], float],
        get_open: Callable[[Rectangle], float],
        get_close: Callable[[Rectangle], float],
        get_size: Callable[[Rectangle], float],
        make_rect: Callable[[float, float, float, float], Rectangle],
        find_neighbours: Callable[[_Node, RBTree[_Node]], None],
    ):
        self.get_centre = get_centre
        self.get_open = get_open
        self.get_close = get_close
        self.get_size = get_size
        self.make_rect = make_rect
        self.find_neighbours = find_neighbours


def _find_x_neighbours(v: _Node, scanline: RBTree[_Node]) -> None:
    """Find x-axis neighbors for a node."""

    def f(forward: str, reverse: str) -> None:
        it = scanline.find_iter(v)
        while True:
            u = getattr(it, forward)()
            if u is None:
                break
            u_over_v_x = u.r.overlap_x(v.r)
            if u_over_v_x <= 0 or u_over_v_x <= u.r.overlap_y(v.r):
                getattr(v, forward).insert(u)
                getattr(u, reverse).insert(v)
            if u_over_v_x <= 0:
                break

    f("next", "prev")
    f("prev", "next")


def _find_y_neighbours(v: _Node, scanline: RBTree[_Node]) -> None:
    """Find y-axis neighbors for a node."""

    def f(forward: str, reverse: str) -> None:
        u = getattr(scanline.find_iter(v), forward)()
        if u is not None and u.r.overlap_x(v.r) > 0:
            getattr(v, forward).insert(u)
            getattr(u, reverse).insert(v)

    f("next", "prev")
    f("prev", "next")


# X-axis accessors
_x_rect = _RectAccessors(
    get_centre=lambda r: r.cx(),
    get_open=lambda r: r.y,
    get_close=lambda r: r.Y,
    get_size=lambda r: r.width(),
    make_rect=lambda open, close, center, size: Rectangle(
        center - size / 2, center + size / 2, open, close
    ),
    find_neighbours=_find_x_neighbours,
)

# Y-axis accessors
_y_rect = _RectAccessors(
    get_centre=lambda r: r.cy(),
    get_open=lambda r: r.x,
    get_close=lambda r: r.X,
    get_size=lambda r: r.height(),
    make_rect=lambda open, close, center, size: Rectangle(
        open, close, center - size / 2, center + size / 2
    ),
    find_neighbours=_find_y_neighbours,
)


def _generate_constraints(
    rs: list[Rectangle], vars: list[Variable], rect: _RectAccessors, min_sep: float
) -> list[Constraint]:
    """Generate non-overlap constraints using sweep line algorithm."""
    n = len(rs)
    N = 2 * n
    assert len(vars) >= n

    events: list[_Event] = []
    for i in range(n):
        r = rs[i]
        v = _Node(vars[i], r, rect.get_centre(r))
        events.append(_Event(True, v, rect.get_open(r)))
        events.append(_Event(False, v, rect.get_close(r)))

    from functools import cmp_to_key

    events.sort(key=cmp_to_key(_compare_events))

    cs: list[Constraint] = []
    scanline: RBTree[_Node] = RBTree(lambda a, b: a.pos - b.pos)

    for e in events:
        v = e.v
        if e.is_open:
            scanline.insert(v)
            rect.find_neighbours(v, scanline)
        else:
            scanline.remove(v)

            def make_constraint(l: _Node, r: _Node) -> None:
                sep = (rect.get_size(l.r) + rect.get_size(r.r)) / 2 + min_sep
                cs.append(Constraint(l.v, r.v, sep))

            def visit_neighbours(forward: str, reverse: str, mkcon: Callable) -> None:
                it = getattr(v, forward).iterator()
                while True:
                    u = getattr(it, forward)()
                    if u is None:
                        break
                    mkcon(u, v)
                    getattr(u, reverse).remove(v)

            visit_neighbours("prev", "next", lambda u, v: make_constraint(u, v))
            visit_neighbours("next", "prev", lambda u, v: make_constraint(v, u))

    assert scanline.size == 0
    return cs


def generate_x_constraints(rs: list[Rectangle], vars: list[Variable]) -> list[Constraint]:
    """Generate x-axis non-overlap constraints."""
    return _generate_constraints(rs, vars, _x_rect, 1e-6)


def generate_y_constraints(rs: list[Rectangle], vars: list[Variable]) -> list[Constraint]:
    """Generate y-axis non-overlap constraints."""
    return _generate_constraints(rs, vars, _y_rect, 1e-6)


def remove_overlaps(rs: list[Rectangle]) -> None:
    """
    Remove overlaps between rectangles.

    Args:
        rs: List of rectangles (modified in place)
    """
    # X-axis projection
    vs = [Variable(r.cx()) for r in rs]
    cs = generate_x_constraints(rs, vs)
    solver = Solver(vs, cs)
    solver.solve()
    for i, v in enumerate(vs):
        rs[i].set_x_centre(v.position())

    # Y-axis projection
    vs = [Variable(r.cy()) for r in rs]
    cs = generate_y_constraints(rs, vs)
    solver = Solver(vs, cs)
    solver.solve()
    for i, v in enumerate(vs):
        rs[i].set_y_centre(v.position())


class GraphNode(Leaf):
    """Graph node for layout."""

    def __init__(self) -> None:
        super().__init__()
        self.fixed: bool = False
        self.fixed_weight: Optional[float] = None
        self.width: float = 0.0
        self.height: float = 0.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.px: float = 0.0
        self.py: float = 0.0


class IndexedVariable(Variable):
    """Variable with index."""

    def __init__(self, index: int, w: float):
        super().__init__(0.0, w)
        self.index = index


class Projection:
    """Projection for constrained graph layout."""

    def __init__(
        self,
        nodes: list[GraphNode],
        groups: list[ProjectionGroup],
        root_group: Optional[ProjectionGroup] = None,
        constraints: Optional[list[Any]] = None,
        avoid_overlaps: bool = False,
    ):
        self.nodes = nodes
        self.groups = groups
        self.root_group = root_group
        self.avoid_overlaps = avoid_overlaps

        self.variables: list[Variable] = []
        for i, v in enumerate(nodes):
            v.variable = IndexedVariable(i, 1.0)
            self.variables.append(v.variable)

        self.x_constraints: list[Constraint] = []
        self.y_constraints: list[Constraint] = []

        if constraints is not None:
            self._create_constraints(constraints)

        if avoid_overlaps and root_group and root_group.groups is not None:
            for v in nodes:
                if not v.width or not v.height:
                    v.bounds = Rectangle(v.x, v.x, v.y, v.y)
                else:
                    w2 = v.width / 2
                    h2 = v.height / 2
                    v.bounds = Rectangle(v.x - w2, v.x + w2, v.y - h2, v.y + h2)

            compute_group_bounds(root_group)

            i = len(nodes)
            for g in groups:
                stiffness = g.stiffness if g.stiffness is not None else 0.01
                g.min_var = IndexedVariable(i, stiffness)
                self.variables.append(g.min_var)
                i += 1
                g.max_var = IndexedVariable(i, stiffness)
                self.variables.append(g.max_var)
                i += 1

    def _create_separation(self, c: dict) -> Constraint:
        """Create separation constraint."""
        return Constraint(
            self.nodes[c["left"]].variable,
            self.nodes[c["right"]].variable,
            c["gap"],
            c.get("equality", False),
        )

    def _make_feasible(self, c: dict) -> None:
        """Make alignment constraint feasible."""
        if not self.avoid_overlaps:
            return

        axis = "x"
        dim = "width"
        if c.get("axis") == "x":
            axis = "y"
            dim = "height"

        vs = [self.nodes[o["node"]] for o in c["offsets"]]
        vs.sort(key=lambda v: getattr(v, axis))

        p = None
        for v in vs:
            if p is not None:
                next_pos = getattr(p, axis) + getattr(p, dim)
                if next_pos > getattr(v, axis):
                    setattr(v, axis, next_pos)
            p = v

    def _create_alignment(self, c: dict) -> None:
        """Create alignment constraint."""
        u = self.nodes[c["offsets"][0]["node"]].variable
        self._make_feasible(c)

        cs = self.x_constraints if c["axis"] == "x" else self.y_constraints
        for o in c["offsets"][1:]:
            v = self.nodes[o["node"]].variable
            cs.append(Constraint(u, v, o["offset"], True))

    def _create_constraints(self, constraints: list[dict[str, Any]]) -> None:
        """Create constraints from specifications."""

        def is_sep(c: dict[str, Any]) -> bool:
            return "type" not in c or c["type"] == "separation"

        self.x_constraints = [
            self._create_separation(c) for c in constraints if c.get("axis") == "x" and is_sep(c)
        ]

        self.y_constraints = [
            self._create_separation(c) for c in constraints if c.get("axis") == "y" and is_sep(c)
        ]

        for c in constraints:
            if c.get("type") == "alignment":
                self._create_alignment(c)

    def project_functions(self) -> list[Callable]:
        """Get projection functions for x and y axes."""
        return [
            lambda x0, y0, x: self.x_project(x0, y0, x),
            lambda x0, y0, y: self.y_project(x0, y0, y),
        ]

    def x_project(self, x0: list[float], y0: list[float], x: list[float]) -> None:
        """Project x coordinates."""
        # Implementation simplified for brevity
        pass

    def y_project(self, x0: list[float], y0: list[float], y: list[float]) -> None:
        """Project y coordinates."""
        # Implementation simplified for brevity
        pass

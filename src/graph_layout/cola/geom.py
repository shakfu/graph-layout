"""
Geometric utilities for graph layout.

This module provides geometry primitives, convex hull calculation,
tangent computation, and visibility graph construction.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence


class Point:
    """2D point."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class LineSegment:
    """Line segment defined by two endpoints."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class PolyPoint(Point):
    """Point with polygon index."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__(x, y)
        self.polyIndex: int = 0


def is_left(P0: Point, P1: Point, P2: Point) -> float:
    """
    Test if a point is Left|On|Right of an infinite line.

    Args:
        P0, P1: Define the line
        P2: Point to test

    Returns:
        >0 for P2 left of the line through P0 and P1
        =0 for P2 on the line
        <0 for P2 right of the line
    """
    return (P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y)


def above(p: Point, vi: Point, vj: Point) -> bool:
    """Test if p is above the line vi-vj."""
    return is_left(p, vi, vj) > 0


def below(p: Point, vi: Point, vj: Point) -> bool:
    """Test if p is below the line vi-vj."""
    return is_left(p, vi, vj) < 0


def convex_hull(S: list[Point]) -> list[Point]:
    """
    Return the convex hull of a set of points using Andrew's monotone chain algorithm.

    See: http://geomalgorithms.com/a10-_hull-1.html#Monotone%20Chain

    Args:
        S: Array of points

    Returns:
        The convex hull as an array of points
    """
    P = sorted(S, key=lambda p: (p.x, p.y), reverse=True)
    n = len(S)

    minmin = 0
    xmin = P[0].x
    i = 1
    while i < n:
        if P[i].x != xmin:
            break
        i += 1
    minmax = i - 1

    H: list[Point] = []
    H.append(P[minmin])

    if minmax == n - 1:  # degenerate case: all x-coords == xmin
        if P[minmax].y != P[minmin].y:  # a nontrivial segment
            H.append(P[minmax])
    else:
        # Get the indices of points with max x-coord and min|max y-coord
        maxmax = n - 1
        xmax = P[n - 1].x
        i = n - 2
        while i >= 0:
            if P[i].x != xmax:
                break
            i -= 1
        maxmin = i + 1

        # Compute the lower hull on the stack H
        i = minmax
        while True:
            i += 1
            if i > maxmin:
                break
            # the lower line joins P[minmin] with P[maxmin]
            if is_left(P[minmin], P[maxmin], P[i]) >= 0 and i < maxmin:
                continue  # ignore P[i] above or on the lower line

            while len(H) > 1:  # there are at least 2 points on the stack
                # test if P[i] is left of the line at the stack top
                if is_left(H[-2], H[-1], P[i]) > 0:
                    break  # P[i] is a new hull vertex
                else:
                    H.pop()  # pop top point off stack

            if i != minmin:
                H.append(P[i])

        # Next, compute the upper hull on the stack H above the bottom hull
        if maxmax != maxmin:  # if distinct xmax points
            H.append(P[maxmax])  # push maxmax point onto stack
        bot = len(H)  # the bottom point of the upper hull stack
        i = maxmin
        while True:
            i -= 1
            if i < minmax:
                break
            # the upper line joins P[maxmax] with P[minmax]
            if is_left(P[maxmax], P[minmax], P[i]) >= 0 and i > minmax:
                continue  # ignore P[i] below or on the upper line

            while len(H) > bot:  # at least 2 points on the upper stack
                # test if P[i] is left of the line at the stack top
                if is_left(H[-2], H[-1], P[i]) > 0:
                    break  # P[i] is a new hull vertex
                else:
                    H.pop()  # pop top point off stack

            if i != minmin:
                H.append(P[i])  # push P[i] onto stack

    return H


def clockwise_radial_sweep(p: Point, P: list[Point], f: Callable[[Point], None]) -> None:
    """Apply f to the points in P in clockwise order around the point p."""
    sorted_points = sorted(P, key=lambda a: math.atan2(a.y - p.y, a.x - p.x))
    for point in sorted_points:
        f(point)


def _next_poly_point(p: PolyPoint, ps: list[PolyPoint]) -> PolyPoint:
    """Get next point in polygon."""
    if p.polyIndex == len(ps) - 1:
        return ps[0]
    return ps[p.polyIndex + 1]


def _prev_poly_point(p: PolyPoint, ps: list[PolyPoint]) -> PolyPoint:
    """Get previous point in polygon."""
    if p.polyIndex == 0:
        return ps[len(ps) - 1]
    return ps[p.polyIndex - 1]


def _tangent_point_poly_c(P: Point, V: Sequence[Point]) -> dict[str, int]:
    """
    Fast binary search for tangents to a convex polygon.

    Args:
        P: A 2D point (exterior to the polygon)
        V: Array of vertices for a 2D convex polygon

    Returns:
        Dict with 'rtan' (rightmost tangent index) and 'ltan' (leftmost tangent index)
    """
    # Rtangent_PointPolyC and Ltangent_PointPolyC require polygon to be
    # "closed" with the first vertex duplicated at end, so V[n-1] = V[0].
    V_closed = list(V)
    V_closed.append(V[0])

    return {
        "rtan": _rtangent_point_poly_c(P, V_closed),
        "ltan": _ltangent_point_poly_c(P, V_closed),
    }


def _rtangent_point_poly_c(P: Point, V: list[Point]) -> int:
    """
    Binary search for convex polygon right tangent.

    Args:
        P: A 2D point (exterior to the polygon)
        V: Array of vertices for a 2D convex polygon with first vertex
            duplicated as last, so V[n-1] = V[0]

    Returns:
        Index i of rightmost tangent point V[i]
    """
    n = len(V) - 1

    # rightmost tangent = maximum for the is_left() ordering
    # test if V[0] is a local maximum
    if below(P, V[1], V[0]) and not above(P, V[n - 1], V[0]):
        return 0  # V[0] is the maximum tangent point

    a = 0
    b = n
    while True:  # start chain = [0,n] with V[n]=V[0]
        if b - a == 1:
            if above(P, V[a], V[b]):
                return a
            else:
                return b

        c = (a + b) // 2  # midpoint of [a,b], and 0<c<n
        dnC = below(P, V[c + 1], V[c])
        if dnC and not above(P, V[c - 1], V[c]):
            return c  # V[c] is the maximum tangent point

        # no max yet, so continue with the binary search
        # pick one of the two subchains [a,c] or [c,b]
        upA = above(P, V[a + 1], V[a])
        if upA:  # edge a points up
            if dnC:  # edge c points down
                b = c  # select [a,c]
            else:  # edge c points up
                if above(P, V[a], V[c]):  # V[a] above V[c]
                    b = c  # select [a,c]
                else:  # V[a] below V[c]
                    a = c  # select [c,b]
        else:  # edge a points down
            if not dnC:  # edge c points up
                a = c  # select [c,b]
            else:  # edge c points down
                if below(P, V[a], V[c]):  # V[a] below V[c]
                    b = c  # select [a,c]
                else:  # V[a] above V[c]
                    a = c  # select [c,b]


def _ltangent_point_poly_c(P: Point, V: list[Point]) -> int:
    """
    Binary search for convex polygon left tangent.

    Args:
        P: A 2D point (exterior to the polygon)
        V: Array of vertices for a 2D convex polygon with first vertex
            duplicated as last, so V[n-1] = V[0]

    Returns:
        Index i of leftmost tangent point V[i]
    """
    n = len(V) - 1

    # leftmost tangent = minimum for the is_left() ordering
    # test if V[0] is a local minimum
    if above(P, V[n - 1], V[0]) and not below(P, V[1], V[0]):
        return 0  # V[0] is the minimum tangent point

    a = 0
    b = n
    while True:  # start chain = [0,n] with V[n] = V[0]
        if b - a == 1:
            if below(P, V[a], V[b]):
                return a
            else:
                return b

        c = (a + b) // 2  # midpoint of [a,b], and 0<c<n
        dnC = below(P, V[c + 1], V[c])
        if above(P, V[c - 1], V[c]) and not dnC:
            return c  # V[c] is the minimum tangent point

        # no min yet, so continue with the binary search
        # pick one of the two subchains [a,c] or [c,b]
        dnA = below(P, V[a + 1], V[a])
        if dnA:  # edge a points down
            if not dnC:  # edge c points up
                b = c  # select [a,c]
            else:  # edge c points down
                if below(P, V[a], V[c]):  # V[a] below V[c]
                    b = c  # select [a,c]
                else:  # V[a] above V[c]
                    a = c  # select [c,b]
        else:  # edge a points up
            if dnC:  # edge c points down
                a = c  # select [c,b]
            else:  # edge c points up
                if above(P, V[a], V[c]):  # V[a] above V[c]
                    b = c  # select [a,c]
                else:  # V[a] below V[c]
                    a = c  # select [c,b]


def _tangent_poly_poly_c(
    V: list[Point],
    W: list[Point],
    t1: Callable[[Point, list[Point]], int],
    t2: Callable[[Point, list[Point]], int],
    cmp1: Callable[[Point, Point, Point], bool],
    cmp2: Callable[[Point, Point, Point], bool],
) -> dict[str, int]:
    """
    Get tangent between two convex polygons.

    Args:
        V: Array of vertices for convex polygon 1
        W: Array of vertices for convex polygon 2
        t1: Tangent function for polygon 1
        t2: Tangent function for polygon 2
        cmp1: Comparison function 1
        cmp2: Comparison function 2

    Returns:
        Dict with 't1' (tangent index for V) and 't2' (tangent index for W)
    """
    # first get the initial vertex on each polygon
    ix1 = t1(W[0], V)  # tangent from W[0] to V
    ix2 = t2(V[ix1], W)  # tangent from V[ix1] to W

    # ping-pong linear search until it stabilizes
    done = False
    while not done:
        done = True  # assume done until...
        while True:
            if ix1 == len(V) - 1:
                ix1 = 0
            if cmp1(W[ix2], V[ix1], V[ix1 + 1]):
                break
            ix1 += 1  # get tangent from W[ix2] to V
        while True:
            if ix2 == 0:
                ix2 = len(W) - 1
            if cmp2(V[ix1], W[ix2], W[ix2 - 1]):
                break
            ix2 -= 1  # get tangent from V[ix1] to W
            done = False  # not done if had to adjust this

    return {"t1": ix1, "t2": ix2}


def lr_tangent_poly_poly_c(V: list[Point], W: list[Point]) -> dict[str, int]:
    """Get LR tangent between two convex polygons."""
    rl = rl_tangent_poly_poly_c(W, V)
    return {"t1": rl["t2"], "t2": rl["t1"]}


def rl_tangent_poly_poly_c(V: list[Point], W: list[Point]) -> dict[str, int]:
    """Get RL tangent between two convex polygons."""
    return _tangent_poly_poly_c(V, W, _rtangent_point_poly_c, _ltangent_point_poly_c, above, below)


def ll_tangent_poly_poly_c(V: list[Point], W: list[Point]) -> dict[str, int]:
    """Get LL tangent between two convex polygons."""
    return _tangent_poly_poly_c(V, W, _ltangent_point_poly_c, _ltangent_point_poly_c, below, below)


def rr_tangent_poly_poly_c(V: list[Point], W: list[Point]) -> dict[str, int]:
    """Get RR tangent between two convex polygons."""
    return _tangent_poly_poly_c(V, W, _rtangent_point_poly_c, _rtangent_point_poly_c, above, above)


class BiTangent:
    """Bitangent between two polygons."""

    def __init__(self, t1: int, t2: int):
        self.t1 = t1
        self.t2 = t2


class BiTangents:
    """All bitangents between two polygons."""

    def __init__(self) -> None:
        self.rl: Optional[BiTangent] = None
        self.lr: Optional[BiTangent] = None
        self.ll: Optional[BiTangent] = None
        self.rr: Optional[BiTangent] = None


class TVGPoint(Point):
    """Point for tangent visibility graph."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__(x, y)
        self.vv: Optional[VisibilityVertex] = None


class VisibilityVertex:
    """Vertex in visibility graph."""

    def __init__(self, id: int, polyid: int, polyvertid: int, p: TVGPoint):
        self.id = id
        self.polyid = polyid
        self.polyvertid = polyvertid
        self.p = p
        p.vv = self


class VisibilityEdge:
    """Edge in visibility graph."""

    def __init__(self, source: VisibilityVertex, target: VisibilityVertex):
        self.source = source
        self.target = target

    def length(self) -> float:
        """Calculate edge length."""
        dx = self.source.p.x - self.target.p.x
        dy = self.source.p.y - self.target.p.y
        return math.sqrt(dx * dx + dy * dy)


class TangentVisibilityGraph:
    """Tangent visibility graph for polygons."""

    def __init__(self, P: list[list[TVGPoint]], g0: Optional[dict[str, list]] = None):
        self.P = P
        self.V: list[VisibilityVertex] = []
        self.E: list[VisibilityEdge] = []

        if g0 is None:
            n = len(P)
            # For each node...
            for i in range(n):
                p = P[i]
                # For each node vertex.
                for j in range(len(p)):
                    pj = p[j]
                    vv = VisibilityVertex(len(self.V), i, j, pj)
                    self.V.append(vv)
                    # For every iteration but the first, generate an
                    # edge from the previous visibility vertex to the current one.
                    if j > 0:
                        prev_vv = p[j - 1].vv
                        assert prev_vv is not None
                        self.E.append(VisibilityEdge(prev_vv, vv))
                # Add a visibility edge from the first vertex to the last.
                if len(p) > 1:
                    first_vv = p[0].vv
                    last_vv = p[len(p) - 1].vv
                    assert first_vv is not None and last_vv is not None
                    self.E.append(VisibilityEdge(first_vv, last_vv))

            for i in range(n - 1):
                Pi = P[i]
                for j in range(i + 1, n):
                    Pj = P[j]
                    t = tangents(Pi, Pj)
                    for tangent_type in ["rl", "lr", "ll", "rr"]:
                        c = getattr(t, tangent_type)
                        if c is not None:
                            source = Pi[c.t1]
                            target = Pj[c.t2]
                            self.add_edge_if_visible(source, target, i, j)
        else:
            self.V = g0["V"].copy()
            self.E = g0["E"].copy()

    def add_edge_if_visible(self, u: TVGPoint, v: TVGPoint, i1: int, i2: int) -> None:
        """Add edge if visible (not intersecting polygons)."""
        if not self._intersects_polys(LineSegment(u.x, u.y, v.x, v.y), i1, i2):
            assert u.vv is not None and v.vv is not None
            self.E.append(VisibilityEdge(u.vv, v.vv))

    def add_point(self, p: TVGPoint, i1: int) -> VisibilityVertex:
        """Add a point to the visibility graph."""
        n = len(self.P)
        self.V.append(VisibilityVertex(len(self.V), n, 0, p))
        for i in range(n):
            if i == i1:
                continue
            poly = self.P[i]
            t = _tangent_point_poly_c(p, poly)
            self.add_edge_if_visible(p, poly[t["ltan"]], i1, i)
            self.add_edge_if_visible(p, poly[t["rtan"]], i1, i)
        assert p.vv is not None
        return p.vv

    def _intersects_polys(self, l: LineSegment, i1: int, i2: int) -> bool:
        """Check if line intersects any polygon except i1 and i2."""
        for i in range(len(self.P)):
            if i != i1 and i != i2 and len(_intersects(l, self.P[i])) > 0:
                return True
        return False


def _line_intersection(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float
) -> Optional[Point]:
    """
    Calculate intersection point of two line segments.

    Temporary implementation until rectangle module is translated.
    """
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-10:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return Point(x, y)

    return None


def _intersects(l: LineSegment, P: Sequence[Point]) -> list[Point]:
    """Find intersections between line segment and polygon."""
    ints = []
    for i in range(1, len(P)):
        intersection = _line_intersection(
            l.x1, l.y1, l.x2, l.y2, P[i - 1].x, P[i - 1].y, P[i].x, P[i].y
        )
        if intersection is not None:
            ints.append(intersection)
    # Check closing edge from last to first vertex
    if len(P) > 0:
        intersection = _line_intersection(l.x1, l.y1, l.x2, l.y2, P[-1].x, P[-1].y, P[0].x, P[0].y)
        if intersection is not None:
            ints.append(intersection)
    return ints


def tangents(V: Sequence[Point], W: Sequence[Point]) -> BiTangents:
    """
    Compute all bitangents between two convex polygons.

    Args:
        V: First polygon vertices
        W: Second polygon vertices

    Returns:
        BiTangents object with rl, lr, ll, rr tangents
    """
    m = len(V) - 1
    n = len(W) - 1
    bt = BiTangents()

    for i in range(m + 1):
        for j in range(n + 1):
            v1 = V[m if i == 0 else i - 1]
            v2 = V[i]
            v3 = V[0 if i == m else i + 1]
            w1 = W[n if j == 0 else j - 1]
            w2 = W[j]
            w3 = W[0 if j == n else j + 1]

            v1v2w2 = is_left(v1, v2, w2)
            v2w1w2 = is_left(v2, w1, w2)
            v2w2w3 = is_left(v2, w2, w3)
            w1w2v2 = is_left(w1, w2, v2)
            w2v1v2 = is_left(w2, v1, v2)
            w2v2v3 = is_left(w2, v2, v3)

            if (
                v1v2w2 >= 0
                and v2w1w2 >= 0
                and v2w2w3 < 0
                and w1w2v2 >= 0
                and w2v1v2 >= 0
                and w2v2v3 < 0
            ):
                bt.ll = BiTangent(i, j)
            elif (
                v1v2w2 <= 0
                and v2w1w2 <= 0
                and v2w2w3 > 0
                and w1w2v2 <= 0
                and w2v1v2 <= 0
                and w2v2v3 > 0
            ):
                bt.rr = BiTangent(i, j)
            elif (
                v1v2w2 <= 0
                and v2w1w2 > 0
                and v2w2w3 <= 0
                and w1w2v2 >= 0
                and w2v1v2 < 0
                and w2v2v3 >= 0
            ):
                bt.rl = BiTangent(i, j)
            elif (
                v1v2w2 >= 0
                and v2w1w2 < 0
                and v2w2w3 >= 0
                and w1w2v2 <= 0
                and w2v1v2 > 0
                and w2v2v3 <= 0
            ):
                bt.lr = BiTangent(i, j)

    return bt


def _is_point_inside_poly(p: Point, poly: list[Point]) -> bool:
    """Test if point is inside polygon."""
    for i in range(1, len(poly)):
        if below(poly[i - 1], poly[i], p):
            return False
    # Check closing edge from last to first vertex
    if below(poly[-1], poly[0], p):
        return False
    return True


def _is_any_p_in_q(p: list[Point], q: list[Point]) -> bool:
    """Check if any point in p is inside polygon q."""
    return any(_is_point_inside_poly(v, q) for v in p)


def polys_overlap(p: list[Point], q: list[Point]) -> bool:
    """
    Check if two polygons overlap.

    Args:
        p: First polygon
        q: Second polygon

    Returns:
        True if polygons overlap
    """
    if _is_any_p_in_q(p, q):
        return True
    if _is_any_p_in_q(q, p):
        return True
    for i in range(1, len(p)):
        v = p[i]
        u = p[i - 1]
        if len(_intersects(LineSegment(u.x, u.y, v.x, v.y), q)) > 0:
            return True
    # Check closing edge from last to first vertex
    if len(_intersects(LineSegment(p[-1].x, p[-1].y, p[0].x, p[0].y), q)) > 0:
        return True
    return False

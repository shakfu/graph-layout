"""
Grid-based orthogonal edge routing.

This module provides algorithms for routing edges on a grid structure,
avoiding obstacles and minimizing crossings.
"""

from __future__ import annotations

import math
from typing import Callable, Generic, Optional, Protocol, TypeVar

from .geom import Point
from .rectangle import Rectangle
from .shortestpaths import Calculator
from .vpsc import Constraint, Solver, Variable

T = TypeVar("T")


class NodeAccessor(Generic[T], Protocol):
    """Interface for accessing node properties."""

    def get_children(self, v: T) -> list[int]:
        """Get children indices for a node."""
        ...

    def get_bounds(self, v: T) -> Rectangle:
        """Get bounding rectangle for a node."""
        ...


class NodeWrapper:
    """Wrapper for nodes in the routing graph."""

    def __init__(self, node_id: int, rect: Rectangle, children: Optional[list[int]] = None):
        self.id = node_id
        self.rect = rect
        self.children = children if children is not None else []
        self.leaf = len(self.children) == 0
        self.parent: Optional[NodeWrapper] = None
        self.ports: list[Vert] = []


class Vert:
    """Vertex in the routing graph."""

    def __init__(
        self, vert_id: int, x: float, y: float, node: Optional[NodeWrapper] = None, line=None
    ):
        self.id = vert_id
        self.x = x
        self.y = y
        self.node = node
        self.line = line


class LongestCommonSubsequence(Generic[T]):
    """Find longest common subsequence between two sequences."""

    def __init__(self, s: list[T], t: list[T]):
        self.s = s
        self.t = t

        mf = self._find_match(s, t)
        tr = t[::-1]
        mr = self._find_match(s, tr)

        if mf["length"] >= mr["length"]:
            self.length = mf["length"]
            self.si = mf["si"]
            self.ti = mf["ti"]
            self.reversed = False
        else:
            self.length = mr["length"]
            self.si = mr["si"]
            self.ti = len(t) - mr["ti"] - mr["length"]
            self.reversed = True

    @staticmethod
    def _find_match(s: list[T], t: list[T]) -> dict:
        """Find longest common substring."""
        m = len(s)
        n = len(t)
        match = {"length": 0, "si": -1, "ti": -1}
        l = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    v = 1 if (i == 0 or j == 0) else l[i - 1][j - 1] + 1
                    l[i][j] = v
                    if v > match["length"]:
                        match["length"] = v
                        match["si"] = i - v + 1
                        match["ti"] = j - v + 1
                else:
                    l[i][j] = 0

        return match

    def get_sequence(self) -> list[T]:
        """Get the common subsequence."""
        if self.length >= 0:
            return self.s[self.si : self.si + self.length]
        return []


class GridLine:
    """A horizontal or vertical line of nodes."""

    def __init__(self, nodes: list[NodeWrapper], pos: float):
        self.nodes = nodes
        self.pos = pos


class GridRouter(Generic[T]):
    """Grid-based orthogonal edge router."""

    def __init__(
        self, originalnodes: list[T], accessor: NodeAccessor[T], group_padding: float = 12
    ):
        self.originalnodes = originalnodes
        self.group_padding = group_padding

        # Create node wrappers
        self.nodes = [
            NodeWrapper(i, accessor.get_bounds(v), accessor.get_children(v))
            for i, v in enumerate(originalnodes)
        ]

        self.leaves = [v for v in self.nodes if v.leaf]
        self.groups = [g for g in self.nodes if not g.leaf]

        self.cols = self._get_grid_lines("x")
        self.rows = self._get_grid_lines("y")

        # Create parent relationships
        for v in self.groups:
            for c in v.children:
                self.nodes[c].parent = v

        # Root claims orphans
        self.root = type("Root", (), {"children": []})()
        for v in self.nodes:
            if v.parent is None:
                v.parent = self.root
                self.root.children.append(v.id)

        # Order nodes by depth
        self.back_to_front = sorted(self.nodes, key=lambda x: self._get_depth(x))

        # Compute group boundaries (front to back)
        front_to_back_groups = [g for g in reversed(self.back_to_front) if not g.leaf]
        for v in front_to_back_groups:
            r = Rectangle.empty()
            for c in v.children:
                r = r.union(self.nodes[c].rect)
            v.rect = r.inflate(self.group_padding)

        # Create grid
        col_mids = self._mid_points([r.pos for r in self.cols])
        row_mids = self._mid_points([r.pos for r in self.rows])

        rowx = col_mids[0]
        rowX = col_mids[-1]
        coly = row_mids[0]
        colY = row_mids[-1]

        # Horizontal lines
        hlines = [
            {"x1": rowx, "x2": rowX, "y1": r.pos, "y2": r.pos, "verts": []} for r in self.rows
        ] + [{"x1": rowx, "x2": rowX, "y1": m, "y2": m, "verts": []} for m in row_mids]

        # Vertical lines
        vlines = [
            {"x1": c.pos, "x2": c.pos, "y1": coly, "y2": colY, "verts": []} for c in self.cols
        ] + [{"x1": m, "x2": m, "y1": coly, "y2": colY, "verts": []} for m in col_mids]

        lines = hlines + vlines

        # Create routing graph
        self.verts: list[Vert] = []
        self.edges: list[dict] = []

        # Create vertices at line intersections
        for h in hlines:
            for v in vlines:
                p = Vert(len(self.verts), v["x1"], h["y1"])
                h["verts"].append(p)
                v["verts"].append(p)
                self.verts.append(p)

                # Assign to nodes
                for node in reversed(self.back_to_front):
                    r = node.rect
                    dx = abs(p.x - r.cx())
                    dy = abs(p.y - r.cy())
                    if dx < r.width() / 2 and dy < r.height() / 2:
                        p.node = node
                        break

        # Create vertices at node-line intersections
        for li, l in enumerate(lines):
            for v in self.nodes:
                intersections = v.rect.line_intersections(l["x1"], l["y1"], l["x2"], l["y2"])
                for intersect in intersections:
                    p = Vert(len(self.verts), intersect.x, intersect.y, v, l)
                    self.verts.append(p)
                    l["verts"].append(p)
                    v.ports.append(p)

            # Create edges along lines
            is_horiz = abs(l["y1"] - l["y2"]) < 0.1

            def delta(a, b):
                return b.x - a.x if is_horiz else b.y - a.y

            l["verts"].sort(key=lambda a: (a.x if is_horiz else a.y))

            for i in range(1, len(l["verts"])):
                u = l["verts"][i - 1]
                v_vert = l["verts"][i]
                # Skip edges within same leaf node
                if u.node and u.node == v_vert.node and u.node.leaf:
                    continue
                self.edges.append(
                    {"source": u.id, "target": v_vert.id, "length": abs(delta(u, v_vert))}
                )

        # Ensure all nodes have at least one port (center point)
        for v in self.nodes:
            if not v.ports:
                center_port = Vert(len(self.verts), v.rect.cx(), v.rect.cy(), v)
                self.verts.append(center_port)
                v.ports.append(center_port)

        self.obstacles = []
        self.passable_edges = []

    def _avg(self, a: list[float]) -> float:
        """Calculate average."""
        return sum(a) / len(a) if a else 0

    def _get_grid_lines(self, axis: str) -> list[GridLine]:
        """Find overlapping sets of leaves in the given axis."""
        columns = []
        ls = self.leaves[:]

        while ls:
            # Find overlapping nodes
            first = ls[0]
            overlapping = [v for v in ls if getattr(v.rect, f"overlap_{axis}")(first.rect)]

            # Safeguard: if nothing overlaps, at least take the first node
            if not overlapping:
                overlapping = [first]

            pos = self._avg([getattr(v.rect, f"c{axis}")() for v in overlapping])
            col = GridLine(overlapping, pos)
            columns.append(col)

            for v in col.nodes:
                if v in ls:
                    ls.remove(v)

        columns.sort(key=lambda a: a.pos)
        return columns

    def _get_depth(self, v: NodeWrapper) -> int:
        """Get depth of node in hierarchy."""
        depth = 0
        while v.parent != self.root:
            depth += 1
            v = v.parent
        return depth

    def _mid_points(self, a: list[float]) -> list[float]:
        """Calculate midpoints for grid boundaries."""
        if len(a) == 1:
            return [a[0]]

        gap = a[1] - a[0]
        mids = [a[0] - gap / 2]
        for i in range(1, len(a)):
            mids.append((a[i] + a[i - 1]) / 2)
        mids.append(a[-1] + gap / 2)
        return mids

    def _find_lineage(self, v: NodeWrapper) -> list[NodeWrapper]:
        """Find path from node to root."""
        lineage = [v]
        while v != self.root:
            v = v.parent
            lineage.append(v)
        return list(reversed(lineage))

    def _find_ancestor_path_between(self, a: NodeWrapper, b: NodeWrapper) -> dict:
        """Find path through lowest common ancestor."""
        aa = self._find_lineage(a)
        ba = self._find_lineage(b)
        i = 0
        while i < len(aa) and i < len(ba) and aa[i] == ba[i]:
            i += 1
        return {"commonAncestor": aa[i - 1], "lineages": aa[i:] + ba[i:]}

    def sibling_obstacles(self, a: NodeWrapper, b: NodeWrapper) -> list[NodeWrapper]:
        """Find sibling obstacles between two nodes."""
        path = self._find_ancestor_path_between(a, b)
        lineage_lookup = {v.id: True for v in path["lineages"]}

        obstacles = [v for v in path["commonAncestor"].children if v not in lineage_lookup]

        for v in path["lineages"]:
            if v.parent != path["commonAncestor"]:
                obstacles.extend([c for c in v.parent.children if c != v.id])

        return [self.nodes[v] for v in obstacles]

    def route(self, s: int, t: int) -> list[Point]:
        """Find route between two nodes."""
        source = self.nodes[s]
        target = self.nodes[t]
        self.obstacles = self.sibling_obstacles(source, target)

        obstacle_lookup = {o.id: o for o in self.obstacles}

        self.passable_edges = [
            e
            for e in self.edges
            if not (
                (
                    self.verts[e["source"]].node
                    and self.verts[e["source"]].node.id in obstacle_lookup
                )
                or (
                    self.verts[e["target"]].node
                    and self.verts[e["target"]].node.id in obstacle_lookup
                )
            )
        ]

        # Add dummy edges within source
        for i in range(1, len(source.ports)):
            self.passable_edges.append(
                {"source": source.ports[0].id, "target": source.ports[i].id, "length": 0}
            )

        # Add dummy edges within target
        for i in range(1, len(target.ports)):
            self.passable_edges.append(
                {"source": target.ports[0].id, "target": target.ports[i].id, "length": 0}
            )

        calc = Calculator(
            len(self.verts),
            self.passable_edges,
            lambda e: e["source"],
            lambda e: e["target"],
            lambda e: e["length"],
        )

        def bend_penalty(u: int, v: int, w: int) -> float:
            a = self.verts[u]
            b = self.verts[v]
            c = self.verts[w]
            dx = abs(c.x - a.x)
            dy = abs(c.y - a.y)

            # Don't count bends from internal edges
            if (a.node == source and a.node == b.node) or (b.node == target and b.node == c.node):
                return 0

            return 1000 if dx > 1 and dy > 1 else 0

        shortest_path = calc.path_from_node_to_node_with_prev_cost(
            source.ports[0].id, target.ports[0].id, bend_penalty
        )

        # Reverse and add target port
        path_points = [self.verts[vi] for vi in reversed(shortest_path)]
        path_points.append(self.nodes[target.id].ports[0])

        # Filter internal points
        return [
            v
            for i, v in enumerate(path_points)
            if not (
                (
                    i < len(path_points) - 1
                    and path_points[i + 1].node == source
                    and v.node == source
                )
                or (i > 0 and v.node == target and path_points[i - 1].node == target)
            )
        ]

    @staticmethod
    def make_segments(path: list[Point]) -> list[list[Point]]:
        """Create segments from path, merging straight sections."""

        def copy_point(p: Point) -> Point:
            return Point(p.x, p.y)

        def is_straight(a: Point, b: Point, c: Point) -> bool:
            return abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) < 0.001

        segments = []
        a = copy_point(path[0])

        for i in range(1, len(path)):
            b = copy_point(path[i])
            c = path[i + 1] if i < len(path) - 1 else None

            if not c or not is_straight(a, b, c):
                segments.append([a, b])
                a = b

        return segments

    @staticmethod
    def get_segment_sets(routes: list, x: str, y: str) -> list[dict]:
        """Group segments by position in given axis."""
        vsegments = []

        for ei, route in enumerate(routes):
            for si, s in enumerate(route):
                # Wrap segment with metadata
                seg_with_meta = {
                    "segment": s,
                    "edgeid": ei,
                    "i": si,
                    0: s[0],  # for compatibility
                    1: s[1],
                }
                sdx = s[1][x] - s[0][x]
                if abs(sdx) < 0.1:
                    vsegments.append(seg_with_meta)

        vsegments.sort(key=lambda a: a[0][x])

        # Group by position
        vsegmentsets = []
        segmentset = None

        for s in vsegments:
            if not segmentset or abs(s[0][x] - segmentset["pos"]) > 0.1:
                segmentset = {"pos": s[0][x], "segments": []}
                vsegmentsets.append(segmentset)
            segmentset["segments"].append(s)

        return vsegmentsets

    @staticmethod
    def nudge_segs(x: str, y: str, routes: list, segments: list, left_of: Callable, gap: float):
        """Nudge segments apart using VPSC."""
        n = len(segments)
        if n <= 1:
            return

        vs = [Variable(s[0][x]) for s in segments]
        cs = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                s1 = segments[i]
                s2 = segments[j]
                e1 = s1["edgeid"]
                e2 = s2["edgeid"]
                lind = -1
                rind = -1

                if x == "x":
                    if left_of(e1, e2):
                        if s1[0][y] < s1[1][y]:
                            lind, rind = j, i
                        else:
                            lind, rind = i, j
                else:
                    if left_of(e1, e2):
                        if s1[0][y] < s1[1][y]:
                            lind, rind = i, j
                        else:
                            lind, rind = j, i

                if lind >= 0:
                    cs.append(Constraint(vs[lind], vs[rind], gap))

        solver = Solver(vs, cs)
        solver.solve()

        for i, v in enumerate(vs):
            s = segments[i]
            pos = v.position()
            # Update the wrapped segment
            seg = s["segment"] if "segment" in s else s
            seg[0][x] = seg[1][x] = pos
            # Also update the references in s
            s[0][x] = s[1][x] = pos

            route = routes[s["edgeid"]]
            if s["i"] > 0:
                route[s["i"] - 1][1][x] = pos
            if s["i"] < len(route) - 1:
                route[s["i"] + 1][0][x] = pos

    @staticmethod
    def nudge_segments(routes: list, x: str, y: str, left_of: Callable, gap: float):
        """Nudge all overlapping segment bundles."""
        vsegmentsets = GridRouter.get_segment_sets(routes, x, y)

        for ss in vsegmentsets:
            events = []
            for s in ss["segments"]:
                events.append({"type": 0, "s": s, "pos": min(s[0][y], s[1][y])})
                events.append({"type": 1, "s": s, "pos": max(s[0][y], s[1][y])})

            events.sort(key=lambda a: (a["pos"], a["type"]))

            open_segs = []
            open_count = 0

            for e in events:
                if e["type"] == 0:
                    open_segs.append(e["s"])
                    open_count += 1
                else:
                    open_count -= 1

                if open_count == 0:
                    GridRouter.nudge_segs(x, y, routes, open_segs, left_of, gap)
                    open_segs = []

    @staticmethod
    def is_left(a: Point, b: Point, c: Point) -> bool:
        """Check if path a-b-c makes left turn."""
        return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) <= 0

    @staticmethod
    def get_order(pairs: list[dict]) -> Callable:
        """Create lookup function for edge ordering."""
        outgoing = {}
        for p in pairs:
            if p["l"] not in outgoing:
                outgoing[p["l"]] = {}
            outgoing[p["l"]][p["r"]] = True

        return lambda l, r: l in outgoing and r in outgoing[l]

    @staticmethod
    def order_edges(edges: list) -> Callable:
        """Determine ordering to minimize crossings."""
        edge_order = []

        for i in range(len(edges) - 1):
            for j in range(i + 1, len(edges)):
                e = edges[i]
                f = edges[j]
                lcs = LongestCommonSubsequence(e, f)

                if lcs.length == 0:
                    continue

                if lcs.reversed:
                    f.reverse()
                    f.reversed = True
                    lcs = LongestCommonSubsequence(e, f)

                if (lcs.si <= 0 or lcs.ti <= 0) and (
                    lcs.si + lcs.length >= len(e) or lcs.ti + lcs.length >= len(f)
                ):
                    edge_order.append({"l": i, "r": j})
                    continue

                if lcs.si + lcs.length >= len(e) or lcs.ti + lcs.length >= len(f):
                    u = e[lcs.si + 1]
                    vj = e[lcs.si - 1]
                    vi = f[lcs.ti - 1]
                else:
                    u = e[lcs.si + lcs.length - 2]
                    vi = e[lcs.si + lcs.length]
                    vj = f[lcs.ti + lcs.length]

                if GridRouter.is_left(u, vi, vj):
                    edge_order.append({"l": j, "r": i})
                else:
                    edge_order.append({"l": i, "r": j})

        return GridRouter.get_order(edge_order)

    @staticmethod
    def unreverse_edges(routes: list, route_paths: list):
        """Restore original edge direction."""
        for i, segments in enumerate(routes):
            path = route_paths[i]
            if hasattr(path, "reversed") and path.reversed:
                segments.reverse()
                for segment in segments:
                    segment.reverse()

    def route_edges(
        self,
        edges: list[T],
        nudge_gap: float,
        source: Callable[[T], int],
        target: Callable[[T], int],
    ) -> list[list[list[Point]]]:
        """Route edges with nudging to minimize crossings."""
        route_paths = [self.route(source(e), target(e)) for e in edges]
        order = GridRouter.order_edges(route_paths)
        routes = [GridRouter.make_segments(e) for e in route_paths]

        # Convert to dicts for nudging
        routes_dicts = []
        for route in routes:
            route_dict = []
            for seg in route:
                route_dict.append([{"x": seg[0].x, "y": seg[0].y}, {"x": seg[1].x, "y": seg[1].y}])
            routes_dicts.append(route_dict)

        GridRouter.nudge_segments(routes_dicts, "x", "y", order, nudge_gap)
        GridRouter.nudge_segments(routes_dicts, "y", "x", order, nudge_gap)

        # Convert back to Points
        routes_points = []
        for route_dict in routes_dicts:
            route_points = []
            for seg_dict in route_dict:
                route_points.append(
                    [
                        Point(seg_dict[0]["x"], seg_dict[0]["y"]),
                        Point(seg_dict[1]["x"], seg_dict[1]["y"]),
                    ]
                )
            routes_points.append(route_points)

        GridRouter.unreverse_edges(routes_points, route_paths)
        return routes_points

    @staticmethod
    def angle_between_2_lines(line1: list[Point], line2: list[Point]) -> float:
        """Calculate angle between two lines."""
        angle1 = math.atan2(line1[0].y - line1[1].y, line1[0].x - line1[1].x)
        angle2 = math.atan2(line2[0].y - line2[1].y, line2[0].x - line2[1].x)
        diff = angle1 - angle2
        if diff > math.pi or diff < -math.pi:
            diff = angle2 - angle1
        return diff

    @staticmethod
    def get_route_path(
        route: list[list[Point]], cornerradius: float, arrowwidth: float, arrowheight: float
    ) -> dict:
        """Generate SVG path with rounded corners and arrow."""
        result = {"routepath": f"M {route[0][0].x} {route[0][0].y} ", "arrowpath": ""}

        if len(route) > 1:
            for i, li in enumerate(route):
                x, y = li[1].x, li[1].y
                dx = x - li[0].x
                dy = y - li[0].y

                if i < len(route) - 1:
                    if abs(dx) > 0:
                        x -= dx / abs(dx) * cornerradius
                    else:
                        y -= dy / abs(dy) * cornerradius

                    result["routepath"] += f"L {x} {y} "

                    l = route[i + 1]
                    x0, y0 = l[0].x, l[0].y
                    x1, y1 = l[1].x, l[1].y
                    dx = x1 - x0
                    dy = y1 - y0

                    angle = 1 if GridRouter.angle_between_2_lines(li, l) < 0 else 0

                    if abs(dx) > 0:
                        x2 = x0 + dx / abs(dx) * cornerradius
                        y2 = y0
                    else:
                        x2 = x0
                        y2 = y0 + dy / abs(dy) * cornerradius

                    cx = abs(x2 - x)
                    cy = abs(y2 - y)
                    result["routepath"] += f"A {cx} {cy} 0 0 {angle} {x2} {y2} "
                else:
                    arrowtip = [x, y]
                    if abs(dx) > 0:
                        x -= dx / abs(dx) * arrowheight
                        arrowcorner1 = [x, y + arrowwidth]
                        arrowcorner2 = [x, y - arrowwidth]
                    else:
                        y -= dy / abs(dy) * arrowheight
                        arrowcorner1 = [x + arrowwidth, y]
                        arrowcorner2 = [x - arrowwidth, y]

                    result["routepath"] += f"L {x} {y} "
                    if arrowheight > 0:
                        result["arrowpath"] = (
                            f"M {arrowtip[0]} {arrowtip[1]} "
                            f"L {arrowcorner1[0]} {arrowcorner1[1]} "
                            f"L {arrowcorner2[0]} {arrowcorner2[1]}"
                        )
        else:
            li = route[0]
            x, y = li[1].x, li[1].y
            dx = x - li[0].x
            dy = y - li[0].y
            arrowtip = [x, y]

            if abs(dx) > 0:
                x -= dx / abs(dx) * arrowheight
                arrowcorner1 = [x, y + arrowwidth]
                arrowcorner2 = [x, y - arrowwidth]
            else:
                y -= dy / abs(dy) * arrowheight
                arrowcorner1 = [x + arrowwidth, y]
                arrowcorner2 = [x - arrowwidth, y]

            result["routepath"] += f"L {x} {y} "
            if arrowheight > 0:
                result["arrowpath"] = (
                    f"M {arrowtip[0]} {arrowtip[1]} "
                    f"L {arrowcorner1[0]} {arrowcorner1[1]} "
                    f"L {arrowcorner2[0]} {arrowcorner2[1]}"
                )

        return result

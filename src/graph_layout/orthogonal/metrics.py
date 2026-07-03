"""Topology-Shape-Metrics: the "metrics" (shape realization) stage.

The orthogonalization phase produces an :class:`OrthogonalRepresentation` -- a
*combinatorial* description of an orthogonal drawing: the angle at every
vertex-face incidence and the turn sequence (bends) along every edge. It carries
no coordinates. This module turns that representation into an actual orthogonal
shape by assigning a compass direction (East/North/West/South) to every segment
of every edge, which is the prerequisite for coordinate assignment.

Direction convention (counter-clockwise, in quarter turns):

    0 = +x (East), 1 = +y (North), 2 = -x (West), 3 = -y (South)

The shape is derived by propagating directions around each face. Traversing a
face with its interior on the left, the direction leaving a corner of interior
angle ``a`` (in units of 90 degrees) is the arrival direction plus ``2 - a``
quarter turns; each bend along an edge adds its own +/-1 quarter turn. For a
valid orthogonal representation these turns are globally consistent and the
turns around any bounded face sum to +4 (the outer face to -4).

This module implements only the shape step. Coordinate assignment and wiring
into the GIOTTO / Kandinsky layouts build on the ``EdgeShape`` results here.

Scope: the orthogonalization flow model (and hence a realizable shape) is
defined for biconnected planar graphs of maximum degree 4. Outside that domain
the representation is not a valid orthogonal shape -- a vertex of degree > 4
makes the flow infeasible (it needs the Kandinsky 0-degree-angle model), and
bridges / cut vertices give faces with repeated vertices. In those cases the
representation's face turns do not sum to +/-4, ``compute_orthogonal_shape``
returns ``valid=False``, and callers should fall back to a heuristic router.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .orthogonalization import Face, OrthogonalRepresentation

# Direction constants (counter-clockwise quarter turns).
EAST, NORTH, WEST, SOUTH = 0, 1, 2, 3

# Unit step per direction, for turning a direction sequence into geometry later.
DIRECTION_STEP: dict[int, tuple[int, int]] = {
    EAST: (1, 0),
    NORTH: (0, 1),
    WEST: (-1, 0),
    SOUTH: (0, -1),
}


@dataclass
class EdgeShape:
    """Shape of one directed edge (dart).

    ``segments`` is the ordered list of compass directions of the straight
    pieces between successive bends: a straight edge has one segment, an edge
    with ``k`` bends has ``k + 1`` segments. ``start_direction`` is the
    direction leaving the tail vertex (``segments[0]``) and ``end_direction`` is
    the direction arriving at the head vertex (``segments[-1]``).
    """

    tail: int
    head: int
    segments: list[int]

    @property
    def start_direction(self) -> int:
        return self.segments[0]

    @property
    def end_direction(self) -> int:
        return self.segments[-1]


@dataclass
class ShapeResult:
    """Result of the shape computation.

    ``valid`` is False when the representation could not be realized as a
    consistent orthogonal shape (e.g. a degenerate or inconsistent rep, or a
    cut vertex whose per-corner angles collide in ``vertex_face_angles``); in
    that case callers should fall back to a heuristic router.
    """

    valid: bool
    edge_shapes: dict[tuple[int, int], EdgeShape] = field(default_factory=dict)
    reason: str = ""


def _face_of_each_dart(faces: list[Face]) -> dict[tuple[int, int], int]:
    """Map every directed edge (dart) to the index of the face it borders."""
    dart_face: dict[tuple[int, int], int] = {}
    for face in faces:
        for dart in face.edges:
            dart_face[dart] = face.index
    return dart_face


def _bend_turns(rep: OrthogonalRepresentation, dart: tuple[int, int]) -> list[int]:
    """Turn (+1 left / -1 right) contributed by each bend along ``dart``."""
    return list(rep.edge_bends.get(dart, []))


def compute_orthogonal_shape(
    faces: list[Face],
    rep: OrthogonalRepresentation,
) -> ShapeResult:
    """Assign compass directions to every edge segment from the representation.

    Args:
        faces: Faces of the embedding (each with its cyclic dart list in
            ``edges``), as produced by :func:`compute_faces`.
        rep: The orthogonal representation (angles + bends).

    Returns:
        A :class:`ShapeResult`. When ``valid`` is True, ``edge_shapes`` contains
        an :class:`EdgeShape` for every dart (both directions of every edge).
    """
    if not faces:
        return ShapeResult(valid=False, reason="no faces")

    dart_face = _face_of_each_dart(faces)
    faces_by_index = {f.index: f for f in faces}

    # Direction leaving the tail of each dart (segments[0]).
    start_dir: dict[tuple[int, int], int] = {}

    def arrival_direction(dart: tuple[int, int]) -> int:
        """Direction in which ``dart`` arrives at its head vertex."""
        d = start_dir[dart]
        for turn in _bend_turns(rep, dart):
            d = (d + turn) % 4
        return d

    def propagate_face(face: Face) -> bool:
        """Assign directions around ``face`` given one already-fixed dart.

        Returns False if the representation is inconsistent (the turns do not
        close up), signalling an invalid shape.
        """
        darts = face.edges
        k = len(darts)
        if k == 0:
            return True
        # Find a starting dart in this face whose direction is already known.
        start_idx = next((i for i, d in enumerate(darts) if d in start_dir), None)
        if start_idx is None:
            return False

        for step in range(k):
            i = (start_idx + step) % k
            cur = darts[i]
            nxt = darts[(i + 1) % k]
            corner_vertex = cur[1]  # cur = (a, b), nxt = (b, c); corner at b
            angle = rep.vertex_face_angles.get((corner_vertex, face.index))
            if angle is None:
                return False
            arr = arrival_direction(cur)
            new_dir = (arr + 2 - angle) % 4
            if nxt in start_dir:
                # Already assigned (closes the loop): must agree.
                if start_dir[nxt] != new_dir:
                    return False
            else:
                start_dir[nxt] = new_dir
        return True

    # Seed: pick the first dart of the first face heading East, then propagate
    # across the whole map. Adjacent faces are reached through reverse darts:
    # the reverse of a dart leaves the shared vertex opposite to the dart's
    # arrival direction.
    seed_face = faces[0]
    if not seed_face.edges:
        return ShapeResult(valid=False, reason="empty seed face")
    start_dir[seed_face.edges[0]] = EAST

    queue: deque[int] = deque([seed_face.index])
    processed: set[int] = set()

    while queue:
        fi = queue.popleft()
        if fi in processed:
            continue
        face = faces_by_index[fi]
        if not propagate_face(face):
            return ShapeResult(valid=False, reason=f"inconsistent turns at face {fi}")
        processed.add(fi)

        # Push neighbouring faces, seeding their shared reverse dart.
        for dart in face.edges:
            u, v = dart
            rev = (v, u)
            if rev not in start_dir:
                start_dir[rev] = (arrival_direction(dart) + 2) % 4
            rev_face = dart_face.get(rev)
            if rev_face is not None and rev_face not in processed:
                queue.append(rev_face)

    # Build per-dart segment direction sequences.
    edge_shapes: dict[tuple[int, int], EdgeShape] = {}
    for dart, d0 in start_dir.items():
        segments = [d0]
        d = d0
        for turn in _bend_turns(rep, dart):
            d = (d + turn) % 4
            segments.append(d)
        edge_shapes[dart] = EdgeShape(tail=dart[0], head=dart[1], segments=segments)

    return ShapeResult(valid=True, edge_shapes=edge_shapes)


@dataclass
class DrawingResult:
    """Integer coordinates realizing a shape.

    ``vertex_positions`` maps each vertex to its ``(x, y)`` grid point;
    ``edge_routes`` maps each undirected edge ``(min(u, v), max(u, v))`` to the
    ordered polyline (vertex -> bends... -> vertex) of grid points. ``valid`` is
    False if the shape was invalid or the coordinate constraints were
    contradictory (which should not happen for a valid shape).
    """

    valid: bool
    vertex_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    edge_routes: dict[tuple[int, int], list[tuple[int, int]]] = field(default_factory=dict)
    reason: str = ""


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _assign_axis(
    n_points: int,
    segments: list[tuple[int, int, int]],
    horizontal: bool,
    spread: bool = False,
) -> Optional[dict[int, int]]:
    """Assign one integer coordinate per point.

    For the x axis (``horizontal=True``): vertical segments (N/S) force equal x
    on their endpoints; horizontal segments (E/W) order the resulting classes
    (East increases x). The y axis is symmetric. Returns None on a contradiction
    (two points forced equal yet ordered) or a cycle (not possible for a valid
    orthogonal shape).

    ``spread=False`` packs coordinates with longest-path (compact). ``spread=True``
    gives every class a distinct coordinate by topological rank, which separates
    independent classes that longest-path would collapse to the same value -- a
    cheap partial substitute for face rectangularization.
    """
    equal_dirs = (NORTH, SOUTH) if horizontal else (EAST, WEST)
    order_dirs = (EAST, WEST) if horizontal else (NORTH, SOUTH)
    increasing = EAST if horizontal else NORTH

    uf = _UnionFind(n_points)
    for a, b, d in segments:
        if d in equal_dirs:
            uf.union(a, b)

    classes = {uf.find(i) for i in range(n_points)}
    adj: dict[int, set[int]] = {c: set() for c in classes}
    indeg: dict[int, int] = {c: 0 for c in classes}
    for a, b, d in segments:
        if d not in order_dirs:
            continue
        ca, cb = uf.find(a), uf.find(b)
        # Edge points from the smaller coordinate to the larger.
        lo, hi = (ca, cb) if d == increasing else (cb, ca)
        if lo == hi:
            return None  # equal class but ordered -> contradiction
        if hi not in adj[lo]:
            adj[lo].add(hi)
            indeg[hi] += 1

    coord = {c: 0 for c in classes}
    queue = deque(sorted(c for c in classes if indeg[c] == 0))
    seen = 0
    next_rank = 0
    while queue:
        c = queue.popleft()
        seen += 1
        if spread:
            # Distinct coordinate per class, in topological order.
            coord[c] = max(coord[c], next_rank)
            next_rank = coord[c] + 1
        for nb in adj[c]:
            if coord[nb] < coord[c] + 1:
                coord[nb] = coord[c] + 1
            indeg[nb] -= 1
            if indeg[nb] == 0:
                queue.append(nb)
    if seen != len(classes):
        return None  # cycle
    return {i: coord[uf.find(i)] for i in range(n_points)}


def compute_coordinates(shape: ShapeResult, edges: list[tuple[int, int]]) -> DrawingResult:
    """Assign integer coordinates realizing ``shape``.

    Builds a point per vertex and per bend, records each segment's direction,
    and solves the two axes independently by longest-path in the horizontal and
    vertical constraint graphs (minimum segment length 1). The result is an
    orthogonal drawing in which every segment is axis-aligned in its shape
    direction.

    Two coordinate assignments are tried and the first clean one is returned:
    compact longest-path (smallest drawing), then a "spread" assignment that
    gives every coordinate class a distinct value (separating independent
    features that longest-path collapses). If neither is clean -- overlaps,
    crossings, or an edge through a vertex remain, which full face
    rectangularization would resolve -- the result is invalid so callers fall
    back to the heuristic router.
    """
    if not shape.valid:
        return DrawingResult(valid=False, reason="invalid shape")

    # One canonical dart per undirected edge.
    darts: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if (u, v) in shape.edge_shapes:
            darts[key] = (u, v)
        elif (v, u) in shape.edge_shapes:
            darts[key] = (v, u)

    point_id: dict[object, int] = {}

    def pid(pkey: object) -> int:
        idx = point_id.get(pkey)
        if idx is None:
            idx = len(point_id)
            point_id[pkey] = idx
        return idx

    segments: list[tuple[int, int, int]] = []
    edge_chain: dict[tuple[int, int], list[object]] = {}
    vertices: set[int] = set()

    for ekey, dart in darts.items():
        es = shape.edge_shapes[dart]
        tail, head = dart
        vertices.add(tail)
        vertices.add(head)
        n_bends = len(es.segments) - 1
        chain: list[object] = [("v", tail)]
        for i in range(1, n_bends + 1):
            chain.append(("b", dart, i))
        chain.append(("v", head))
        for i, direction in enumerate(es.segments):
            segments.append((pid(chain[i]), pid(chain[i + 1]), direction))
        edge_chain[ekey] = chain

    n_points = len(point_id)

    last_reason = "contradictory coordinate constraints"
    for spread in (False, True):
        xs = _assign_axis(n_points, segments, horizontal=True, spread=spread)
        ys = _assign_axis(n_points, segments, horizontal=False, spread=spread)
        if xs is None or ys is None:
            continue

        vertex_positions = {v: (xs[pid(("v", v))], ys[pid(("v", v))]) for v in vertices}
        if len(set(vertex_positions.values())) != len(vertex_positions):
            last_reason = "coincident vertices"
            continue

        edge_routes = {
            ekey: [(xs[point_id[k]], ys[point_id[k]]) for k in chain]
            for ekey, chain in edge_chain.items()
        }
        conflict = _drawing_conflict(vertex_positions, edge_routes)
        if conflict is not None:
            last_reason = conflict
            continue

        return DrawingResult(
            valid=True, vertex_positions=vertex_positions, edge_routes=edge_routes
        )

    return DrawingResult(valid=False, reason=last_reason)


def _drawing_conflict(
    vertex_positions: dict[int, tuple[int, int]],
    edge_routes: dict[tuple[int, int], list[tuple[int, int]]],
) -> Optional[str]:
    """Return a reason string if the integer drawing is not a clean orthogonal
    drawing (overlapping/crossing edges, or an edge through a non-endpoint
    vertex), else None."""
    # Flatten to axis-aligned segments tagged by their edge.
    segs: list[tuple[tuple[int, int], object]] = []  # ((x1,y1,x2,y2 normalized), edge)
    for ekey, route in edge_routes.items():
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            segs.append(((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), ekey))

    vertex_at = {p: v for v, p in vertex_positions.items()}

    def on_segment(px: int, py: int, s: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = s
        return x1 <= px <= x2 and y1 <= py <= y2

    # 1) Edge through a non-endpoint vertex.
    for route in edge_routes.values():
        endpoints = {route[0], route[-1]}
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            s = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            for p, _v in vertex_at.items():
                if p in endpoints:
                    continue
                if on_segment(p[0], p[1], s):
                    return "edge through vertex"

    # 2) Overlapping or crossing segments from different edges.
    for i in range(len(segs)):
        (ax1, ay1, ax2, ay2), ea = segs[i]
        a_horizontal = ay1 == ay2
        for j in range(i + 1, len(segs)):
            (bx1, by1, bx2, by2), eb = segs[j]
            if ea == eb:
                continue
            b_horizontal = by1 == by2
            if a_horizontal and b_horizontal and ay1 == by1:
                # Collinear horizontals: overlap if x-ranges share more than a point.
                if ax1 < bx2 and bx1 < ax2:
                    return "overlapping edges"
            elif (not a_horizontal) and (not b_horizontal) and ax1 == bx1:
                if ay1 < by2 and by1 < ay2:
                    return "overlapping edges"
            elif a_horizontal != b_horizontal:
                # One horizontal, one vertical: interior crossing.
                h = (ax1, ay1, ax2, ay2) if a_horizontal else (bx1, by1, bx2, by2)
                v = (bx1, by1, bx2, by2) if a_horizontal else (ax1, ay1, ax2, ay2)
                hx1, hy, hx2, _ = h
                vx, vy1, _, vy2 = v
                if hx1 < vx < hx2 and vy1 < hy < vy2:
                    return "crossing edges"
    return None


def face_turn_sum(face: Face, rep: OrthogonalRepresentation) -> Optional[int]:
    """Sum of quarter-turns around a face: +4 for bounded, -4 for the outer face.

    Combines corner turns ``2 - angle`` with the bend turns of each dart. Returns
    None if an angle is missing. Useful as a consistency check on the rep.
    """
    total = 0
    for a, b in face.edges:
        angle = rep.vertex_face_angles.get((b, face.index))
        if angle is None:
            return None
        total += 2 - angle
    for dart in face.edges:
        total += sum(rep.edge_bends.get(dart, []))
    return total

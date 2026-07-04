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

Scope: connected planar graphs of maximum degree 4, biconnected or not.
Bridges and cut vertices are handled by per-corner angles (angles keyed by the
incoming dart, since a face walk may visit a vertex more than once), and
degree-1 vertices put 360-degree corners on the walk, which rectangularization
splits with zero-min-length virtual darts. Vertices of degree > 4 are outside
this module's model and are handled upstream by vertex expansion into cages
(see :mod:`.expansion`). When a representation is not a realizable shape (e.g.
extracted from an infeasible flow, or a disconnected input),
``compute_orthogonal_shape`` returns ``valid=False`` and callers should fall
back to a heuristic router.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

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
    consistent orthogonal shape (e.g. a degenerate or inconsistent rep, or
    angles that do not sum to 360 degrees around every vertex); in that case
    callers should fall back to a heuristic router.
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

    # Pre-validate the angle assignment: every corner needs an angle in
    # 1..4 quarter turns, and the angles around each vertex must sum to a
    # full 360 degrees (4 units). This catches representations extracted
    # from infeasible flows (e.g. degree > 4) whose face turns can happen
    # to close up even though the assignment is geometric nonsense.
    vertex_angle_sum: dict[int, int] = {}
    for face in faces:
        for dart in face.edges:
            angle = rep.corner_angle(dart, face.index)
            if angle is None or not (1 <= angle <= 4):
                return ShapeResult(
                    valid=False, reason=f"missing or out-of-range angle at corner {dart}"
                )
            vertex_angle_sum[dart[1]] = vertex_angle_sum.get(dart[1], 0) + angle
    for v, total in vertex_angle_sum.items():
        if total != 4:
            return ShapeResult(valid=False, reason=f"angles at vertex {v} sum to {total} != 4")

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
            # cur = (a, b), nxt = (b, c); the corner at b is keyed by the
            # incoming dart cur (unique even when b borders the face several
            # times, as at cut vertices / bridges).
            angle = rep.corner_angle(cur, face.index)
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


# =============================================================================
# Rectangularization (turn-regularization for compaction)
# =============================================================================
#
# The coordinate assignment below solves two independent 1-D constraint systems
# built from the edges' own segments. That is provably sufficient only when
# every face is a rectangle: then any two segments needing separation are
# directly connected by boundary edges. For non-rectangular faces the classical
# fix (Tamassia; Di Battista et al., "Graph Drawing", ch. 5) is *refinement*:
# project every reflex corner of every bounded face onto the edge it faces,
# adding a dummy vertex and a dummy axis-parallel edge, until all faces are
# rectangles. The dummy edges are pure separation constraints (never drawn).
#
# The outer face needs the same treatment but cannot be refined directly (an
# unbounded region always keeps four excess reflex corners). It is handled by
# the classical enclosing rectangle: four dummy connector rays from boundary
# corners covering all four compass directions attach the boundary to a dummy
# surrounding rectangle, splitting the outer region into four bounded annulus
# faces, which are then refined like any other face.

# A point in the drawing: ("v", vertex), ("b", dart, i) for bends, or a dummy
# ("d", counter) introduced by rectangularization.
_PointKey = Any
_Dart = tuple[_PointKey, _PointKey, int]


def _signed_turn(d_in: int, d_out: int) -> int:
    """Turn at a corner: +1 convex (left), 0 flat, -1 reflex (right), -2 U-turn.

    A direction reversal is ambiguous (+2 or -2) from directions alone, but in
    this flow model corner angles are always in 1..4 quarter turns (0-degree
    Kandinsky angles do not occur), so a reversal is always a 360-degree corner
    -- the face walk turning around a degree-1 vertex -- with turn 2 - 4 = -2.
    """
    return {0: 0, 1: 1, 3: -1}.get((d_out - d_in) % 4, -2)


def _walk_turns(walk: list[_Dart]) -> list[int]:
    m = len(walk)
    return [_signed_turn(walk[(i - 1) % m][2], walk[i][2]) for i in range(m)]


def _build_face_walks(
    faces: list[Face],
    shape: ShapeResult,
    darts: dict[tuple[int, int], tuple[int, int]],
    edge_chain: dict[tuple[int, int], list[_PointKey]],
) -> Optional[tuple[list[list[_Dart]], Optional[list[_Dart]]]]:
    """Express every face as a cyclic walk of point-level darts.

    Each face dart (u, v) expands into its chain of vertex/bend points with the
    direction of every straight segment. Returns ``(bounded_walks, outer_walk)``
    or None if any walk violates the turn invariant (bounded +4, outer -4),
    which signals an out-of-domain input.
    """
    bounded: list[list[_Dart]] = []
    outer: Optional[list[_Dart]] = None
    for face in faces:
        walk: list[_Dart] = []
        for u, v in face.edges:
            key = (min(u, v), max(u, v))
            if key not in darts or (u, v) not in shape.edge_shapes:
                return None
            chain = edge_chain[key]
            if darts[key] != (u, v):
                chain = list(reversed(chain))
            seg_dirs = shape.edge_shapes[(u, v)].segments
            if len(seg_dirs) != len(chain) - 1:
                return None
            for i, d in enumerate(seg_dirs):
                walk.append((chain[i], chain[i + 1], d))
        if not walk:
            return None
        turns = _walk_turns(walk)
        total = sum(turns)
        if face.is_outer:
            if total != -4 or outer is not None:
                return None
            outer = walk
        else:
            if total != 4:
                return None
            bounded.append(walk)
    return bounded, outer


def _split_uturn_corners(
    walk: list[_Dart],
    counter: "itertools.count[int]",
    zero_pairs: set[tuple[_PointKey, _PointKey]],
) -> tuple[list[_Dart], list[_Dart]]:
    """Replace every 360-degree (U-turn) corner by two reflex corners.

    A face walk turns by -2 around a degree-1 vertex (the two sides of a
    pendant edge are the same face). Refinement only handles -1 corners, so
    each -2 corner at point P (incoming direction ``d``) is split by inserting
    a virtual zero-min-length dart (P, E) in direction ``d - 1``: the turns
    become d -> d-1 (-1) and d-1 -> d+2 (-1), preserving the walk's turn sum.
    The dart is emitted as a coordinate constraint of minimum length 0, so E
    coincides with P on the pendant edge's axis (perpendicular equality) and
    may only be pushed sideways when the dissection genuinely needs room.

    Returns the rewritten walk and the inserted virtual darts.
    """
    turns = _walk_turns(walk)
    if all(t != -2 for t in turns):
        return walk, []
    m = len(walk)
    new_walk: list[_Dart] = []
    eps_darts: list[_Dart] = []
    for i in range(m):
        if turns[i] == -2:
            p_pt = walk[i][0]
            d_in = walk[(i - 1) % m][2]
            e_pt: _PointKey = ("d", next(counter))
            eps = (p_pt, e_pt, (d_in + 3) % 4)
            eps_darts.append(eps)
            zero_pairs.add((p_pt, e_pt))
            new_walk.append(eps)
            new_walk.append((e_pt, walk[i][1], walk[i][2]))
        else:
            new_walk.append(walk[i])
    return new_walk, eps_darts


def _enclose_outer(
    outer_walk: list[_Dart], counter: "itertools.count[int]"
) -> Optional[tuple[list[list[_Dart]], list[_Dart]]]:
    """Surround the outer boundary with a dummy rectangle.

    Chooses four reflex corners of the outer walk at the first arrival times of
    the lifted rotation at -1..-4; their in-directions cover the four compass
    directions in clockwise order. From each, a dummy connector ray runs outward
    to an attachment point on the rectangle. Returns the four bounded annulus
    face walks and the dummy darts (connectors + rectangle sides), or None if
    the construction fails.
    """
    m = len(outer_walk)
    turns = _walk_turns(outer_walk)
    # First arrival of the lifted rotation at -1, -2, -3, -4 (each necessarily
    # reached via a -1 corner).
    t_idx: list[int] = []
    r = 0
    target = -1
    for i in range(m):
        r += turns[i]
        if r == target:
            t_idx.append(i)
            target -= 1
            if target < -4:
                break
    if len(t_idx) != 4:
        return None
    corners = [outer_walk[t][0] for t in t_idx]
    dirs = [outer_walk[(t - 1) % m][2] for t in t_idx]  # in-directions, clockwise
    if len(set(dirs)) != 4 or len(set(corners)) != 4:
        return None

    attach: list[_PointKey] = [("d", next(counter)) for _ in range(4)]
    rect: list[_PointKey] = [("d", next(counter)) for _ in range(4)]
    dummy: list[_Dart] = []
    for i in range(4):
        dummy.append((corners[i], attach[i], dirs[i]))  # connector ray
        dummy.append((attach[(i + 1) % 4], rect[i], dirs[i]))  # rectangle side
        dummy.append((rect[i], attach[i], (dirs[i] + 1) % 4))  # rectangle side

    annulus: list[list[_Dart]] = []
    for i in range(4):
        j = (i + 1) % 4
        walk: list[_Dart] = [(attach[i], corners[i], (dirs[i] + 2) % 4)]
        k = t_idx[i]
        while k != t_idx[j]:
            walk.append(outer_walk[k])
            k = (k + 1) % m
        walk.append((corners[j], attach[j], dirs[j]))
        walk.append((attach[j], rect[i], dirs[i]))
        walk.append((rect[i], attach[i], (dirs[i] + 1) % 4))
        if sum(_walk_turns(walk)) != 4:
            return None
        annulus.append(walk)
    return annulus, dummy


def _refine_to_rectangles(
    walks: list[list[_Dart]],
    counter: "itertools.count[int]",
    zero_pairs: Optional[set[tuple[_PointKey, _PointKey]]] = None,
) -> Optional[list[_Dart]]:
    """Dissect bounded faces until none has a reflex corner.

    Each reflex corner P is projected onto its *front* dart -- the first dart
    after P where the cumulative turn reaches +2 -- splitting that dart at a new
    dummy point W and adding the dummy dart (P, W) in the direction of the edge
    entering P. Every refinement removes one reflex corner and introduces none,
    so this terminates with every face a rectangle. Returns the dummy darts
    (segment constraints), or None on any anomaly.

    ``zero_pairs`` marks darts with minimum length 0 (the virtual U-turn split
    darts); splitting such a dart propagates the marker to both halves so the
    dissection never forces a pendant edge apart.
    """
    if zero_pairs is None:
        zero_pairs = set()
    extra: list[_Dart] = []
    stack = [w for w in walks]
    guard = sum(len(w) for w in walks) * 8 + 64
    while stack:
        guard -= 1
        if guard < 0:
            return None
        walk = stack.pop()
        m = len(walk)
        if m < 4:
            continue
        turns = _walk_turns(walk)
        if any(t == -2 for t in turns):
            return None  # U-turns must have been split before refinement
        if -1 not in turns:
            continue  # turn-regular (rectangle up to flat corners): done
        k = turns.index(-1)
        reflex_point = walk[k][0]
        d_in = walk[(k - 1) % m][2]

        # Front dart: first dart after the corner with cumulative turn +2.
        r = 0
        front = None
        for step in range(1, m):
            j = (k + step) % m
            r += turns[j]
            if r == 2:
                front = j
                break
        if front is None or front == k or front == (k - 1) % m:
            return None

        a_pt, b_pt, front_dir = walk[front]
        w_pt: _PointKey = ("d", next(counter))
        if (a_pt, b_pt) in zero_pairs:
            zero_pairs.add((a_pt, w_pt))
            zero_pairs.add((w_pt, b_pt))
        extra.append((a_pt, w_pt, front_dir))
        extra.append((w_pt, b_pt, front_dir))
        extra.append((reflex_point, w_pt, d_in))

        def _cyc(lo: int, hi: int) -> list[_Dart]:
            out: list[_Dart] = []
            i = lo
            while i != hi:
                out.append(walk[i])
                i = (i + 1) % m
            return out

        first = [(reflex_point, w_pt, d_in), (w_pt, b_pt, front_dir)]
        second = [(a_pt, w_pt, front_dir), (w_pt, reflex_point, (d_in + 2) % 4)]
        stack.append(first + _cyc((front + 1) % m, k))
        stack.append(_cyc(k, front) + second)
    return extra


def _rectangularize(
    faces: list[Face],
    shape: ShapeResult,
    darts: dict[tuple[int, int], tuple[int, int]],
    edge_chain: dict[tuple[int, int], list[_PointKey]],
) -> Optional[tuple[list[_Dart], set[tuple[_PointKey, _PointKey]]]]:
    """Compute the dummy separation darts that make every face a rectangle.

    Returns the full list of dummy darts (enclosing rectangle + refinement
    projections + virtual U-turn splits) to feed into the axis constraint
    systems, together with the set of darts that carry minimum length 0 (the
    U-turn splits), or None if the input is out of domain -- callers then fall
    back to the unrefined assignment.
    """
    built = _build_face_walks(faces, shape, darts, edge_chain)
    if built is None:
        return None
    bounded, outer = built
    counter = itertools.count()
    zero_pairs: set[tuple[_PointKey, _PointKey]] = set()
    dummy: list[_Dart] = []

    # Split 360-degree corners (pendant edges) before enclosure / refinement,
    # which only handle turns in {-1, 0, +1}.
    split_bounded: list[list[_Dart]] = []
    for walk in bounded:
        new_walk, eps_darts = _split_uturn_corners(walk, counter, zero_pairs)
        dummy.extend(eps_darts)
        split_bounded.append(new_walk)
    bounded = split_bounded
    if outer is not None:
        outer, eps_darts = _split_uturn_corners(outer, counter, zero_pairs)
        dummy.extend(eps_darts)

    if outer is not None:
        enclosed = _enclose_outer(outer, counter)
        if enclosed is None:
            return None
        annulus, rect_darts = enclosed
        bounded = bounded + annulus
        dummy.extend(rect_darts)
    refined = _refine_to_rectangles(bounded, counter, zero_pairs)
    if refined is None:
        return None
    dummy.extend(refined)
    return dummy, zero_pairs


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
    segments: list[tuple[int, int, int, int]],
    horizontal: bool,
    spread: bool = False,
) -> Optional[dict[int, int]]:
    """Assign one integer coordinate per point.

    ``segments`` entries are ``(a, b, direction, min_length)``. For the x axis
    (``horizontal=True``): vertical segments (N/S) force equal x on their
    endpoints; horizontal segments (E/W) order the resulting classes (East
    increases x) with the given minimum length -- 1 for real and dissection
    segments, 0 for the virtual U-turn split darts, which only pin a dummy
    point to its pendant edge without forcing separation. Returns None on a
    contradiction (two points forced equal yet strictly ordered) or a cycle
    (not possible for a valid orthogonal shape).

    ``spread=False`` packs coordinates with longest-path (compact). ``spread=True``
    gives every class a distinct coordinate by topological rank, which separates
    independent classes that longest-path would collapse to the same value -- a
    cheap partial substitute for face rectangularization.
    """
    equal_dirs = (NORTH, SOUTH) if horizontal else (EAST, WEST)
    order_dirs = (EAST, WEST) if horizontal else (NORTH, SOUTH)
    increasing = EAST if horizontal else NORTH

    uf = _UnionFind(n_points)
    for a, b, d, _w in segments:
        if d in equal_dirs:
            uf.union(a, b)

    classes = {uf.find(i) for i in range(n_points)}
    adj: dict[int, dict[int, int]] = {c: {} for c in classes}  # lo -> {hi: min_length}
    indeg: dict[int, int] = {c: 0 for c in classes}
    for a, b, d, w in segments:
        if d not in order_dirs:
            continue
        ca, cb = uf.find(a), uf.find(b)
        # Edge points from the smaller coordinate to the larger.
        lo, hi = (ca, cb) if d == increasing else (cb, ca)
        if lo == hi:
            if w > 0:
                return None  # equal class but strictly ordered -> contradiction
            continue  # zero-min-length within one class: trivially satisfied
        if hi not in adj[lo]:
            adj[lo][hi] = w
            indeg[hi] += 1
        else:
            adj[lo][hi] = max(adj[lo][hi], w)

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
        for nb, w in adj[c].items():
            if coord[nb] < coord[c] + w:
                coord[nb] = coord[c] + w
            indeg[nb] -= 1
            if indeg[nb] == 0:
                queue.append(nb)
    if seen != len(classes):
        return None  # cycle
    return {i: coord[uf.find(i)] for i in range(n_points)}


def compute_coordinates(
    shape: ShapeResult,
    edges: list[tuple[int, int]],
    faces: Optional[list[Face]] = None,
) -> DrawingResult:
    """Assign integer coordinates realizing ``shape``.

    Builds a point per vertex and per bend, records each segment's direction,
    and solves the two axes independently by longest-path in the horizontal and
    vertical constraint graphs (minimum segment length 1). The result is an
    orthogonal drawing in which every segment is axis-aligned in its shape
    direction.

    When ``faces`` is provided, the faces are first *rectangularized* (classical
    turn-regularization): every bounded face is dissected into rectangles and
    the outer face is enclosed in a dummy rectangle, yielding dummy separation
    constraints under which the per-edge constraint graphs are provably
    sufficient for a planar drawing. Assignments are tried in order --
    rectangularized compact, rectangularized spread, then the unrefined compact
    and spread fallbacks -- and the first clean one is returned. If none is
    clean the result is invalid so callers fall back to the heuristic router.
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

    segments: list[tuple[int, int, int, int]] = []
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
            segments.append((pid(chain[i]), pid(chain[i + 1]), direction, 1))
        edge_chain[ekey] = chain

    # Rectangularization: dummy separation constraints that make the per-edge
    # constraint graphs sufficient (see _rectangularize). Dummy points receive
    # point ids too but are never read back into the drawing. Virtual U-turn
    # split darts carry minimum length 0.
    rect_segments: Optional[list[tuple[int, int, int, int]]] = None
    if faces is not None:
        rectangularized = _rectangularize(faces, shape, darts, edge_chain)
        if rectangularized is not None:
            dummy_darts, zero_pairs = rectangularized
            rect_segments = [
                (pid(a), pid(b), d, 0 if (a, b) in zero_pairs else 1) for a, b, d in dummy_darts
            ]

    n_points = len(point_id)

    attempts: list[tuple[bool, list[tuple[int, int, int, int]]]] = []
    if rect_segments is not None:
        attempts.append((False, segments + rect_segments))
        attempts.append((True, segments + rect_segments))
    attempts.append((False, segments))
    attempts.append((True, segments))

    last_reason = "contradictory coordinate constraints"
    for spread, segs in attempts:
        xs = _assign_axis(n_points, segs, horizontal=True, spread=spread)
        ys = _assign_axis(n_points, segs, horizontal=False, spread=spread)
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

        return DrawingResult(valid=True, vertex_positions=vertex_positions, edge_routes=edge_routes)

    return DrawingResult(valid=False, reason=last_reason)


def _drawing_conflict(
    vertex_positions: dict[int, tuple[int, int]],
    edge_routes: dict[tuple[int, int], list[tuple[int, int]]],
) -> Optional[str]:
    """Return a reason string if the integer drawing is not a clean orthogonal
    drawing (overlapping/crossing edges, or an edge through a non-endpoint
    vertex), else None."""
    # Flatten to axis-aligned segments tagged by their edge.
    segs: list[tuple[tuple[int, int, int, int], object]] = []  # ((x1,y1,x2,y2 normalized), edge)
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
        angle = rep.corner_angle((a, b), face.index)
        if angle is None:
            return None
        total += 2 - angle
    for dart in face.edges:
        total += sum(rep.edge_bends.get(dart, []))
    return total

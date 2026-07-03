"""Tests for the Topology-Shape-Metrics shape stage (orthogonal/metrics.py).

The shape stage turns an orthogonal representation (angles + bends) into compass
directions for every edge segment. These tests exercise it on graphs whose
representation is a valid orthogonal shape (simple cycles) and confirm it
*detects* representations that are not realizable (out-of-domain inputs -- see
the module-level note and docs/rectangularization-plan.md).
"""

from __future__ import annotations

from graph_layout import check_planarity
from graph_layout.orthogonal.metrics import (
    DIRECTION_STEP,
    compute_coordinates,
    compute_orthogonal_shape,
    face_turn_sum,
)
from graph_layout.orthogonal.orthogonalization import (
    build_flow_network,
    compute_faces,
    flow_to_orthogonal_rep,
    solve_min_cost_flow_simple,
)
from graph_layout.planarity import MaxFaceEmbedder


def _drawing_of(n, edges):
    _faces, _rep, shape = _shape_of(n, edges)
    return shape, compute_coordinates(shape, edges)


def _shape_of(n, edges):
    pr = check_planarity(n, edges)
    assert pr.is_planar
    emb = MaxFaceEmbedder().embed(n, edges, planarity_result=pr)
    faces = compute_faces(n, edges, embedding=emb)
    net = build_flow_network(n, edges, faces)
    solve_min_cost_flow_simple(net)
    rep = flow_to_orthogonal_rep(net, edges)
    return faces, rep, compute_orthogonal_shape(faces, rep)


# ---------------------------------------------------------------------------
# Valid shapes (simple cycles draw as rectangles with no bends)
# ---------------------------------------------------------------------------


def test_square_shape_is_valid_rectangle():
    faces, rep, shape = _shape_of(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert shape.valid

    # Every bounded face turns +4, the outer face -4.
    sums = sorted(face_turn_sum(f, rep) for f in faces)
    assert sums == [-4, 4]

    # No bends -> each dart is a single axis-aligned segment.
    for es in shape.edge_shapes.values():
        assert len(es.segments) == 1
        assert es.segments[0] in DIRECTION_STEP

    # The four directed edges around the inner face use all four directions.
    inner = next(f for f in faces if not f.is_outer)
    dirs = {shape.edge_shapes[d].start_direction for d in inner.edges}
    assert dirs == {0, 1, 2, 3}


def test_reverse_darts_are_opposite():
    _faces, _rep, shape = _shape_of(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert shape.valid
    for (u, v), es in shape.edge_shapes.items():
        rev = shape.edge_shapes.get((v, u))
        assert rev is not None
        # The direction a dart arrives at its head is opposite to the direction
        # its reverse leaves that same vertex.
        assert es.end_direction == (rev.start_direction + 2) % 4


def test_valid_shape_directions_are_orthogonal_turns():
    """Consecutive segments of a dart differ by a single quarter turn."""
    _faces, _rep, shape = _shape_of(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert shape.valid
    for es in shape.edge_shapes.values():
        for a, b in zip(es.segments, es.segments[1:]):
            assert (b - a) % 4 in (1, 3)  # +/-1 quarter turn (a bend)


# ---------------------------------------------------------------------------
# The flow model emits valid representations for graphs that require bends
# ---------------------------------------------------------------------------


def test_representation_is_valid_for_graphs_needing_bends():
    """Graphs with triangular faces (which force bends) now yield valid shapes.

    Every triangular face must turn by +/-4 quarter-turns. Before the flow-model
    fix the bend-to-face sign was unaligned with the embedding, so K4 came out
    with face turn-sums like [0, 0, 4, 4] and no valid shape existed. With bends
    attributed to the correct dart/side, every face turns +/-4.
    """
    for name, n, edges in [
        ("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        ("wheel5", 5, [(1, 2), (2, 3), (3, 4), (4, 1), (0, 1), (0, 2), (0, 3), (0, 4)]),
        ("prism", 6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
    ]:
        faces, rep, shape = _shape_of(n, edges)
        turn_sums = [face_turn_sum(f, rep) for f in faces]
        assert all(t in (4, -4) for t in turn_sums), f"{name}: {turn_sums}"
        assert shape.valid, name
        # Exactly one outer face turns -4; all bounded faces turn +4.
        assert turn_sums.count(-4) == 1, f"{name}: {turn_sums}"


def _grid(w: int, h: int) -> tuple[int, list[tuple[int, int]]]:
    """A w x h grid graph: biconnected, planar, max degree 4."""
    edges: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            v = r * w + c
            if c + 1 < w:
                edges.add((v, v + 1))
            if r + 1 < h:
                edges.add((v, v + w))
    return w * h, sorted(edges)


def test_all_faces_turn_pm4_for_biconnected_maxdeg4_grids():
    """The flow model emits valid representations across the model's domain.

    Standard orthogonalization is defined for biconnected planar graphs of max
    degree <= 4; grids are a clean deterministic family of them. Every face must
    turn +/-4 and the shape must be realizable. (Cut vertices / bridges and
    degree > 4 are separate known limitations -- see
    docs/rectangularization-plan.md.)
    """
    for w in range(2, 6):
        for h in range(2, 6):
            n, edges = _grid(w, h)
            faces, rep, shape = _shape_of(n, edges)
            turn_sums = [face_turn_sum(f, rep) for f in faces]
            assert all(t in (4, -4) for t in turn_sums), f"grid {w}x{h}: {turn_sums}"
            assert shape.valid, f"grid {w}x{h}"
            assert turn_sums.count(-4) == 1, f"grid {w}x{h}: {turn_sums}"


# ---------------------------------------------------------------------------
# Detection of representations that are not valid orthogonal shapes
# ---------------------------------------------------------------------------


def test_inconsistent_representation_is_detected():
    """A representation whose turns do not close up is reported invalid.

    Constructed by corrupting a valid square's angles so a face no longer turns
    +4. The shape stage must return ``valid=False`` so callers fall back.
    """
    faces, rep, shape = _shape_of(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert shape.valid  # sanity: valid before corruption

    # Break one corner angle so the inner face no longer turns +4.
    inner = next(f for f in faces if not f.is_outer)
    corner = inner.edges[0][1]
    rep.vertex_face_angles[(corner, inner.index)] = 3  # was 1

    corrupted = compute_orthogonal_shape(faces, rep)
    assert not corrupted.valid


def test_empty_faces_is_invalid():
    from graph_layout.orthogonal.orthogonalization import OrthogonalRepresentation

    shape = compute_orthogonal_shape([], OrthogonalRepresentation())
    assert not shape.valid


# ---------------------------------------------------------------------------
# Coordinate assignment (stage 2): shape -> integer coordinates
# ---------------------------------------------------------------------------


def _assert_axis_aligned(drawing):
    """Every segment of every edge route changes exactly one axis by >= 1."""
    for route in drawing.edge_routes.values():
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            assert (x1 == x2) ^ (y1 == y2), f"non-orthogonal segment {(x1, y1)}->{(x2, y2)}"
            assert abs(x1 - x2) + abs(y1 - y2) >= 1


def test_square_draws_as_unit_rectangle():
    _shape, drawing = _drawing_of(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert drawing.valid
    _assert_axis_aligned(drawing)

    # Four distinct corners spanning a 1x1 box, no bends.
    coords = set(drawing.vertex_positions.values())
    assert len(coords) == 4
    xs = {x for x, _ in coords}
    ys = {y for _, y in coords}
    assert xs == {0, 1} and ys == {0, 1}
    for route in drawing.edge_routes.values():
        assert len(route) == 2  # no bend points


def test_grids_draw_axis_aligned_with_distinct_vertices():
    # Not every grid embedding packs cleanly without rectangularization; those
    # are correctly flagged invalid (callers fall back). Every drawing reported
    # valid must be axis-aligned with distinct vertices, and at least some grids
    # draw cleanly.
    valid_count = 0
    for w in range(2, 5):
        for h in range(2, 5):
            n, edges = _grid(w, h)
            _shape, drawing = _drawing_of(n, edges)
            if not drawing.valid:
                continue
            valid_count += 1
            _assert_axis_aligned(drawing)
            coords = list(drawing.vertex_positions.values())
            assert len(set(coords)) == len(coords), f"grid {w}x{h}: overlapping vertices"
            assert len(coords) == n
    assert valid_count >= 1


def test_bent_graph_draws_orthogonally():
    """K4 requires bends; the drawing must realize them as orthogonal segments."""
    _shape, drawing = _drawing_of(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert drawing.valid
    _assert_axis_aligned(drawing)
    assert len(set(drawing.vertex_positions.values())) == 4  # distinct vertices
    # At least one edge is routed with a bend.
    assert any(len(route) > 2 for route in drawing.edge_routes.values())


def test_invalid_shape_yields_invalid_drawing():
    # Degree > 4 (star K1,5 center) is out of the model's domain -> invalid shape.
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    _shape, drawing = _drawing_of(6, edges)
    assert not drawing.valid


def _drawing_has_conflict(drawing) -> bool:
    """Independent (cell-based) overlap / through-vertex check for validation."""
    occupied: dict[tuple, tuple[int, int]] = {}
    vertex_at = {p: v for v, p in drawing.vertex_positions.items()}
    for ekey, route in drawing.edge_routes.items():
        endpoints = {route[0], route[-1]}
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            # unit cells along the segment
            if x1 == x2:
                lo, hi = sorted((y1, y2))
                cells = [("v", x1, y) for y in range(lo, hi)]
                pts = [(x1, y) for y in range(lo, hi + 1)]
            else:
                lo, hi = sorted((x1, x2))
                cells = [("h", x, y1) for x in range(lo, hi)]
                pts = [(x, y1) for x in range(lo, hi + 1)]
            for c in cells:
                if c in occupied and occupied[c] != ekey:
                    return True
                occupied[c] = ekey
            for p in pts:
                if p in vertex_at and p not in endpoints:
                    return True
    return False


def test_conflicting_drawing_is_flagged_for_fallback():
    """A biconnected max-degree-4 graph whose coordinate assignment would cross
    must be reported invalid so callers fall back (full face rectangularization
    would resolve it). Neither the compact nor the spread assignment clears it."""
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (6, 7)]
    _shape, drawing = _drawing_of(8, edges)
    assert not drawing.valid
    assert drawing.reason  # a specific reason (overlap / crossing / through vertex)


def test_valid_drawings_are_conflict_free():
    """Safety invariant: any drawing reported valid is genuinely clean.

    Uses an independent overlap check (not the internal detector) over grids and
    named graphs.
    """
    graphs = [_grid(w, h) for w in range(2, 5) for h in range(2, 5)]
    graphs += [
        (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),  # K4
        (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),  # prism
        (
            8,
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
        ),  # cube
    ]
    for n, edges in graphs:
        _shape, drawing = _drawing_of(n, edges)
        if drawing.valid:
            assert not _drawing_has_conflict(drawing), f"n={n} valid but has conflict"
